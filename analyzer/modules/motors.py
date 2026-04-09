"""
Motor & ESC Analysis Module
============================
Analyses RCOU (RC Output) messages to assess motor health and balance.

For multirotors the motor outputs must be symmetric under steady hover —
any persistent imbalance indicates a mechanical issue (bent prop, worn
bearing, frame damage) or a miscalibrated ESC.

Checks performed
----------------
1.  Output symmetry — std-dev of motor outputs at each sample; high = imbalance
2.  Throttle headroom — average throttle as % of max; high = overloaded drone
3.  Output oscillation — high-frequency jitter in motor commands = PID tuning issue
4.  Motor saturation — any motor hitting max PWM = loss of control authority
5.  Dead motor detection — any channel stuck at idle during flight
6.  Hover throttle comparison — actual vs expected from profile

Scoring deductions (from 100)
------------------------------
-5  to -20   Poor output symmetry (per imbalance level)
-5  to -15   Low throttle headroom (consistently >80% throttle)
-10 to -15   High oscillation index
-20          Motor saturation events
-30          Dead / stuck motor detected
-5           Hover throttle significantly differs from expected
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)


def analyse(result: ParseResult, profile: DroneProfile, arm_time: float = 0.0) -> Dict[str, Any]:
    """
    Run motor/ESC analysis.

    Returns standardised result dict.
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    rcou_df = result.get("RCOU")
    if rcou_df is None or len(rcou_df) < 10:
        return {
            "score": 50.0,
            "grade": "C",
            "available": False,
            "issues": [_issue("warning", "MOT-000",
                              "No motor output data (RCOU) found in log.")],
            "metrics": {},
            "summary": "Motor output data not available in this log.",
        }

    # Filter to armed window
    flight_rcou = rcou_df[rcou_df["timestamp"] >= arm_time].copy()
    if len(flight_rcou) < 10:
        flight_rcou = rcou_df.copy()

    # ── Identify motor channels ───────────────────────────────────────────────
    motor_cols = _get_motor_columns(flight_rcou, profile)
    if not motor_cols:
        return {
            "score": 40.0,
            "grade": "D",
            "available": False,
            "issues": [_issue("warning", "MOT-001",
                              "Could not identify motor output channels in RCOU data.")],
            "metrics": {},
            "summary": "Motor channel identification failed.",
        }

    metrics["motor_channels"] = motor_cols
    metrics["motor_count"] = len(motor_cols)

    is_fixed_wing = profile.type == "fixed_wing"
    is_vtol       = profile.type == "vtol"

    # For fixed-wing with a single throttle channel — use simplified analysis
    if is_fixed_wing and len(motor_cols) == 1:
        return _analyse_fixed_wing_throttle(flight_rcou, motor_cols, profile, metrics)

    # For VTOL: run a separate pusher motor check and exclude the pusher channel
    # from the lift motor symmetry/headroom analysis to prevent false imbalance.
    pusher_issues: List[Dict] = []
    if is_vtol and profile.pusher_motor is not None:
        pusher_col = f"C{profile.pusher_motor.channel}"
        if pusher_col in motor_cols:
            motor_cols = [c for c in motor_cols if c != pusher_col]
            logger.info("Excluded pusher motor %s from lift-motor symmetry analysis.", pusher_col)
        pusher_result = _analyse_pusher_throttle(flight_rcou, pusher_col, profile, metrics)
        pusher_issues = pusher_result["issues"]
        metrics.update(pusher_result["metrics"])

    # Extract motor PWM matrix — shape (N_samples, N_motors)
    pwm_matrix = flight_rcou[motor_cols].values.astype(float)

    # Convert PWM to throttle percentage  0-100
    min_pwm = profile.motors.min_pwm
    max_pwm = profile.motors.max_pwm
    pwm_range = max_pwm - min_pwm
    throttle_matrix = np.clip((pwm_matrix - min_pwm) / pwm_range * 100.0, 0.0, 100.0)

    # ── Filter for actual flight samples (at least one motor > 20%) ──────────
    in_flight_mask = np.max(throttle_matrix, axis=1) > 20.0
    flight_throttle = throttle_matrix[in_flight_mask]

    if len(flight_throttle) < 10:
        return {
            "score": 60.0,
            "grade": "C",
            "available": True,
            "issues": [_issue("info", "MOT-002",
                              "Very few airborne motor output samples found.")],
            "metrics": metrics,
            "summary": "Insufficient in-flight motor data for full analysis.",
        }

    # ── 1. Output symmetry ────────────────────────────────────────────────────
    sym_result = _symmetry_analysis(flight_throttle, motor_cols, profile)
    metrics.update(sym_result["metrics"])
    issues.extend(sym_result["issues"])

    # ── 2. Throttle headroom ──────────────────────────────────────────────────
    headroom_result = _throttle_headroom_analysis(flight_throttle, profile)
    metrics.update(headroom_result["metrics"])
    issues.extend(headroom_result["issues"])

    # ── 3. Output oscillation ─────────────────────────────────────────────────
    osc_result = _oscillation_analysis(flight_throttle, flight_rcou, motor_cols)
    metrics.update(osc_result["metrics"])
    issues.extend(osc_result["issues"])

    # ── 4. Motor saturation ───────────────────────────────────────────────────
    sat_result = _saturation_analysis(flight_throttle, motor_cols)
    metrics.update(sat_result["metrics"])
    issues.extend(sat_result["issues"])

    # ── 5. Dead motor detection ───────────────────────────────────────────────
    dead_result = _dead_motor_analysis(flight_throttle, motor_cols)
    metrics.update(dead_result["metrics"])
    issues.extend(dead_result["issues"])

    # ── 6. Per-motor statistics ───────────────────────────────────────────────
    per_motor_stats = []
    for i, col in enumerate(motor_cols):
        col_data = flight_throttle[:, i]
        per_motor_stats.append({
            "channel": col,
            "avg_throttle_pct": round(float(np.mean(col_data)), 2),
            "max_throttle_pct": round(float(np.max(col_data)), 2),
            "min_throttle_pct": round(float(np.min(col_data)), 2),
            "std_throttle_pct": round(float(np.std(col_data)), 2),
        })
    metrics["per_motor"] = per_motor_stats

    # ── Score ─────────────────────────────────────────────────────────────────
    all_issues = issues + pusher_issues
    score = _compute_score(metrics, all_issues)
    grade = _grade(score)

    avg_throttle = metrics.get("avg_throttle_pct", 0.0)
    sym_score = metrics.get("symmetry_score", 100.0)
    lift_count = len(motor_cols)
    summary = (
        f"Motors: {lift_count}  |  "
        f"Avg throttle: {avg_throttle:.1f}%  |  "
        f"Symmetry score: {sym_score:.0f}/100"
    )

    return {
        "score": round(score, 1),
        "grade": grade,
        "available": True,
        "issues": all_issues,
        "metrics": metrics,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-wing throttle analysis (single engine — no symmetry check)
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_fixed_wing_throttle(
    flight_rcou: pd.DataFrame,
    motor_cols: List[str],
    profile: DroneProfile,
    metrics: Dict,
) -> Dict[str, Any]:
    """
    Simplified motor analysis for fixed-wing aircraft with a single throttle channel.
    Checks: throttle headroom, saturation, stuck/dead throttle.
    Symmetry analysis is skipped — not applicable for single-engine aircraft.
    """
    issues: List[Dict] = []

    min_pwm   = profile.motors.min_pwm
    max_pwm   = profile.motors.max_pwm
    pwm_range = max_pwm - min_pwm

    col = motor_cols[0]
    pwm_vals   = flight_rcou[col].values.astype(float)
    throttle   = np.clip((pwm_vals - min_pwm) / pwm_range * 100.0, 0.0, 100.0)

    # Filter to airborne samples (throttle > 10%)
    airborne = throttle[throttle > 10.0]
    if len(airborne) < 10:
        return {
            "score": 60.0, "grade": "C", "available": True,
            "issues": [_issue("info", "MOT-002", "Very few airborne throttle samples found.")],
            "metrics": metrics,
            "summary": "Insufficient in-flight throttle data.",
        }

    avg_thr    = float(np.mean(airborne))
    max_thr    = float(np.max(airborne))
    median_thr = float(np.median(airborne))
    p95_thr    = float(np.percentile(airborne, 95))

    metrics.update({
        "avg_throttle_pct":    round(avg_thr,    2),
        "max_throttle_pct":    round(max_thr,    2),
        "median_throttle_pct": round(median_thr, 2),
        "p95_throttle_pct":    round(p95_thr,    2),
        "symmetry_score":      None,   # not applicable
    })

    warn_thr = profile.motors.high_throttle_warn_pct
    crit_thr = profile.motors.high_throttle_critical_pct

    frac_high = float(np.mean(airborne > warn_thr))
    metrics["frac_time_high_throttle"] = round(frac_high, 3)

    # Throttle headroom check
    if frac_high > 0.30:
        sev = "critical" if avg_thr > crit_thr else "warning"
        issues.append(_issue(
            sev, "MOT-006",
            f"Engine throttle high: {frac_high*100:.0f}% of flight above {warn_thr:.0f}% "
            f"(avg: {avg_thr:.1f}%). Check propeller pitch, payload weight, or engine health.",
            value=round(avg_thr, 2), threshold=warn_thr,
        ))

    # Saturation check
    sat_frac = float(np.mean(airborne >= 99.0))
    metrics["overall_saturation_frac"] = round(sat_frac, 4)
    sat_warn_pct = profile.get_threshold("mot_sat_warn_pct", 5.0)
    sat_crit_pct = profile.get_threshold("mot_sat_crit_pct", 15.0)
    if sat_frac * 100 > sat_crit_pct:
        issues.append(_issue(
            "critical", "MOT-008",
            f"Engine at full throttle {sat_frac*100:.1f}% of flight — insufficient thrust margin. "
            f"This can indicate an underpowered motor/prop combination, excessive headwind, "
            f"or sustained climbs. Cross-check with wind and altitude data.",
            value=round(sat_frac * 100, 2), threshold=sat_crit_pct,
        ))
    elif sat_frac * 100 > sat_warn_pct:
        issues.append(_issue(
            "warning", "MOT-008",
            f"Engine at full throttle {sat_frac*100:.1f}% of flight — limited thrust margin. "
            f"Brief throttle saturation during climbs or headwind gusts is normal.",
            value=round(sat_frac * 100, 2), threshold=sat_warn_pct,
        ))

    # Dead throttle check (engine never responding)
    if max_thr < 15.0:
        issues.append(_issue(
            "critical", "MOT-009",
            f"Throttle channel {col} never exceeded 15% during flight — possible engine failure.",
            value=round(max_thr, 2), threshold=15.0,
        ))
        metrics["dead_motor_count"] = 1
    else:
        metrics["dead_motor_count"] = 0

    score = 100.0
    if avg_thr > crit_thr:
        score -= 15.0
    elif avg_thr > warn_thr:
        score -= 8.0
    sat_crit_pct = profile.get_threshold("mot_sat_crit_pct", 15.0)
    sat_warn_pct = profile.get_threshold("mot_sat_warn_pct", 5.0)
    if sat_frac * 100 > sat_crit_pct:
        score -= 20.0
    elif sat_frac * 100 > sat_warn_pct:
        score -= 10.0
    # Dead engine — not captured by the throttle/saturation metrics above
    if metrics.get("dead_motor_count", 0) > 0:
        score -= 25.0
    score = max(0.0, min(100.0, score))

    from analyzer.scoring.engine import _grade
    grade = _grade(score)

    summary = (
        f"Fixed-wing throttle  |  "
        f"Avg: {avg_thr:.1f}%  |  "
        f"Peak: {max_thr:.1f}%  |  "
        f"High-throttle time: {frac_high*100:.0f}%"
    )

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pusher / cruise motor analysis (VTOL QuadPlane)
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_pusher_throttle(
    flight_rcou: pd.DataFrame,
    pusher_col: str,
    profile: DroneProfile,
    metrics: Dict,
) -> Dict[str, Any]:
    """
    Analyse the pusher/cruise motor channel on a VTOL QuadPlane separately
    from the lift motors.  Uses profile.pusher_motor thresholds.
    Returns a dict with only 'issues' and 'metrics' keys (not a full result).
    """
    issues: List[Dict] = []
    pm = profile.pusher_motor  # guaranteed non-None by caller

    if pusher_col not in flight_rcou.columns:
        return {"issues": [], "metrics": {}}

    min_pwm   = pm.min_pwm
    max_pwm   = pm.max_pwm
    pwm_range = max_pwm - min_pwm
    if pwm_range <= 0:
        return {"issues": [], "metrics": {}}

    pwm_vals = flight_rcou[pusher_col].values.astype(float)
    throttle = np.clip((pwm_vals - min_pwm) / pwm_range * 100.0, 0.0, 100.0)

    # Only analyse samples where the pusher is actually running (> 10%)
    active = throttle[throttle > 10.0]
    if len(active) < 10:
        metrics["pusher_active_pct"] = 0.0
        return {"issues": [], "metrics": metrics}

    avg_thr  = float(np.mean(active))
    max_thr  = float(np.max(active))
    p95_thr  = float(np.percentile(active, 95))
    sat_frac = float(np.mean(active >= 99.0))
    active_pct = float(len(active) / len(throttle) * 100.0)

    metrics["pusher_avg_throttle_pct"] = round(avg_thr, 2)
    metrics["pusher_max_throttle_pct"] = round(max_thr, 2)
    metrics["pusher_p95_throttle_pct"] = round(p95_thr, 2)
    metrics["pusher_saturation_frac"]  = round(sat_frac, 4)
    metrics["pusher_active_pct"]       = round(active_pct, 1)

    warn_thr = pm.high_throttle_warn_pct
    crit_thr = pm.high_throttle_critical_pct
    frac_high = float(np.mean(active > warn_thr))

    if frac_high > 0.30:
        sev = "critical" if avg_thr > crit_thr else "warning"
        issues.append(_issue(
            sev, "MOT-010",
            f"Pusher/cruise motor ({pusher_col}) throttle high: "
            f"{frac_high*100:.0f}% of cruise above {warn_thr:.0f}% "
            f"(avg: {avg_thr:.1f}%). Check propeller pitch, airspeed, or motor health.",
            value=round(avg_thr, 2), threshold=warn_thr,
        ))

    if sat_frac * 100 > 5.0:
        sev = "critical" if sat_frac * 100 > 15.0 else "warning"
        issues.append(_issue(
            sev, "MOT-011",
            f"Pusher motor ({pusher_col}) at full throttle {sat_frac*100:.1f}% of cruise time "
            "— insufficient thrust margin. Check airframe drag, headwinds, or motor condition.",
            value=round(sat_frac * 100, 2), threshold=5.0,
        ))

    return {"issues": issues, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis sub-functions
# ─────────────────────────────────────────────────────────────────────────────

def _symmetry_analysis(throttle_matrix: np.ndarray, motor_cols: List[str], profile: DroneProfile) -> Dict:
    """
    For a healthy multirotor in symmetric hover all motors should be roughly equal.
    Compute per-sample std-dev across motors → average over flight.
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    n_motors = throttle_matrix.shape[1]
    if n_motors < 2:
        return {"issues": issues, "metrics": metrics}

    # Per-sample cross-motor std-dev
    per_sample_std = np.std(throttle_matrix, axis=1)
    avg_imbalance = float(np.mean(per_sample_std))
    max_imbalance = float(np.percentile(per_sample_std, 95))

    # Symmetry score: perfect=0 stddev → 100; 20 stddev → 0
    symmetry_score = max(0.0, 100.0 - avg_imbalance * 5.0)

    metrics["avg_motor_imbalance_pct"] = round(avg_imbalance, 2)
    metrics["max_motor_imbalance_pct"] = round(max_imbalance, 2)
    metrics["symmetry_score"] = round(symmetry_score, 1)

    # Per-motor average (to identify which motor is the outlier)
    avg_per_motor = np.mean(throttle_matrix, axis=0)
    overall_avg = float(np.mean(avg_per_motor))
    for i, col in enumerate(motor_cols):
        deviation = abs(avg_per_motor[i] - overall_avg)
        if deviation > 25.0:
            issues.append(_issue(
                "critical", "MOT-003",
                f"Motor {col} has SEVERE output deviation ({avg_per_motor[i]:.1f}% vs avg {overall_avg:.1f}%). "
                "Possible damaged prop, worn bearing, or failing ESC.",
                value=round(float(avg_per_motor[i]), 2),
                threshold=round(overall_avg, 2),
            ))
        elif deviation > 15.0:
            issues.append(_issue(
                "warning", "MOT-003",
                f"Motor {col} has significant average output deviation "
                f"({avg_per_motor[i]:.1f}% vs fleet avg {overall_avg:.1f}%). "
                "Check prop balance, motor bearing, or ESC calibration.",
                value=round(float(avg_per_motor[i]), 2),
                threshold=round(overall_avg, 2),
            ))

    if avg_imbalance > 12.0:
        issues.append(_issue(
            "warning", "MOT-004",
            f"High average motor imbalance ({avg_imbalance:.1f}%). "
            "Drone is consistently compensating — inspect props, motors, and frame.",
            value=round(avg_imbalance, 2), threshold=12.0,
        ))
    elif avg_imbalance > 5.0:
        issues.append(_issue(
            "info", "MOT-004",
            f"Moderate motor output imbalance ({avg_imbalance:.1f}% mean deviation). "
            "Score reduced due to asymmetric workload — check propeller balance and ESC calibration.",
            value=round(avg_imbalance, 2), threshold=5.0,
        ))

    return {"issues": issues, "metrics": metrics}


def _throttle_headroom_analysis(throttle_matrix: np.ndarray, profile: DroneProfile) -> Dict:
    """
    Check whether motors have headroom (capacity to respond to demands).
    Consistently high throttle = overloaded drone, high wind, or heavy payload.
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    # Use mean of all motor outputs as overall throttle
    mean_throttle = np.mean(throttle_matrix, axis=1)
    avg_throttle = float(np.mean(mean_throttle))
    p95_throttle = float(np.percentile(mean_throttle, 95))

    warn_thresh = profile.motors.high_throttle_warn_pct
    crit_thresh = profile.motors.high_throttle_critical_pct
    hover_expected = profile.motors.hover_throttle_pct

    # Fraction of time above warn threshold
    frac_high = float(np.mean(mean_throttle > warn_thresh))

    metrics["avg_throttle_pct"] = round(avg_throttle, 2)
    metrics["p95_throttle_pct"] = round(p95_throttle, 2)
    metrics["frac_time_high_throttle"] = round(frac_high, 3)
    metrics["expected_hover_throttle_pct"] = hover_expected

    # Compare actual hover-ish throttle to expected
    # Use median as proxy for hover throttle
    median_throttle = float(np.median(mean_throttle))
    metrics["median_throttle_pct"] = round(median_throttle, 2)

    hover_delta = median_throttle - hover_expected
    metrics["hover_throttle_delta_pct"] = round(hover_delta, 2)

    if hover_delta > 15:
        issues.append(_issue(
            "warning", "MOT-005",
            f"Hover throttle ({median_throttle:.1f}%) significantly above expected "
            f"({hover_expected:.0f}%). Drone may be carrying more payload than rated, "
            "or motor/prop efficiency has degraded.",
            value=round(median_throttle, 2), threshold=hover_expected,
        ))

    if frac_high > 0.25:
        sev = "critical" if avg_throttle > crit_thresh else "warning"
        issues.append(_issue(
            sev, "MOT-006",
            f"Drone spent {frac_high * 100:.0f}% of flight above {warn_thresh:.0f}% throttle "
            f"(avg: {avg_throttle:.1f}%). Limited authority margin for gusts or manoeuvres. "
            "Consider reducing payload or improving aerodynamic efficiency.",
            value=round(avg_throttle, 2), threshold=warn_thresh,
        ))

    return {"issues": issues, "metrics": metrics}


def _oscillation_analysis(
    throttle_matrix: np.ndarray,
    flight_rcou: pd.DataFrame,
    motor_cols: List[str],
) -> Dict:
    """
    High-frequency oscillations in motor outputs = PID oscillation or resonance.
    Metric: mean absolute change per sample, normalised.
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    if throttle_matrix.shape[0] < 20:
        return {"issues": issues, "metrics": metrics}

    # Compute mean absolute first-difference across all motor channels
    diffs = np.abs(np.diff(throttle_matrix, axis=0))
    oscillation_index = float(np.mean(diffs))

    metrics["oscillation_index"] = round(oscillation_index, 3)

    # Heuristic thresholds: < 1.0 = smooth, > 3.0 = oscillating, > 6.0 = severe
    if oscillation_index > 6.0:
        issues.append(_issue(
            "critical", "MOT-007",
            f"Severe motor output oscillation (index: {oscillation_index:.2f}). "
            "PID gains are almost certainly too high — drone may have been unstable.",
            value=round(oscillation_index, 3), threshold=6.0,
        ))
    elif oscillation_index > 3.0:
        issues.append(_issue(
            "warning", "MOT-007",
            f"Elevated motor output oscillation (index: {oscillation_index:.2f}). "
            "Consider PID retuning or vibration dampening.",
            value=round(oscillation_index, 3), threshold=3.0,
        ))

    return {"issues": issues, "metrics": metrics}


def _saturation_analysis(throttle_matrix: np.ndarray, motor_cols: List[str]) -> Dict:
    """Detect samples where any motor hit 100% — loss of control authority."""
    issues: List[Dict] = []
    metrics: Dict = {}

    saturated_per_motor = []
    for i, col in enumerate(motor_cols):
        sat_count = int(np.sum(throttle_matrix[:, i] >= 99.0))
        sat_frac = sat_count / len(throttle_matrix)
        saturated_per_motor.append({
            "channel": col,
            "saturation_count": sat_count,
            "saturation_frac": round(sat_frac, 4),
        })

    total_sat_frac = float(np.mean([s["saturation_frac"] for s in saturated_per_motor]))
    metrics["motor_saturation"] = saturated_per_motor
    metrics["overall_saturation_frac"] = round(total_sat_frac, 4)

    if total_sat_frac > 0.05:
        issues.append(_issue(
            "critical", "MOT-008",
            f"Motor saturation: {total_sat_frac * 100:.1f}% of flight time with at least "
            "one motor at maximum output. Drone had NO control authority margin — "
            "dangerous in gusty conditions.",
            value=round(total_sat_frac * 100, 2), threshold=5.0,
        ))
    elif total_sat_frac > 0.01:
        issues.append(_issue(
            "warning", "MOT-008",
            f"Motor saturation events detected ({total_sat_frac * 100:.2f}% of samples). "
            "Reduce payload or avoid aggressive manoeuvres.",
            value=round(total_sat_frac * 100, 3), threshold=1.0,
        ))

    return {"issues": issues, "metrics": metrics}


def _dead_motor_analysis(throttle_matrix: np.ndarray, motor_cols: List[str]) -> Dict:
    """Detect motors stuck at or near idle during flight (possible motor/ESC failure)."""
    issues: List[Dict] = []
    metrics: Dict = {}

    dead_motors = []
    for i, col in enumerate(motor_cols):
        col_data = throttle_matrix[:, i]
        # A motor is "dead" if its max throttle never exceeds 15% during flight
        if float(np.max(col_data)) < 15.0:
            dead_motors.append(col)
            issues.append(_issue(
                "critical", "MOT-009",
                f"Motor {col} appears to be DEAD or disconnected — never exceeded 15% throttle "
                "during flight. Immediate inspection required.",
                value=round(float(np.max(col_data)), 2), threshold=15.0,
            ))

    metrics["dead_motors"] = dead_motors
    metrics["dead_motor_count"] = len(dead_motors)

    return {"issues": issues, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Channel identification
# ─────────────────────────────────────────────────────────────────────────────

def _get_motor_columns(rcou_df: pd.DataFrame, profile: DroneProfile) -> List[str]:
    """
    Return list of column names in RCOU that correspond to motor channels.

    Strategy (in order):
    1. Profile-specified channels — if they exist with a valid PWM range.
       The idle threshold heuristic is NOT applied here; the user-supplied
       profile is authoritative.
    2. If the log contains MORE motor channels than the profile lists
       (e.g. a hexacopter log analysed with a quad profile), expand to all
       auto-detected channels.  Fixed-wing is excluded from this expansion.
    3. Full auto-detection when no profile channel is found in the log.

    Motor vs servo heuristic (used for auto-detection only)
    --------------------------------------------------------
    Motors idle at ``min_pwm`` (e.g. 1000 µs) when disarmed, so their
    minimum PWM in the log is typically near ``min_pwm``.
    Servo channels are always positioned (centre ~1500 µs); their minimum
    only reaches ~1000 µs during full-throw commands — but their *mean* is
    near 1500.  Threshold: ``col_min <= min_pwm + 80``  (≤ 1080 for 1000 µs).
    """
    def _valid_pwm(col: str) -> bool:
        """Basic PWM-range sanity check (no idle-position filter)."""
        if col not in rcou_df.columns:
            return False
        col_min = float(rcou_df[col].min())
        col_max = float(rcou_df[col].max())
        return col_max - col_min > 100 and 800 <= col_min and col_max <= 2200

    def _is_motor_auto(col: str) -> bool:
        """PWM sanity + idle-position filter (for auto-detect only)."""
        if not _valid_pwm(col):
            return False
        col_min = float(rcou_df[col].min())
        idle_thresh = profile.motors.min_pwm + 80   # e.g. 1080 for 1000 µs
        return col_min <= idle_thresh

    # ── Step 1: profile-specified channels (trusted — no idle filter) ─────────
    profile_cols  = [f"C{ch}" for ch in profile.motors.channels]
    profile_found = [c for c in profile_cols if _valid_pwm(c)]

    # ── Step 2: full auto-detection with idle filter (C1–C16) ─────────────────
    auto_cols = [f"C{i}" for i in range(1, 17) if _is_motor_auto(f"C{i}")]

    if profile_found:
        # Multirotor / VTOL: expand to all auto-detected when more found than
        # the profile specifies — this handles hex/octo with a quad profile.
        if profile.type != "fixed_wing" and len(auto_cols) > len(profile_found):
            logger.info(
                "Profile specifies %d motor channel(s) %s but %d auto-detected %s — "
                "using all detected channels.",
                len(profile_found), profile_found,
                len(auto_cols), auto_cols,
            )
            return auto_cols
        return profile_found

    # ── Step 3: no profile channels found — full auto ─────────────────────────
    return auto_cols


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict]) -> float:
    score = 100.0

    # Dead motors — hard penalty
    dead_count = metrics.get("dead_motor_count", 0)
    if dead_count > 0:
        score -= dead_count * 30.0

    # Symmetry
    sym = metrics.get("symmetry_score", 100.0)
    score -= max(0.0, (100.0 - sym) * 0.4)

    # Throttle headroom
    avg_throttle = metrics.get("avg_throttle_pct", 0.0)
    if avg_throttle > 85:
        score -= 15.0
    elif avg_throttle > 75:
        score -= 8.0

    # Saturation
    sat_frac = metrics.get("overall_saturation_frac", 0.0)
    if sat_frac > 0.05:
        score -= 20.0
    elif sat_frac > 0.01:
        score -= 8.0

    # Oscillation
    osc = metrics.get("oscillation_index", 0.0)
    if osc > 6.0:
        score -= 15.0
    elif osc > 3.0:
        score -= 8.0

    # Dead motor is a catastrophic failure not covered by the symmetry/oscillation
    # metrics above (a dead motor drives symmetry down, but the 30-point dead_count
    # penalty already accounts for that — no additional loop deduction needed).
    # All other motor issues (MOT-003/004/007 etc.) are already captured through
    # the symmetry and oscillation metric deductions.  An issue-severity loop here
    # would double-penalise those codes, so it is intentionally absent.

    return max(0.0, min(100.0, score))


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _issue(
    severity: str,
    code: str,
    message: str,
    value: Any = None,
    threshold: Any = None,
    timestamp: Optional[float] = None,
) -> Dict:
    d: Dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if value is not None:
        d["value"] = value
    if threshold is not None:
        d["threshold"] = threshold
    if timestamp is not None:
        d["timestamp_s"] = timestamp
    return d
