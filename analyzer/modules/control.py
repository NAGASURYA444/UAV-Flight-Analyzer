"""
Control Performance Analysis Module
=====================================
Analyses ATT (attitude) messages to assess how accurately the flight controller
tracked desired roll, pitch and yaw commands throughout the flight.

Checks performed
----------------
1.  Roll tracking error   — mean, RMS and max deviation (DesRoll vs Roll)
2.  Pitch tracking error  — mean, RMS and max deviation (DesPitch vs Pitch)
3.  Yaw tracking error    — mean, RMS and max deviation (DesYaw vs Yaw)
4.  Time with large error — % of samples where |error| > large-error threshold
5.  Attitude oscillation  — sliding-window variance to detect PID hunting

Thresholds (degrees)
--------------------
                    multirotor      fixed_wing / vtol
large_error_deg         5.0              8.0
critical_error_deg     10.0             15.0
oscillation_rms_deg     2.0              3.0

Scoring deductions (from 100)
------------------------------
-10   Roll RMS > warning threshold
-15   Roll RMS > critical threshold
-10   Pitch RMS > warning threshold
-15   Pitch RMS > critical threshold
- 5   Yaw RMS > warning threshold
-10   Yaw RMS > critical threshold
-10   > 5 % of time with large roll or pitch errors
-15   PID oscillation detected (sliding-window variance elevated)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# ─── Thresholds ──────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "multirotor": {
        "rms_warn_deg":      1.5,
        "rms_critical_deg":  3.5,
        "yaw_rms_warn_deg":  3.0,
        "yaw_rms_crit_deg":  6.0,
        "large_error_deg":   5.0,
        "large_error_pct":   5.0,   # % of samples
        "osc_window_s":      2.0,
        "osc_rms_warn":      2.0,
        "osc_rms_crit":      4.0,
    },
    "fixed_wing": {
        "rms_warn_deg":      4.0,
        "rms_critical_deg":  8.0,
        "yaw_rms_warn_deg":  None,  # yaw not scored for fixed_wing
        "yaw_rms_crit_deg":  None,
        "large_error_deg":   8.0,
        "large_error_pct":   5.0,
        "osc_window_s":      3.0,
        "osc_rms_warn":      4.0,
        "osc_rms_crit":      7.0,
    },
    "vtol": {
        "rms_warn_deg":      4.0,   # fixed-wing cruise has higher baseline error than multirotor
        "rms_critical_deg":  8.0,
        "yaw_rms_warn_deg":  None,  # yaw not scored for vtol — controlled via bank angle
        "yaw_rms_crit_deg":  None,
        "large_error_deg":   8.0,
        "large_error_pct":   5.0,
        "osc_window_s":      2.0,
        "osc_rms_warn":      4.0,
        "osc_rms_crit":      7.0,
    },
}


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run control performance analysis.

    Parameters
    ----------
    result      : ParseResult from bin_parser
    profile     : DroneProfile
    arm_time    : float — timestamp of arming (seconds)
    disarm_time : float or None — timestamp of disarming (seconds)

    Returns
    -------
    Standard module dict: score, grade, available, issues, metrics, summary
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    att_df = result.get("ATT")
    if att_df is None or att_df.empty:
        logger.warning("ATT messages not found — control analysis unavailable.")
        return _unavailable("ATT messages not found in log")

    # ── Filter to armed window ────────────────────────────────────────────────
    end_time = disarm_time if disarm_time else att_df["timestamp"].max()
    flight_att = att_df[
        (att_df["timestamp"] >= arm_time) & (att_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_att) < 20:
        return _unavailable(f"Too few ATT samples during armed window ({len(flight_att)})")

    # ── Check required columns ────────────────────────────────────────────────
    required = {"DesRoll", "Roll", "DesPitch", "Pitch"}
    missing = required - set(flight_att.columns)
    if missing:
        return _unavailable(f"Missing ATT columns: {missing}")

    is_fixedwing = profile.type in ("fixed_wing", "vtol")
    # For fixed_wing/vtol, DesYaw vs Yaw is not a meaningful tracking metric —
    # heading is controlled via bank angle (crab angle), not a yaw channel.
    has_yaw = (
        "DesYaw" in flight_att.columns
        and "Yaw" in flight_att.columns
        and not is_fixedwing
    )

    # ── Compute tracking errors ───────────────────────────────────────────────
    roll_err  = (flight_att["DesRoll"]  - flight_att["Roll"]).values
    pitch_err = (flight_att["DesPitch"] - flight_att["Pitch"]).values

    # Yaw wraps at ±180° — always compute with wrap correction so that
    # e.g. DesYaw=1° vs Yaw=359° gives 2° error, not 358°.
    if has_yaw:
        yaw_raw = flight_att["DesYaw"].values - flight_att["Yaw"].values
        yaw_err = ((yaw_raw + 180) % 360) - 180   # wrap to [-180, 180]

        # For fixed-wing / VTOL: exclude active-turn samples from yaw RMS.
        # During a banked turn the heading naturally lags DesYaw — this is
        # normal aerodynamics, not a control fault.  We identify turns as
        # samples where |DesYaw rate of change| > 3 °/s.
        if is_fixedwing and len(flight_att) > 2:
            ts = flight_att["timestamp"].values
            dt = np.diff(ts, prepend=ts[0] + 1e-6)
            dt = np.where(dt <= 0, 0.1, dt)
            des_yaw_vals = flight_att["DesYaw"].values
            des_yaw_diff = ((np.diff(des_yaw_vals, prepend=des_yaw_vals[0]) + 180) % 360) - 180
            des_yaw_rate = np.abs(des_yaw_diff / dt)
            straight_mask = des_yaw_rate < 3.0   # °/s — below = straight leg
            yaw_err_straight = yaw_err[straight_mask]
            yaw_scored = yaw_err_straight if len(yaw_err_straight) > 10 else yaw_err
            yaw_turn_excluded = int(np.sum(~straight_mask))
        else:
            yaw_scored = yaw_err
            straight_mask = np.ones(len(yaw_err), dtype=bool)
            yaw_turn_excluded = 0
    else:
        yaw_err = None
        yaw_scored = None
        yaw_turn_excluded = 0

    # ── Select thresholds for drone type ─────────────────────────────────────
    dtype = profile.type if profile.type in _THRESHOLDS else "multirotor"
    thr = dict(_THRESHOLDS[dtype])  # copy so we can override
    # Apply profile threshold overrides (profile YAML thresholds: section)
    _THR_KEYS = [
        "rms_warn_deg", "rms_critical_deg", "yaw_rms_warn_deg", "yaw_rms_crit_deg",
        "large_error_deg", "large_error_pct", "osc_window_s", "osc_rms_warn", "osc_rms_crit",
    ]
    for key in _THR_KEYS:
        profile_val = profile.thresholds.get(f"ctl_{key}")
        if profile_val is not None:
            thr[key] = float(profile_val)

    # ── Compute per-axis statistics ───────────────────────────────────────────
    roll_rms  = float(np.sqrt(np.mean(roll_err ** 2)))
    roll_mean = float(np.mean(np.abs(roll_err)))
    roll_max  = float(np.max(np.abs(roll_err)))

    pitch_rms  = float(np.sqrt(np.mean(pitch_err ** 2)))
    pitch_mean = float(np.mean(np.abs(pitch_err)))
    pitch_max  = float(np.max(np.abs(pitch_err)))

    if yaw_scored is not None and len(yaw_scored) > 0:
        yaw_rms  = float(np.sqrt(np.mean(yaw_scored ** 2)))
        yaw_mean = float(np.mean(np.abs(yaw_scored)))
        yaw_max  = float(np.max(np.abs(yaw_err)))   # overall max (includes turns)
    else:
        yaw_rms = yaw_mean = yaw_max = None

    # % samples with large error (roll or pitch)
    large_err_mask = (np.abs(roll_err) > thr["large_error_deg"]) | \
                     (np.abs(pitch_err) > thr["large_error_deg"])
    large_err_pct = float(np.sum(large_err_mask) / len(roll_err) * 100)

    # ── Oscillation detection via sliding window ──────────────────────────────
    timestamps = flight_att["timestamp"].values
    dt_median = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.1
    window_samples = max(10, int(thr["osc_window_s"] / max(dt_median, 0.001)))

    roll_osc_rms, pitch_osc_rms = _oscillation_rms(roll_err, pitch_err, window_samples)

    # ── Store metrics ─────────────────────────────────────────────────────────
    metrics.update({
        "att_samples":          len(flight_att),
        "roll_rms_deg":         round(roll_rms,  2),
        "roll_mean_err_deg":    round(roll_mean, 2),
        "roll_max_err_deg":     round(roll_max,  1),
        "pitch_rms_deg":        round(pitch_rms,  2),
        "pitch_mean_err_deg":   round(pitch_mean, 2),
        "pitch_max_err_deg":    round(pitch_max,  1),
        "large_error_pct":      round(large_err_pct, 1),
        "roll_osc_rms_deg":     round(roll_osc_rms,  2),
        "pitch_osc_rms_deg":    round(pitch_osc_rms, 2),
        "has_yaw":              has_yaw,
        "yaw_turn_excluded":    yaw_turn_excluded,
    })
    if yaw_rms is not None:
        metrics.update({
            "yaw_rms_deg":         round(yaw_rms,  2),
            "yaw_mean_err_deg":    round(yaw_mean, 2),
            "yaw_max_err_deg":     round(yaw_max,  1),
            "yaw_straight_only":   is_fixedwing,
        })

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = 100.0

    # Roll
    if roll_rms >= thr["rms_critical_deg"]:
        score -= 15
        issues.append({
            "severity": "critical",
            "code": "CTL-001",
            "message": f"Roll tracking error critical: RMS {roll_rms:.1f}° (threshold {thr['rms_critical_deg']}°)",
            "value": round(roll_rms, 2),
            "threshold": thr["rms_critical_deg"],
        })
    elif roll_rms >= thr["rms_warn_deg"]:
        score -= 10
        issues.append({
            "severity": "warning",
            "code": "CTL-001",
            "message": f"Roll tracking error elevated: RMS {roll_rms:.1f}° (threshold {thr['rms_warn_deg']}°)",
            "value": round(roll_rms, 2),
            "threshold": thr["rms_warn_deg"],
        })

    # Pitch
    if pitch_rms >= thr["rms_critical_deg"]:
        score -= 15
        issues.append({
            "severity": "critical",
            "code": "CTL-002",
            "message": f"Pitch tracking error critical: RMS {pitch_rms:.1f}° (threshold {thr['rms_critical_deg']}°)",
            "value": round(pitch_rms, 2),
            "threshold": thr["rms_critical_deg"],
        })
    elif pitch_rms >= thr["rms_warn_deg"]:
        score -= 10
        issues.append({
            "severity": "warning",
            "code": "CTL-002",
            "message": f"Pitch tracking error elevated: RMS {pitch_rms:.1f}° (threshold {thr['rms_warn_deg']}°)",
            "value": round(pitch_rms, 2),
            "threshold": thr["rms_warn_deg"],
        })

    # Yaw — only scored for multirotor (fixed_wing/vtol use bank angle to control heading)
    yaw_crit = thr.get("yaw_rms_crit_deg")
    yaw_warn = thr.get("yaw_rms_warn_deg")
    if yaw_rms is not None and yaw_crit is not None and yaw_warn is not None:
        if yaw_rms >= yaw_crit:
            score -= 10
            issues.append({
                "severity": "critical",
                "code": "CTL-003",
                "message": f"Yaw tracking error critical: RMS {yaw_rms:.1f}° (threshold {yaw_crit}°)",
                "value": round(yaw_rms, 2),
                "threshold": yaw_crit,
            })
        elif yaw_rms >= yaw_warn:
            score -= 5
            issues.append({
                "severity": "warning",
                "code": "CTL-003",
                "message": f"Yaw tracking error elevated: RMS {yaw_rms:.1f}° (threshold {yaw_warn}°)",
                "value": round(yaw_rms, 2),
                "threshold": yaw_warn,
            })

    # Large error time
    if large_err_pct > thr["large_error_pct"]:
        score -= 10
        issues.append({
            "severity": "warning",
            "code": "CTL-004",
            "message": f"{large_err_pct:.1f}% of flight with attitude error > {thr['large_error_deg']}° "
                       f"(threshold {thr['large_error_pct']}%)",
            "value": round(large_err_pct, 1),
            "threshold": thr["large_error_pct"],
        })

    # Oscillation
    max_osc_rms = max(roll_osc_rms, pitch_osc_rms)
    if max_osc_rms >= thr["osc_rms_crit"]:
        score -= 15
        issues.append({
            "severity": "critical",
            "code": "CTL-005",
            "message": f"PID oscillation detected: peak window RMS {max_osc_rms:.1f}° "
                       f"(threshold {thr['osc_rms_crit']}°)",
            "value": round(max_osc_rms, 2),
            "threshold": thr["osc_rms_crit"],
        })
    elif max_osc_rms >= thr["osc_rms_warn"]:
        score -= 10
        issues.append({
            "severity": "warning",
            "code": "CTL-005",
            "message": f"Possible PID oscillation: peak window RMS {max_osc_rms:.1f}° "
                       f"(threshold {thr['osc_rms_warn']}°)",
            "value": round(max_osc_rms, 2),
            "threshold": thr["osc_rms_warn"],
        })

    score = max(0.0, min(100.0, score))

    from analyzer.scoring.engine import _grade  # local import to avoid circular
    grade = _grade(score)

    # ── Summary ───────────────────────────────────────────────────────────────
    yaw_note = " (straight legs only)" if (yaw_rms is not None and is_fixedwing and yaw_turn_excluded > 0) else ""
    summary = (
        f"Roll RMS {roll_rms:.1f}°, Pitch RMS {pitch_rms:.1f}°"
        + (f", Yaw RMS {yaw_rms:.1f}°{yaw_note}" if yaw_rms is not None else "")
        + f" | Large-error {large_err_pct:.1f}%"
        + (f" | Osc RMS {max_osc_rms:.1f}°" if max_osc_rms > 0.5 else "")
    )

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _oscillation_rms(
    roll_err: np.ndarray,
    pitch_err: np.ndarray,
    window: int,
) -> Tuple[float, float]:
    """
    Compute the 95th-percentile RMS of attitude error within sliding windows.
    A healthy drone should have low short-term variance; PID hunting inflates it.
    """
    if len(roll_err) < window * 2:
        return 0.0, 0.0

    roll_rms_windows  = []
    pitch_rms_windows = []

    step = max(1, window // 4)
    for start in range(0, len(roll_err) - window, step):
        seg_r = roll_err[start: start + window]
        seg_p = pitch_err[start: start + window]
        # Detrend each window (remove mean) before computing RMS to avoid
        # confusing steady-state offsets with oscillation
        roll_rms_windows.append(float(np.sqrt(np.mean((seg_r - seg_r.mean()) ** 2))))
        pitch_rms_windows.append(float(np.sqrt(np.mean((seg_p - seg_p.mean()) ** 2))))

    if not roll_rms_windows:
        return 0.0, 0.0

    # 95th percentile captures worst sustained oscillation, ignores rare spikes
    roll_peak  = float(np.percentile(roll_rms_windows,  95))
    pitch_peak = float(np.percentile(pitch_rms_windows, 95))
    return roll_peak, pitch_peak


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "score":     50.0,
        "grade":     "C",
        "available": False,
        "issues":    [],
        "metrics":   {"unavailable_reason": reason},
        "summary":   f"Control analysis unavailable: {reason}",
    }
