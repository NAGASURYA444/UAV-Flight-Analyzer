"""
ESC Telemetry Health Module
============================
Phase 4.7 — Per-Motor ESC & Motor Health Analysis

Data source: ESC messages logged by ArduPilot when BLHeli32, AM32, or other
MAVLink-capable ESCs have telemetry enabled (SERVO_BLH_TRATE > 0 or
equivalent parameter set in firmware).

ESC message fields (ArduPilot 4.1+):
  Instance : ESC / motor index (0 = Motor 1, 1 = Motor 2, …)
  RPM      : Motor shaft speed (RPM)
  Volt     : ESC input voltage (V)
  Curr     : ESC current draw (A)
  Temp     : ESC board temperature (°C — 0 if hardware unsupported)
  RTemp    : Motor winding temperature (°C — 0 if hardware unsupported)
  Err      : Cumulative error count reported by ESC firmware

This module complements motors.py (which analyses PWM command symmetry from
RCOU) by checking what the ESCs and motors actually measured — not just what
the FC commanded.

Checks performed
----------------
1.  ESC / motor temperature
      — Per-ESC peak and mean temperature across the armed flight window.
      — Skipped silently if all Temp values are zero (hardware does not
        report temperature).
      — Motor winding temperature (RTemp) checked separately when available.

2.  Current imbalance across ESCs
      — Mean current draw per ESC compared to the fleet-wide mean.
      — High spread indicates a worn motor, damaged prop, or ESC issue
        on the outlier motor.

3.  RPM imbalance across ESCs
      — Mean RPM per ESC compared to the fleet-wide mean.
      — An ESC consistently spinning slower than peers (at the same
        commanded throttle) indicates bearing wear, a bent/nicked prop,
        or motor winding degradation.

4.  ESC error count
      — Any increase in the Err counter during the armed window means the
        ESC firmware detected a fault (desync, over-current, over-temp
        protection, or communication error).

Unavailable: ESC messages absent from log. This is expected when no
  BLHeli32 / AM32 ESC telemetry is connected, or SERVO_BLH_TRATE = 0.
  → Returns score=100, grade="A", available=False.

Scoring (from 100)
------------------
  −20 per ESC  ESC temperature critical (> 80 °C)       [cap −40]
  −10 per ESC  ESC temperature warning  (> 60 °C)       [cap −20]
  −20          Current imbalance critical (spread > 40% of mean)
  −10          Current imbalance warning  (spread > 20% of mean)
  −20          RPM imbalance critical    (spread > 30% of mean)
  −10          RPM imbalance warning     (spread > 15% of mean)
  −15          ESC error count > 0 on any motor
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

_TEMP_WARN      = 60.0   # °C — ESC board temperature warning
_TEMP_CRIT      = 80.0   # °C — ESC board temperature critical
_RTEMP_WARN     = 70.0   # °C — motor winding temperature warning (runs hotter)
_RTEMP_CRIT     = 100.0  # °C — motor winding temperature critical

_CURR_IMBAL_WARN = 20.0  # % spread of mean currents — warning
_CURR_IMBAL_CRIT = 40.0  # % spread of mean currents — critical

_RPM_IMBAL_WARN  = 15.0  # % spread of mean RPMs — warning
_RPM_IMBAL_CRIT  = 30.0  # % spread of mean RPMs — critical

_MIN_CURRENT_A   = 0.5   # A — below this, current data is noise; skip imbalance
_MIN_RPM         = 100   # RPM — below this, motor considered stopped; skip imbalance
_MIN_SAMPLES     = 10    # minimum per-ESC samples required for analysis

# Score deduction caps
_TEMP_WARN_CAP   = 20.0
_TEMP_CRIT_CAP   = 40.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    ESC Telemetry Health analysis.

    Returns standardised result dict.
    """
    esc_df = result.get("ESC")

    # ── Availability check ────────────────────────────────────────────────────
    if esc_df is None or len(esc_df) == 0:
        return _unavailable(
            "ESC telemetry messages not present in log. "
            "Enable ESC telemetry (set SERVO_BLH_TRATE > 0 or equivalent) "
            "and use BLHeli32 / AM32 ESCs with telemetry wiring."
        )

    # Scope to armed flight window
    end_time = disarm_time if disarm_time is not None else float("inf")
    armed_df = (
        esc_df[
            (esc_df["timestamp"] >= arm_time) &
            (esc_df["timestamp"] <= end_time)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(armed_df) < _MIN_SAMPLES:
        return _unavailable(
            f"Insufficient ESC telemetry samples in armed window "
            f"({len(armed_df)} found, need >= {_MIN_SAMPLES})."
        )

    # ── Split by ESC instance ─────────────────────────────────────────────────
    per_esc = _split_by_instance(armed_df)
    if not per_esc:
        return _unavailable("ESC Instance column missing — cannot identify per-motor data.")

    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    metrics["esc_count"]   = len(per_esc)
    metrics["sample_count"] = len(armed_df)

    # ── Step 1: Temperature ───────────────────────────────────────────────────
    temp_metrics, temp_issues = _analyse_temperature(per_esc)
    metrics.update(temp_metrics)
    issues.extend(temp_issues)

    # ── Step 2: Current imbalance ─────────────────────────────────────────────
    curr_metrics, curr_issues = _analyse_current_imbalance(per_esc)
    metrics.update(curr_metrics)
    issues.extend(curr_issues)

    # ── Step 3: RPM imbalance ─────────────────────────────────────────────────
    rpm_metrics, rpm_issues = _analyse_rpm_imbalance(per_esc)
    metrics.update(rpm_metrics)
    issues.extend(rpm_issues)

    # ── Step 4: ESC errors ────────────────────────────────────────────────────
    err_metrics, err_issues = _analyse_esc_errors(per_esc)
    metrics.update(err_metrics)
    issues.extend(err_issues)

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues)
    grade = _grade(score)

    # Summary line
    parts = [f"ESCs: {len(per_esc)}"]

    max_temp = metrics.get("max_esc_temp_c")
    if max_temp is not None:
        t_flag = "⚠" if max_temp > _TEMP_WARN else ""
        parts.append(f"Max temp: {max_temp:.0f}°C{t_flag}")

    curr_spread = metrics.get("current_imbalance_pct")
    if curr_spread is not None:
        parts.append(f"Curr imbalance: {curr_spread:.1f}%")

    rpm_spread = metrics.get("rpm_imbalance_pct")
    if rpm_spread is not None:
        parts.append(f"RPM imbalance: {rpm_spread:.1f}%")

    err_total = metrics.get("esc_errors_total", 0)
    parts.append(f"ESC errors: {err_total}")

    summary = "  |  ".join(parts)

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Instance splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_by_instance(
    df: pd.DataFrame,
) -> Dict[int, pd.DataFrame]:
    """
    Split armed DataFrame into per-ESC sub-DataFrames keyed by Instance number.
    If Instance column is absent, treat all rows as Instance 0.
    Only includes instances with at least _MIN_SAMPLES rows.
    """
    if "Instance" not in df.columns:
        if len(df) >= _MIN_SAMPLES:
            return {0: df}
        return {}

    per_esc: Dict[int, pd.DataFrame] = {}
    for inst in sorted(df["Instance"].dropna().unique()):
        sub = df[df["Instance"] == inst].reset_index(drop=True)
        if len(sub) >= _MIN_SAMPLES:
            per_esc[int(inst)] = sub

    return per_esc


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_temperature(
    per_esc: Dict[int, pd.DataFrame],
) -> Tuple[Dict, List[Dict]]:
    """Per-ESC temperature analysis (Temp = ESC board, RTemp = motor winding)."""
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    esc_temps: Dict[int, float] = {}    # mean temp per ESC
    esc_max_temps: Dict[int, float] = {}
    rtemp_max: Dict[int, float] = {}

    temp_warn_count = 0
    temp_crit_count = 0

    for inst, df in per_esc.items():
        label = f"ESC{inst + 1}"

        # ── ESC board temperature ──────────────────────────────────────────
        if "Temp" in df.columns:
            temp_vals = df["Temp"].dropna().values.astype(float)
            # Skip if all zeros — hardware not reporting temperature
            if len(temp_vals) >= 5 and float(np.nanmax(temp_vals)) > 0.0:
                mean_t = float(np.nanmean(temp_vals))
                max_t  = float(np.nanmax(temp_vals))
                esc_temps[inst]     = round(mean_t, 1)
                esc_max_temps[inst] = round(max_t, 1)

                if max_t > _TEMP_CRIT:
                    temp_crit_count += 1
                    issues.append(_issue(
                        "critical", "ESC-001",
                        f"{label} temperature critical: peak {max_t:.0f}°C "
                        f"(mean {mean_t:.0f}°C, threshold {_TEMP_CRIT:.0f}°C). "
                        "ESC is overheating — reduce continuous current draw, "
                        "improve airflow/cooling, or replace the ESC.",
                        value=round(max_t, 1),
                        threshold=_TEMP_CRIT,
                    ))
                elif max_t > _TEMP_WARN:
                    temp_warn_count += 1
                    issues.append(_issue(
                        "warning", "ESC-001",
                        f"{label} temperature elevated: peak {max_t:.0f}°C "
                        f"(mean {mean_t:.0f}°C, threshold {_TEMP_WARN:.0f}°C). "
                        "Monitor for further increase. Check ESC cooling and "
                        "confirm current draw is within ESC rated limits.",
                        value=round(max_t, 1),
                        threshold=_TEMP_WARN,
                    ))

        # ── Motor winding temperature ──────────────────────────────────────
        if "RTemp" in df.columns:
            rtemp_vals = df["RTemp"].dropna().values.astype(float)
            if len(rtemp_vals) >= 5 and float(np.nanmax(rtemp_vals)) > 0.0:
                max_rt = float(np.nanmax(rtemp_vals))
                rtemp_max[inst] = round(max_rt, 1)

                if max_rt > _RTEMP_CRIT:
                    issues.append(_issue(
                        "critical", "ESC-001",
                        f"{label} motor winding temperature critical: "
                        f"peak {max_rt:.0f}°C (threshold {_RTEMP_CRIT:.0f}°C). "
                        "Motor is severely overheating — inspect winding insulation "
                        "and reduce current load immediately.",
                        value=round(max_rt, 1),
                        threshold=_RTEMP_CRIT,
                    ))
                elif max_rt > _RTEMP_WARN:
                    issues.append(_issue(
                        "warning", "ESC-001",
                        f"{label} motor winding temperature elevated: "
                        f"peak {max_rt:.0f}°C (threshold {_RTEMP_WARN:.0f}°C). "
                        "Check motor KV rating vs actual current draw and "
                        "ensure adequate cooling airflow over the motors.",
                        value=round(max_rt, 1),
                        threshold=_RTEMP_WARN,
                    ))

    metrics["esc_temps_mean_c"]    = esc_temps
    metrics["esc_temps_max_c"]     = esc_max_temps
    metrics["motor_temps_max_c"]   = rtemp_max
    metrics["temp_warn_count"]     = temp_warn_count
    metrics["temp_crit_count"]     = temp_crit_count

    if esc_max_temps:
        metrics["max_esc_temp_c"] = round(max(esc_max_temps.values()), 1)
    else:
        metrics["max_esc_temp_c"] = None

    # Scalar motor winding max temp (for CLI display — single value, not a per-ESC dict)
    if rtemp_max:
        metrics["motor_winding_max_c"] = round(max(rtemp_max.values()), 1)
    else:
        metrics["motor_winding_max_c"] = None

    return metrics, issues


def _analyse_current_imbalance(
    per_esc: Dict[int, pd.DataFrame],
) -> Tuple[Dict, List[Dict]]:
    """
    Compare mean current draw across all ESCs.
    A large spread indicates one motor/prop is drawing disproportionately
    more or less current than its peers.
    """
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if not any("Curr" in df.columns for df in per_esc.values()):
        metrics["current_imbalance_available"] = False
        return metrics, issues

    mean_currents: Dict[int, float] = {}
    for inst, df in per_esc.items():
        if "Curr" not in df.columns:
            continue
        curr_vals = df["Curr"].dropna().values.astype(float)
        if len(curr_vals) >= 5:
            mean_currents[inst] = round(float(np.nanmean(curr_vals)), 2)

    if len(mean_currents) < 2:
        metrics["current_imbalance_available"] = False
        return metrics, issues

    vals = list(mean_currents.values())
    overall_mean = float(np.mean(vals))

    # Skip if motors are not spinning meaningfully
    if overall_mean < _MIN_CURRENT_A:
        metrics["current_imbalance_available"] = False
        return metrics, issues

    spread_pct = (max(vals) - min(vals)) / (overall_mean + 1e-9) * 100.0

    metrics["current_imbalance_available"] = True
    metrics["esc_mean_currents_a"]         = mean_currents
    metrics["current_fleet_mean_a"]        = round(overall_mean, 2)
    metrics["current_imbalance_pct"]       = round(spread_pct, 1)

    # Identify worst ESC
    worst_inst = max(mean_currents, key=lambda k: abs(mean_currents[k] - overall_mean))
    worst_curr = mean_currents[worst_inst]
    worst_dev  = abs(worst_curr - overall_mean) / (overall_mean + 1e-9) * 100.0
    metrics["current_worst_esc"]           = worst_inst + 1   # 1-indexed for display
    metrics["current_worst_deviation_pct"] = round(worst_dev, 1)

    if spread_pct > _CURR_IMBAL_CRIT:
        issues.append(_issue(
            "critical", "ESC-002",
            f"Current imbalance critical: {spread_pct:.0f}% spread across ESCs "
            f"(ESC{worst_inst + 1} draws {worst_curr:.1f} A vs fleet mean "
            f"{overall_mean:.1f} A, deviation {worst_dev:.0f}%). "
            "Likely causes: damaged or wrong-pitch propeller, motor bearing "
            "failure, or ESC calibration mismatch on the outlier motor.",
            value=round(spread_pct, 1),
            threshold=_CURR_IMBAL_CRIT,
        ))
    elif spread_pct > _CURR_IMBAL_WARN:
        issues.append(_issue(
            "warning", "ESC-002",
            f"Current imbalance elevated: {spread_pct:.0f}% spread across ESCs "
            f"(ESC{worst_inst + 1} is the outlier at {worst_curr:.1f} A). "
            "Check propeller condition and pitch consistency across all motors. "
            "Inspect motor bearings on the high-draw ESC.",
            value=round(spread_pct, 1),
            threshold=_CURR_IMBAL_WARN,
        ))

    return metrics, issues


def _analyse_rpm_imbalance(
    per_esc: Dict[int, pd.DataFrame],
) -> Tuple[Dict, List[Dict]]:
    """
    Compare mean RPM across all ESCs during the armed window.
    An ESC spinning consistently slower than peers at the same throttle
    indicates prop slip, bearing wear, or motor winding degradation.
    """
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if not any("RPM" in df.columns for df in per_esc.values()):
        metrics["rpm_imbalance_available"] = False
        return metrics, issues

    mean_rpms: Dict[int, float] = {}
    for inst, df in per_esc.items():
        if "RPM" not in df.columns:
            continue
        rpm_vals = df["RPM"].dropna().values.astype(float)
        # Only include samples where the motor is actually spinning
        spinning = rpm_vals[rpm_vals >= _MIN_RPM]
        if len(spinning) >= 5:
            mean_rpms[inst] = round(float(np.nanmean(spinning)), 0)

    if len(mean_rpms) < 2:
        metrics["rpm_imbalance_available"] = False
        return metrics, issues

    vals = list(mean_rpms.values())
    overall_mean = float(np.mean(vals))

    if overall_mean < _MIN_RPM:
        metrics["rpm_imbalance_available"] = False
        return metrics, issues

    spread_pct = (max(vals) - min(vals)) / (overall_mean + 1e-9) * 100.0

    metrics["rpm_imbalance_available"] = True
    metrics["esc_mean_rpms"]           = mean_rpms
    metrics["rpm_fleet_mean"]          = round(overall_mean, 0)
    metrics["rpm_imbalance_pct"]       = round(spread_pct, 1)

    worst_inst = min(mean_rpms, key=lambda k: mean_rpms[k])  # lowest RPM is suspect
    worst_rpm  = mean_rpms[worst_inst]
    worst_dev  = (overall_mean - worst_rpm) / (overall_mean + 1e-9) * 100.0
    metrics["rpm_worst_esc"]           = worst_inst + 1
    metrics["rpm_worst_deviation_pct"] = round(worst_dev, 1)

    if spread_pct > _RPM_IMBAL_CRIT:
        issues.append(_issue(
            "critical", "ESC-003",
            f"RPM imbalance critical: {spread_pct:.0f}% spread across ESCs "
            f"(ESC{worst_inst + 1} averages {worst_rpm:.0f} RPM vs fleet mean "
            f"{overall_mean:.0f} RPM, {worst_dev:.0f}% below). "
            "Inspect prop seating and pitch on ESC{} motor. Check motor "
            "bearings for roughness or binding.".format(worst_inst + 1),
            value=round(spread_pct, 1),
            threshold=_RPM_IMBAL_CRIT,
        ))
    elif spread_pct > _RPM_IMBAL_WARN:
        issues.append(_issue(
            "warning", "ESC-003",
            f"RPM imbalance elevated: {spread_pct:.0f}% spread across ESCs "
            f"(ESC{worst_inst + 1} is the low outlier at {worst_rpm:.0f} RPM). "
            "Check propeller seating, pitch consistency, and motor bearing "
            "condition on the low-RPM motor.",
            value=round(spread_pct, 1),
            threshold=_RPM_IMBAL_WARN,
        ))

    return metrics, issues


def _analyse_esc_errors(
    per_esc: Dict[int, pd.DataFrame],
) -> Tuple[Dict, List[Dict]]:
    """
    Check ESC Err counter. Any increase during the armed window means the
    ESC firmware detected a fault (desync, over-current protection, over-temp
    protection, or communication error).
    """
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if not any("Err" in df.columns for df in per_esc.values()):
        metrics["esc_error_available"] = False
        return metrics, issues

    esc_errors: Dict[int, int] = {}
    total_errors = 0

    for inst, df in per_esc.items():
        if "Err" not in df.columns:
            continue
        err_vals = df["Err"].dropna().values.astype(float)
        if len(err_vals) >= 2:
            # Cumulative counter: increase = errors in this flight
            delta = max(0, int(err_vals[-1]) - int(err_vals[0]))
            esc_errors[inst] = delta
            total_errors += delta

    metrics["esc_error_available"] = True
    metrics["esc_errors_per_motor"] = {k + 1: v for k, v in esc_errors.items()}
    metrics["esc_errors_total"]     = total_errors

    if total_errors > 0:
        error_escs = [f"ESC{k + 1}({v})" for k, v in esc_errors.items() if v > 0]
        issues.append(_issue(
            "critical", "ESC-004",
            f"ESC error count detected: {total_errors} total errors across "
            f"{', '.join(error_escs)}. ESC firmware reported fault events — "
            "possible causes: motor desync (check motor timing/KV vs prop load), "
            "over-current protection trigger, or telemetry communication errors. "
            "Inspect affected ESC and motor before next flight.",
            value=total_errors,
            threshold=0,
        ))

    return metrics, issues


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict]) -> float:
    score = 100.0

    # Temperature deductions (capped per severity)
    temp_warn_ded = min(_TEMP_WARN_CAP, metrics.get("temp_warn_count", 0) * 10.0)
    temp_crit_ded = min(_TEMP_CRIT_CAP, metrics.get("temp_crit_count", 0) * 20.0)
    # Critical replaces warning for same ESC — apply the larger of the two
    score -= max(temp_warn_ded, temp_crit_ded)

    # Current imbalance
    curr_spread = metrics.get("current_imbalance_pct")
    if curr_spread is not None:
        if curr_spread > _CURR_IMBAL_CRIT:
            score -= 20.0
        elif curr_spread > _CURR_IMBAL_WARN:
            score -= 10.0

    # RPM imbalance
    rpm_spread = metrics.get("rpm_imbalance_pct")
    if rpm_spread is not None:
        if rpm_spread > _RPM_IMBAL_CRIT:
            score -= 20.0
        elif rpm_spread > _RPM_IMBAL_WARN:
            score -= 10.0

    # ESC errors
    if metrics.get("esc_errors_total", 0) > 0:
        score -= 15.0

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "score":     100.0,
        "grade":     "A",
        "available": False,
        "issues":    [],
        "metrics":   {"unavailable_reason": reason},
        "summary":   reason,
    }


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
