"""
FC Power & Error Health Module
================================
Phase 4.3 — Flight Controller Power Rail and Firmware Error Analysis

Checks performed
----------------
1.  VCC rail stability    — main 5 V power rail to FC (POWR.Vcc)
2.  Servo rail stability  — servo/actuator rail (POWR.VServo or POWR.Vcc5V)
3.  Firmware error events — non-failsafe ERR messages (Main, CPU, Compass, etc.)

Failsafe ERR events (Battery, RC, GPS, EKF, Geofence, Thrust Loss) are handled
by the rc_link module and are explicitly excluded here to avoid double-counting.

Scoring (from 100)
------------------
-25   VCC critical undervoltage (< 4.3 V) — FC reset risk
-10   VCC rail sag warning (< 4.6 V)
- 8   VCC rail noisy (std dev > 0.15 V)
-20   Servo rail critically low (< 4.3 V)
-10   Servo rail low warning (< 4.7 V)
-20 per critical firmware error event  (up to -40 max)
- 8 per warning firmware error event   (up to -24 max)
- 5   multiple error events aggregate (> 2 triggered events)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# ── Power rail thresholds ─────────────────────────────────────────────────────
VCC_WARN_V       = 4.6    # V — VCC below this → warning
VCC_CRITICAL_V   = 4.3    # V — VCC below this → critical (FC reset risk)
VCC_NOISE_STD    = 0.15   # V — std dev above this → noisy power supply
SERVO_WARN_V     = 4.7    # V — servo rail below this → warning
SERVO_CRITICAL_V = 4.3    # V — servo rail below this → critical

# Failsafe subsystems already handled by rc_link — must exclude from fc_health
# 11 (Crash Check) is now reported as FSF-007 by rc_link — excluded here to avoid double-reporting
_FAILSAFE_SUBSYS = {5, 6, 7, 8, 11, 16, 20}

# FC-level non-failsafe error subsystems tracked here
# Format: subsys_id → (human_name, severity)
_FC_ERROR_SUBSYS: Dict[int, Tuple[str, str]] = {
    1:  ("Main",          "critical"),
    3:  ("Compass",       "warning"),
    10: ("GPS",           "warning"),
    15: ("EKF/DCM Check", "warning"),
    17: ("Barometer",     "warning"),
    18: ("CPU",           "critical"),
    19: ("Logging",       "warning"),
    21: ("Sensor Health", "warning"),
    23: ("Navigation",    "warning"),
}


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    FC Power and Error Health analysis.

    Parameters
    ----------
    result      : ParseResult from bin_parser
    profile     : DroneProfile
    arm_time    : arming timestamp (seconds)
    disarm_time : disarming timestamp (seconds) or None

    Returns
    -------
    Standard module dict: score, grade, available, issues, metrics, summary
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}
    score = 100.0

    powr_df  = result.get("POWR")
    err_df   = result.get("ERR")
    end_time = disarm_time if disarm_time is not None else float("inf")

    # ── Power rail analysis ────────────────────────────────────────────────
    powr_issues, powr_metrics, powr_deduction = _analyse_power_rails(
        powr_df, arm_time, end_time
    )
    issues.extend(powr_issues)
    metrics.update(powr_metrics)
    score -= powr_deduction

    # ── Firmware error event analysis ──────────────────────────────────────
    err_issues, err_metrics, err_deduction = _analyse_errors(
        err_df, arm_time, end_time
    )
    issues.extend(err_issues)
    metrics.update(err_metrics)
    score -= err_deduction

    score = max(0.0, min(100.0, score))

    from analyzer.scoring.engine import _grade
    grade = _grade(score)

    # ── Summary ────────────────────────────────────────────────────────────
    err_count  = metrics.get("fc_error_count", 0)
    powr_avail = metrics.get("powr_available", False)
    vcc_min    = metrics.get("vcc_min_v")

    if powr_avail and vcc_min is not None:
        vcc_status = "stable" if vcc_min >= VCC_WARN_V else "sag detected"
        summary = (
            f"VCC min {vcc_min:.2f} V ({vcc_status})  |  "
            f"{err_count} firmware error event{'s' if err_count != 1 else ''}"
        )
    else:
        summary = (
            f"Power rail data unavailable  |  "
            f"{err_count} firmware error event{'s' if err_count != 1 else ''}"
        )

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─── Power rail analysis ───────────────────────────────────────────────────────

def _analyse_power_rails(
    powr_df: Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
) -> Tuple[List[Dict], Dict[str, Any], float]:
    """Analyse VCC and servo rail voltages from POWR messages."""
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}
    deduction = 0.0

    if powr_df is None or powr_df.empty:
        metrics.update({
            "powr_available":          False,
            "vcc_min_v":               None,
            "vcc_mean_v":              None,
            "vcc_std_v":               None,
            "vcc_undervolt_count":     0,
            "servo_rail_available":    False,
            "servo_min_v":             None,
            "servo_mean_v":            None,
        })
        return issues, metrics, deduction

    # Scope to armed flight window
    flight_powr = powr_df[
        (powr_df["timestamp"] >= arm_time) &
        (powr_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_powr) < 5:
        metrics["powr_available"] = False
        metrics["vcc_min_v"]      = None
        return issues, metrics, deduction

    metrics["powr_available"] = True

    # ── VCC rail ──────────────────────────────────────────────────────────
    vcc_col = _find_col(flight_powr, ["Vcc", "VCC", "vcc"])
    if vcc_col is not None:
        vcc = flight_powr[vcc_col].values.astype(float)

        # Some boards log Vcc in millivolts — correct if mean >> 10 V
        if float(np.nanmean(vcc)) > 10.0:
            vcc = vcc / 1000.0

        vcc_min  = float(np.min(vcc))
        vcc_mean = float(np.mean(vcc))
        vcc_std  = float(np.std(vcc))

        metrics["vcc_min_v"]           = round(vcc_min, 3)
        metrics["vcc_mean_v"]          = round(vcc_mean, 3)
        metrics["vcc_std_v"]           = round(vcc_std, 4)
        metrics["vcc_undervolt_count"] = int(np.sum(vcc < VCC_WARN_V))

        if vcc_min < VCC_CRITICAL_V:
            deduction += 25.0
            issues.append(_issue(
                "critical", "FCH-001",
                f"VCC power rail critically low: min {vcc_min:.2f} V "
                f"(threshold {VCC_CRITICAL_V} V). FC reset or sensor malfunction risk. "
                "Check power module and wiring immediately.",
                value=round(vcc_min, 3),
                threshold=VCC_CRITICAL_V,
            ))
        elif vcc_min < VCC_WARN_V:
            deduction += 10.0
            issues.append(_issue(
                "warning", "FCH-002",
                f"VCC power rail sag: min {vcc_min:.2f} V "
                f"(threshold {VCC_WARN_V} V). "
                "Check power module output and filter capacitor health.",
                value=round(vcc_min, 3),
                threshold=VCC_WARN_V,
            ))

        if vcc_std > VCC_NOISE_STD:
            deduction += 8.0
            issues.append(_issue(
                "warning", "FCH-003",
                f"VCC power rail noisy: std dev {vcc_std:.3f} V "
                f"(threshold {VCC_NOISE_STD} V). "
                "Possible power filter fault or ESC switching noise coupling to FC rail.",
                value=round(vcc_std, 3),
                threshold=VCC_NOISE_STD,
            ))
    else:
        metrics["vcc_min_v"]           = None
        metrics["vcc_mean_v"]          = None
        metrics["vcc_std_v"]           = None
        metrics["vcc_undervolt_count"] = 0

    # ── Servo rail ────────────────────────────────────────────────────────
    servo_col = _find_col(flight_powr, ["VServo", "Vcc5V", "VccBoard", "Vservo"])
    if servo_col is not None:
        servo = flight_powr[servo_col].values.astype(float)
        if float(np.nanmean(servo)) > 10.0:
            servo = servo / 1000.0

        servo_mean = float(np.mean(servo))

        # Skip servo rail analysis if rail is unpowered (≤ 0.5 V mean) —
        # multirotors with direct-drive ESCs typically have no servo rail BEC
        if servo_mean <= 0.5:
            metrics["servo_rail_available"] = False
            metrics["servo_min_v"]          = None
            metrics["servo_mean_v"]         = None
            return issues, metrics, deduction

        servo_min  = float(np.min(servo))

        metrics["servo_rail_available"] = True
        metrics["servo_min_v"]          = round(servo_min, 3)
        metrics["servo_mean_v"]         = round(servo_mean, 3)

        if servo_min < SERVO_CRITICAL_V:
            deduction += 20.0
            issues.append(_issue(
                "critical", "FCH-005",
                f"Servo power rail critically low: min {servo_min:.2f} V "
                f"(threshold {SERVO_CRITICAL_V} V). "
                "Actuator reliability at risk — check BEC/servo rail wiring.",
                value=round(servo_min, 3),
                threshold=SERVO_CRITICAL_V,
            ))
        elif servo_min < SERVO_WARN_V:
            deduction += 10.0
            issues.append(_issue(
                "warning", "FCH-004",
                f"Servo power rail low: min {servo_min:.2f} V "
                f"(threshold {SERVO_WARN_V} V). "
                "Check BEC output voltage and servo power connections.",
                value=round(servo_min, 3),
                threshold=SERVO_WARN_V,
            ))
    else:
        metrics["servo_rail_available"] = False
        metrics["servo_min_v"]          = None
        metrics["servo_mean_v"]         = None

    return issues, metrics, deduction


# ─── Firmware error event analysis ────────────────────────────────────────────

def _analyse_errors(
    err_df: Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
) -> Tuple[List[Dict], Dict[str, Any], float]:
    """Analyse non-failsafe firmware error events from ERR messages."""
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}
    deduction = 0.0
    fc_error_events: List[Dict] = []

    if err_df is None or err_df.empty:
        metrics["fc_error_count"]  = 0
        metrics["fc_error_events"] = []
        return issues, metrics, deduction

    # Scope to armed flight window
    flight_err = err_df[
        (err_df["timestamp"] >= arm_time) &
        (err_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_err) == 0:
        metrics["fc_error_count"]  = 0
        metrics["fc_error_events"] = []
        return issues, metrics, deduction

    critical_count = 0
    warning_count  = 0

    for _, row in flight_err.iterrows():
        subsys = int(row.get("Subsys", -1))
        ecode  = int(row.get("ECode",  0))
        ts     = float(row.get("timestamp", 0))

        # Skip failsafe subsystems — handled by rc_link
        if subsys in _FAILSAFE_SUBSYS:
            continue
        # Skip unknown subsystems
        if subsys not in _FC_ERROR_SUBSYS:
            continue

        subsys_name, subsys_sev = _FC_ERROR_SUBSYS[subsys]

        if ecode == 0:
            # Error cleared — info only, no deduction
            fc_error_events.append({
                "timestamp_s": round(ts, 2),
                "subsystem":   subsys_name,
                "ecode":       ecode,
                "status":      "cleared",
            })
            issues.append(_issue(
                "info", "FCH-007",
                f"FC error cleared: {subsys_name} subsystem at T+{ts:.1f} s.",
                timestamp=ts,
            ))
        else:
            # Error triggered
            fc_error_events.append({
                "timestamp_s": round(ts, 2),
                "subsystem":   subsys_name,
                "ecode":       ecode,
                "status":      "triggered",
            })
            issues.append(_issue(
                subsys_sev, "FCH-006",
                f"FC firmware error: {subsys_name} (ECode {ecode}) at T+{ts:.1f} s. "
                "Review subsystem calibration and hardware connections.",
                value=ecode,
                timestamp=ts,
            ))
            if subsys_sev == "critical":
                critical_count += 1
            else:
                warning_count += 1

    # Per-event deductions (capped)
    deduction += min(40.0, critical_count * 20.0)
    deduction += min(24.0, warning_count  *  8.0)

    # Multiple events aggregate penalty
    total_triggered = critical_count + warning_count
    if total_triggered > 2:
        deduction += 5.0
        issues.append(_issue(
            "warning", "FCH-008",
            f"Multiple FC firmware error events detected ({total_triggered} total). "
            "Review flight controller health logs and sensor calibrations.",
            value=total_triggered,
        ))

    metrics["fc_error_count"]  = total_triggered
    metrics["fc_error_events"] = fc_error_events

    return issues, metrics, deduction


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _issue(
    severity: str,
    code: str,
    message: str,
    value: Any = None,
    threshold: Any = None,
    timestamp: Optional[float] = None,
) -> Dict:
    d: Dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if value     is not None: d["value"]       = value
    if threshold is not None: d["threshold"]   = threshold
    if timestamp is not None: d["timestamp_s"] = timestamp
    return d
