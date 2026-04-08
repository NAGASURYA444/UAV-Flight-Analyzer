"""
RC Link & Failsafe Events Module
==================================
Phase 4.1 — RC Signal Health and Failsafe Event Detection

RC Detection (auto):
  1. No RCIN data → autonomous flight, RC section skipped
  2. RCIN flat (all channels std < 20 µs) → RC connected but not used (autonomous)
  3. RCIN active (any channel std >= 20 µs) → piloted/semi-autonomous flight

RC Signal Analysis (only when RC actively used):
  - Signal dropouts: timestamp gaps > 10× median interval (min 0.5 s)
  - Channel saturation: any RC channel pegged at min/max > 30% of flight

Failsafe Events (always — applies to all drone types, RC or autonomous):
  - Battery failsafe  (ERR subsys 6)
  - RC/Radio failsafe (ERR subsys 5)
  - GPS failsafe      (ERR subsys 7)
  - EKF failsafe      (ERR subsys 16)
  - Geofence failsafe (ERR subsys 8)
  - Thrust loss check (ERR subsys 20)
  - Crash check       (ERR subsys 11)

Scoring deductions (from 100):
  -30   Crash check triggered (drone detected tumble/flip/crash mid-flight)
  -20   RC failsafe triggered in flight
  -20   EKF failsafe triggered
  -20   Thrust loss failsafe triggered
  -15   Battery failsafe triggered
  -10   GPS failsafe triggered
  -5    Geofence breach
  -5 per dropout (max -20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# Failsafe subsystem IDs in ArduPilot ERR messages
FAILSAFE_SUBSYSTEMS: Dict[int, str] = {
    5:  "RC/Radio",
    6:  "Battery",
    7:  "GPS",
    8:  "Geofence",
    11: "Crash Check",
    16: "EKF",
    20: "Thrust Loss",
}


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
) -> Dict[str, Any]:
    """
    RC Link & Failsafe Events analysis.

    Returns standardised result dict.
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    rcin_df = result.get("RCIN")
    err_df  = result.get("ERR")

    # ── Step 1: RC mode detection ─────────────────────────────────────────────
    rc_mode, rc_channels = _detect_rc_mode(rcin_df)
    metrics["rc_mode"] = rc_mode   # "autonomous" | "rc_connected_unused" | "rc_active"
    metrics["rc_channels_detected"] = rc_channels

    if rc_mode == "autonomous":
        issues.append(_issue(
            "info", "RCL-001",
            "No RC receiver detected in log data — autonomous flight.",
        ))
    elif rc_mode == "rc_connected_unused":
        issues.append(_issue(
            "info", "RCL-002",
            "RC receiver connected but no pilot input detected — "
            "autonomous / pre-programmed flight.",
        ))

    # ── Step 2: RC Signal Analysis (only when RC actively used) ───────────────
    metrics["rc_dropout_count"] = 0
    metrics["rc_dropout_total_s"] = 0.0
    metrics["rc_dropouts"] = []
    metrics["rc_saturated_channels"] = []

    if rc_mode == "rc_active" and rcin_df is not None:
        flight_rcin = (
            rcin_df[rcin_df["timestamp"] >= arm_time]
            .copy()
            .reset_index(drop=True)
        )
        if len(flight_rcin) >= 10:
            rc_metrics, rc_issues = _analyse_rc_signal(flight_rcin, rc_channels)
            metrics.update(rc_metrics)
            issues.extend(rc_issues)

    # ── Step 3: Failsafe Events (always — applies to all drone types) ─────────
    fsf_metrics, fsf_issues = _analyse_failsafes(err_df, arm_time)
    metrics.update(fsf_metrics)
    issues.extend(fsf_issues)

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues)
    grade = _grade(score)

    # Summary line
    dropout_count  = metrics.get("rc_dropout_count", 0)
    failsafe_count = metrics.get("failsafe_event_count", 0)
    if rc_mode == "autonomous":
        summary = f"Autonomous flight (no RC)  |  Failsafe events: {failsafe_count}"
    elif rc_mode == "rc_connected_unused":
        summary = f"RC connected, unused  |  Failsafe events: {failsafe_count}"
    else:
        summary = (
            f"RC active  |  "
            f"Dropouts: {dropout_count}  |  "
            f"Failsafe events: {failsafe_count}"
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
# RC Detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_rc_mode(
    rcin_df: Optional[pd.DataFrame],
) -> Tuple[str, List[str]]:
    """
    Auto-detect whether RC was used in this flight.

    Returns
    -------
    (mode, active_channels)
        mode             : "autonomous" | "rc_connected_unused" | "rc_active"
        active_channels  : column names that showed real stick movement
    """
    if rcin_df is None or len(rcin_df) < 5:
        return "autonomous", []

    ch_cols = [
        c for c in rcin_df.columns
        if c.startswith("C") and c[1:].isdigit()
    ]
    if not ch_cols:
        return "autonomous", []

    active_channels: List[str] = []
    for col in ch_cols:
        vals = rcin_df[col].dropna().values
        if len(vals) < 5:
            continue
        # Genuine pilot input requires BIDIRECTIONAL movement around RC center (1500 µs):
        #   p10 < 1400 µs  → stick moves toward low end
        #   p90 > 1600 µs  → stick moves toward high end
        # This correctly excludes:
        #   - Mode switches fixed at max (2000 µs) — p10 never < 1400
        #   - VTOL passthrough channels near one extreme
        #   - Throttle channel held at zero before arming
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        if p10 < 1400 and p90 > 1600:
            active_channels.append(col)

    if not active_channels:
        return "rc_connected_unused", ch_cols

    return "rc_active", active_channels


# ─────────────────────────────────────────────────────────────────────────────
# RC Signal Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_rc_signal(
    flight_rcin: pd.DataFrame,
    rc_channels: List[str],
) -> Tuple[Dict, List[Dict]]:
    """Analyse RC signal quality over the armed flight window."""
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    if "timestamp" not in flight_rcin.columns or len(flight_rcin) < 10:
        metrics.update({
            "rc_dropout_count": 0,
            "rc_dropout_total_s": 0.0,
            "rc_dropouts": [],
            "rc_median_interval_s": 0.0,
            "rc_saturated_channels": [],
        })
        return metrics, issues

    timestamps = flight_rcin["timestamp"].values

    # ── Dropout detection ─────────────────────────────────────────────────────
    gaps = np.diff(timestamps)
    median_gap = float(np.median(gaps))

    # Threshold: 10× median interval, but at least 0.5 s
    dropout_threshold = max(0.5, median_gap * 10.0)

    dropout_events = []
    for idx in np.where(gaps > dropout_threshold)[0]:
        gap_s    = float(gaps[idx])
        ts_start = float(timestamps[idx])
        dropout_events.append({
            "timestamp_s": round(ts_start, 2),
            "duration_s":  round(gap_s, 2),
        })

    total_dropout_s = sum(d["duration_s"] for d in dropout_events)
    metrics["rc_dropout_count"]     = len(dropout_events)
    metrics["rc_dropout_total_s"]   = round(total_dropout_s, 2)
    metrics["rc_dropouts"]          = dropout_events
    metrics["rc_median_interval_s"] = round(median_gap, 4)

    for evt in dropout_events:
        sev = "critical" if evt["duration_s"] > 5.0 else "warning"
        issues.append(_issue(
            sev, "RCL-003",
            f"RC signal dropout: {evt['duration_s']:.1f} s gap "
            f"at T+{evt['timestamp_s']:.1f} s into flight.",
            value=round(evt["duration_s"], 2),
            threshold=round(dropout_threshold, 2),
            timestamp=evt["timestamp_s"],
        ))

    # ── Channel saturation check ──────────────────────────────────────────────
    # Only check channels confirmed as active pilot inputs (bidirectional).
    # Mode switches / VTOL passthrough channels at a fixed extreme are excluded —
    # they are not pilot control inputs and should not raise saturation warnings.
    sat_channels = []
    for col in rc_channels:
        vals = flight_rcin[col].dropna().values
        if len(vals) < 10:
            continue
        pct_at_min = float(np.mean(vals <= 1010))   # near 1000 µs
        pct_at_max = float(np.mean(vals >= 1990))   # near 2000 µs
        if pct_at_min > 0.30 or pct_at_max > 0.30:
            sat_channels.append({
                "channel":    col,
                "pct_at_min": round(pct_at_min * 100, 1),
                "pct_at_max": round(pct_at_max * 100, 1),
            })

    metrics["rc_saturated_channels"] = sat_channels

    for sat in sat_channels:
        issues.append(_issue(
            "warning", "RCL-005",
            f"RC channel {sat['channel']} near limits: "
            f"{sat['pct_at_min']:.0f}% at minimum, "
            f"{sat['pct_at_max']:.0f}% at maximum — "
            "pilot may be flying at control limits.",
            value=round(max(sat["pct_at_min"], sat["pct_at_max"]), 1),
            threshold=30.0,
        ))

    return metrics, issues


# ─────────────────────────────────────────────────────────────────────────────
# Failsafe Event Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_failsafes(
    err_df: Optional[pd.DataFrame],
    arm_time: float,
) -> Tuple[Dict, List[Dict]]:
    """
    Detect failsafe trigger events from ERR messages.
    ECode != 0 = failsafe triggered; ECode == 0 = cleared/resolved.
    """
    issues: List[Dict] = []
    failsafe_events: List[Dict] = []

    if err_df is None or len(err_df) == 0:
        return {"failsafe_event_count": 0, "failsafe_events": []}, issues

    # Scope to armed flight window
    flight_err = (
        err_df[err_df["timestamp"] >= arm_time]
        .copy()
        .reset_index(drop=True)
    )

    if len(flight_err) == 0:
        return {"failsafe_event_count": 0, "failsafe_events": []}, issues

    for _, row in flight_err.iterrows():
        subsys = int(row.get("Subsys", -1))
        ecode  = int(row.get("ECode", 0))
        ts     = float(row.get("timestamp", 0))

        if subsys not in FAILSAFE_SUBSYSTEMS:
            continue
        if ecode == 0:
            continue    # ECode 0 = failsafe cleared — not a trigger event

        fsf_name = FAILSAFE_SUBSYSTEMS[subsys]
        failsafe_events.append({
            "timestamp_s": round(ts, 2),
            "subsystem":   fsf_name,
            "ecode":       ecode,
        })

    # Generate issues per event
    for evt in failsafe_events:
        sub = evt["subsystem"]
        ts  = evt["timestamp_s"]

        if sub == "Battery":
            issues.append(_issue(
                "critical", "FSF-001",
                f"Battery failsafe triggered at T+{ts:.1f}s — "
                "battery was critically low during flight. "
                "Check battery capacity and voltage failsafe thresholds.",
                timestamp=ts,
            ))
        elif sub == "RC/Radio":
            issues.append(_issue(
                "critical", "FSF-002",
                f"RC/Radio failsafe triggered at T+{ts:.1f}s — "
                "RC link was lost during flight. "
                "Check transmitter range, antenna orientation, and RC receiver.",
                timestamp=ts,
            ))
        elif sub == "GPS":
            issues.append(_issue(
                "warning", "FSF-003",
                f"GPS failsafe triggered at T+{ts:.1f}s — "
                "GPS signal lost mid-flight. "
                "Check GPS antenna placement and cable connections.",
                timestamp=ts,
            ))
        elif sub == "EKF":
            issues.append(_issue(
                "critical", "FSF-004",
                f"EKF failsafe triggered at T+{ts:.1f}s — "
                "navigation filter health went critical. "
                "Review GPS, compass, and vibration levels.",
                timestamp=ts,
            ))
        elif sub == "Geofence":
            issues.append(_issue(
                "warning", "FSF-005",
                f"Geofence failsafe triggered at T+{ts:.1f}s — "
                "aircraft breached the geofence boundary.",
                timestamp=ts,
            ))
        elif sub == "Thrust Loss":
            issues.append(_issue(
                "critical", "FSF-006",
                f"Thrust loss check triggered at T+{ts:.1f}s — "
                "one or more motors may have failed or lost output during flight. "
                "Inspect all motors, ESCs, and wiring immediately.",
                timestamp=ts,
            ))
        elif sub == "Crash Check":
            issues.append(_issue(
                "critical", "FSF-007",
                f"Crash check triggered at T+{ts:.1f}s — "
                "ArduPilot detected the aircraft was tumbling, flipping, or in "
                "uncontrolled descent and disarmed the motors. "
                "Do not fly until full structural, motor, and ESC inspection is complete. "
                "Review vibration, EKF, and control logs at this timestamp.",
                timestamp=ts,
            ))

    return {
        "failsafe_event_count": len(failsafe_events),
        "failsafe_events":      failsafe_events,
    }, issues


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict]) -> float:
    score = 100.0

    # RC dropouts (up to -20 total)
    dropout_count = metrics.get("rc_dropout_count", 0)
    if dropout_count > 0:
        score -= min(20.0, dropout_count * 5.0)

    # Failsafe events — weighted by type
    for evt in metrics.get("failsafe_events", []):
        sub = evt["subsystem"]
        if sub == "Crash Check":
            score -= 30.0
        elif sub in ("Battery", "EKF", "Thrust Loss", "RC/Radio"):
            score -= 20.0
        elif sub == "GPS":
            score -= 10.0
        elif sub == "Geofence":
            score -= 5.0

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
