"""
Power Rail Health Assessment Module
=====================================
Phase 4.5 — Analyses the main battery power bus for dynamic voltage sag
under high-current loads and checks POWR.Flags for FC brownout events.

This module complements fc_health.py (which monitors the FC's internal 5 V
and servo rails). power_rail.py focuses on the HIGH-CURRENT MAIN BUS that
feeds the ESCs and motors — the power backbone of the aircraft.

Checks performed
----------------
1.  POWR.Flags brownout detection
      — POWR.Flags bit 4 (0x10, POWER_STATUS_VCC_CHANGED) set during armed flight
        = FC Vcc changed by > 200 mV — real in-flight brownout indicator.
        NOTE: bit 0 (0x01) is USB_CONNECTED (always set when laptop is connected)
        and must NOT be used. Bit 4 is a sticky flag: detect first 0→1 transition,
        not sample count. Pre-arm check skips events that occurred before arming.

2.  Peak-current voltage sag
      — Voltage drop during top-5% current demand vs mean flight voltage.
      — Formula: sag_pct = (V_mean − V_at_peak_current) / V_mean × 100
      — High sag → battery internal resistance too high, undersized wiring,
        or loose power connector.

3.  Sustained high-current sag
      — Mean voltage during top-10% current periods vs overall mean.
      — Detects whether the battery consistently sags during heavy load
        phases (climb, VTOL hover, rapid acceleration).

4.  Power bus voltage noise
      — Coefficient of variation (std / mean) of main bus voltage.
      — Elevated CV without sag: possible cell imbalance, arcing connector,
        or ESC switching noise feeding back onto the power bus.

Scoring (from 100)
------------------
−30   Brownout event detected (POWR.Flags bit 4 VCC_CHANGED, in-flight)   [critical]
−15   Peak-current sag critical  (> 15%)                                   [critical]
−10   Peak-current sag warning   (> 8%)                                    [warning]
− 8   Sustained high-current sag (> 5%)                                    [warning]
− 7   Power bus noise elevated   (CV > 0.06)                               [warning]
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

SAG_PEAK_WARN   = 8.0    # % — peak-current sag warning
SAG_PEAK_CRIT   = 15.0   # % — peak-current sag critical
SAG_SUSTAIN_WARN = 5.0   # % — sustained high-current sag warning
BUS_NOISE_WARN  = 0.06   # CV — power bus noise warning

# Top-N% current thresholds for sag analysis
_PEAK_PCT    = 95   # top-5% samples = peak current demand
_SUSTAIN_PCT = 90   # top-10% samples = sustained high-current phase
_MIN_SAMPLES = 10   # minimum samples needed for a meaningful sag estimate


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Power Rail Health analysis.

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
    end_time = disarm_time if disarm_time is not None else float("inf")

    bat_df  = result.get("BAT")
    powr_df = result.get("POWR")

    if bat_df is None or bat_df.empty:
        return _unavailable("BAT messages not found — power rail analysis unavailable.")

    # ── Filter to armed flight window ─────────────────────────────────────────
    # For multi-monitor boards (fixed-wing): use Instance 0 (main battery)
    if "Instance" in bat_df.columns:
        bat_df = bat_df[bat_df["Instance"] == 0].copy().reset_index(drop=True)

    flight_bat = bat_df[
        (bat_df["timestamp"] >= arm_time) & (bat_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_bat) < _MIN_SAMPLES:
        return _unavailable(
            f"Too few BAT samples in armed window ({len(flight_bat)}) "
            "— power rail analysis unavailable."
        )

    # Detect required columns
    volt_col = _col(flight_bat, ["Volt", "V", "voltage"])
    curr_col = _col(flight_bat, ["Curr", "I", "current"])

    if volt_col is None:
        return _unavailable("BAT voltage column not found in log.")

    volt_arr = flight_bat[volt_col].values.astype(float)

    # Guard against post-shutdown low-voltage artifacts (< 2.5 V/cell)
    cell_count = profile.battery.cell_count
    volt_arr = volt_arr[volt_arr > cell_count * 2.5]

    if len(volt_arr) < _MIN_SAMPLES:
        return _unavailable("Insufficient valid voltage samples after artifact filtering.")

    volt_mean = float(np.mean(volt_arr))
    volt_std  = float(np.std(volt_arr))
    volt_min  = float(np.min(volt_arr))
    volt_max  = float(np.max(volt_arr))
    volt_cv   = volt_std / (volt_mean + 1e-9)

    metrics.update({
        "volt_mean_v":    round(volt_mean, 3),
        "volt_min_v":     round(volt_min, 3),
        "volt_max_v":     round(volt_max, 3),
        "volt_std_v":     round(volt_std, 4),
        "volt_cv":        round(volt_cv, 4),
    })

    # ── 1. POWR.Flags brownout detection ─────────────────────────────────────
    brownout_issues, brownout_deduction, brownout_metrics = _analyse_brownout(
        powr_df, arm_time, end_time
    )
    issues.extend(brownout_issues)
    metrics.update(brownout_metrics)
    score -= brownout_deduction

    # ── 2. Dynamic voltage sag (requires current data) ────────────────────────
    if curr_col is not None:
        curr_arr = flight_bat[curr_col].values.astype(float)

        # Align current array to filtered voltage array length if needed
        # (volt_arr was filtered by voltage threshold; use the same index filter)
        valid_mask = flight_bat[volt_col].values.astype(float) > cell_count * 2.5
        curr_filtered = curr_arr[valid_mask]

        if len(curr_filtered) >= _MIN_SAMPLES:
            sag_issues, sag_deduction, sag_metrics = _analyse_sag(
                volt_arr, curr_filtered, volt_mean
            )
            issues.extend(sag_issues)
            metrics.update(sag_metrics)
            score -= sag_deduction
        else:
            metrics["sag_available"] = False
    else:
        metrics["sag_available"] = False
        logger.debug("BAT current column not found — sag analysis skipped.")

    # ── 3. Power bus voltage noise ────────────────────────────────────────────
    if volt_cv > BUS_NOISE_WARN:
        score -= 7.0
        issues.append(_issue(
            "warning", "PWR-004",
            f"Power bus voltage noise elevated: CV {volt_cv:.3f} "
            f"(threshold {BUS_NOISE_WARN}). "
            "Possible causes: cell imbalance, loose XT connector, or ESC switching "
            "noise feeding back onto the power bus. Inspect battery connector and "
            "power distribution board connections.",
            value=round(volt_cv, 4),
            threshold=BUS_NOISE_WARN,
        ))
    metrics["bus_noise_cv"] = round(volt_cv, 4)

    score = max(0.0, min(100.0, score))

    from analyzer.scoring.engine import _grade
    grade = _grade(score)

    # ── Summary ───────────────────────────────────────────────────────────────
    brownout_n = brownout_metrics.get("brownout_count", 0)
    sag_peak   = metrics.get("peak_sag_pct")
    sag_sus    = metrics.get("sustained_sag_pct")

    parts = [f"Bus mean {volt_mean:.1f} V  /  min {volt_min:.1f} V"]
    if sag_peak is not None:
        parts.append(f"peak-load sag {sag_peak:.1f}%")
    if sag_sus is not None:
        parts.append(f"sustained sag {sag_sus:.1f}%")
    parts.append(f"noise CV {volt_cv:.3f}")
    if brownout_n > 0:
        parts.append(f"[{brownout_n} BROWNOUT EVENT(S)]")

    summary = "  |  ".join(parts)

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ── Brownout analysis ─────────────────────────────────────────────────────────

def _analyse_brownout(
    powr_df: Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
) -> Tuple[List[Dict], float, Dict]:
    """
    Check POWR.Flags bit 4 (POWER_STATUS_VCC_CHANGED) for FC Vcc brownout events.

    ArduPilot POWR.Flags bit map (AP_HAL::Util::PowerStatusFlag):
      bit 0 (0x01): USB connected          — TRANSIENT, not a fault indicator
      bit 1 (0x02): Servo rail valid        — transient
      bit 2 (0x04): Servo brick valid       — transient
      bit 3 (0x08): Servo voltage changed   — NON-TRANSIENT brownout indicator
      bit 4 (0x10): Vcc changed             — NON-TRANSIENT Vcc brownout indicator ← used here

    Bits 3 and 4 are STICKY: once set they remain set for the rest of the power cycle.
    The correct method is to detect the FIRST TRANSITION from 0→1 (the actual event),
    not count how many samples have the bit set (that equals the whole flight if ever set).

    To avoid false positives from pre-flight ground power / USB connect events, we check
    whether the bit was already set in the 30 s window BEFORE arming. If pre-existing,
    the event is pre-flight and not counted as an in-flight brownout.
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}
    deduction = 0.0

    metrics["brownout_check_available"] = False
    metrics["brownout_count"] = 0

    if powr_df is None or powr_df.empty:
        return issues, deduction, metrics

    flight_powr = powr_df[
        (powr_df["timestamp"] >= arm_time) &
        (powr_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_powr) == 0:
        return issues, deduction, metrics

    flags_col = _col(flight_powr, ["Flags", "Flags1", "flags"])
    if flags_col is None:
        return issues, deduction, metrics

    metrics["brownout_check_available"] = True

    try:
        flags_arr = flight_powr[flags_col].values.astype(int)
    except (ValueError, TypeError):
        return issues, deduction, metrics

    # Bit 4 (0x10) = POWER_STATUS_VCC_CHANGED — Vcc changed by >200 mV
    # This is the correct brownout indicator (NOT bit 0 which is USB_CONNECTED).
    vcc_changed = (flags_arr & 0x10).astype(bool)

    if not np.any(vcc_changed):
        # Bit 4 never set during armed flight — no Vcc change event
        metrics["brownout_count"] = 0
        return issues, deduction, metrics

    # Bit 4 is set at some point during armed flight.
    # Check if it was ALREADY SET before arming (pre-flight ground power / USB sequence).
    # Look at POWR samples in the 30 s window before arm_time.
    pre_arm_set = False
    pre_arm_powr = powr_df[
        (powr_df["timestamp"] >= arm_time - 30.0) &
        (powr_df["timestamp"] < arm_time)
    ]
    if len(pre_arm_powr) > 0 and flags_col in pre_arm_powr.columns:
        try:
            pre_flags = pre_arm_powr[flags_col].values.astype(int)
            pre_arm_set = bool(np.any((pre_flags & 0x10).astype(bool)))
        except (ValueError, TypeError):
            pass

    if pre_arm_set:
        # Bit 4 was set before arming — pre-existing event (ground power / bench test).
        # Do NOT count as an in-flight brownout; note in metrics only.
        metrics["brownout_count"] = 0
        metrics["brownout_pre_existing"] = True
        logger.debug("POWR.Flags VCC_CHANGED already set before arming — not counted as in-flight brownout.")
        return issues, deduction, metrics

    # Bit 4 first appeared DURING armed flight → real in-flight Vcc change event.
    first_idx = int(np.argmax(vcc_changed))
    first_ts  = float(flight_powr["timestamp"].iloc[first_idx])
    brownout_count = 1   # one event (sticky flag — triggers once per power cycle)

    metrics["brownout_count"] = brownout_count
    metrics["brownout_first_ts_s"] = round(first_ts, 2)

    deduction += 30.0
    issues.append(_issue(
        "critical", "PWR-001",
        f"FC Vcc brownout event detected: POWR.Flags VCC_CHANGED bit set "
        f"at T+{first_ts:.1f} s during armed flight. The flight controller 5 V rail "
        "experienced a voltage change exceeding 200 mV — indicates a transient brownout "
        "or power module instability. Cross-check with FC Health (VCC min value). "
        "Inspect power module output, main battery connector, and filter capacitor health. "
        "Do not fly until resolved.",
        value=1,
        timestamp=first_ts,
    ))

    return issues, deduction, metrics


# ── Voltage sag analysis ──────────────────────────────────────────────────────

def _analyse_sag(
    volt_arr: np.ndarray,
    curr_arr: np.ndarray,
    volt_mean: float,
) -> Tuple[List[Dict], float, Dict]:
    """Analyse voltage sag under peak and sustained high-current loads."""
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}
    deduction = 0.0

    metrics["sag_available"] = True

    # ── Peak-current sag (top 5% current demand) ──────────────────────────────
    peak_thresh = float(np.percentile(curr_arr, _PEAK_PCT))
    peak_mask   = curr_arr >= peak_thresh

    if np.sum(peak_mask) >= _MIN_SAMPLES:
        volt_at_peak = float(np.mean(volt_arr[peak_mask]))
        peak_sag_pct = (volt_mean - volt_at_peak) / (volt_mean + 1e-9) * 100.0
        peak_sag_pct = max(0.0, peak_sag_pct)   # clamp; can't be negative

        metrics["peak_sag_pct"]           = round(peak_sag_pct, 2)
        metrics["volt_at_peak_current_v"] = round(volt_at_peak, 3)
        metrics["peak_current_threshold_a"] = round(peak_thresh, 1)

        if peak_sag_pct > SAG_PEAK_CRIT:
            deduction += 15.0
            issues.append(_issue(
                "critical", "PWR-002",
                f"Severe voltage sag under peak current: {peak_sag_pct:.1f}% "
                f"(threshold {SAG_PEAK_CRIT}%). Bus drops to {volt_at_peak:.2f} V "
                f"when current exceeds {peak_thresh:.0f} A. "
                "Battery internal resistance is critically high or wiring is undersized. "
                "Check battery health (cycle count, IR test), connector torque, and "
                "main power wire gauge.",
                value=round(peak_sag_pct, 2),
                threshold=SAG_PEAK_CRIT,
            ))
        elif peak_sag_pct > SAG_PEAK_WARN:
            deduction += 10.0
            issues.append(_issue(
                "warning", "PWR-002",
                f"Voltage sag under peak current: {peak_sag_pct:.1f}% "
                f"(threshold {SAG_PEAK_WARN}%). Bus drops to {volt_at_peak:.2f} V "
                f"when current exceeds {peak_thresh:.0f} A. "
                "Check battery connector tightness and consider IR-testing the battery "
                "to detect early cell degradation.",
                value=round(peak_sag_pct, 2),
                threshold=SAG_PEAK_WARN,
            ))
    else:
        metrics["peak_sag_pct"] = None

    # ── Sustained high-current sag (top 10% current) ──────────────────────────
    sustain_thresh = float(np.percentile(curr_arr, _SUSTAIN_PCT))
    sustain_mask   = curr_arr >= sustain_thresh

    if np.sum(sustain_mask) >= _MIN_SAMPLES:
        volt_at_sustain  = float(np.mean(volt_arr[sustain_mask]))
        sustain_sag_pct  = (volt_mean - volt_at_sustain) / (volt_mean + 1e-9) * 100.0
        sustain_sag_pct  = max(0.0, sustain_sag_pct)

        metrics["sustained_sag_pct"]              = round(sustain_sag_pct, 2)
        metrics["volt_at_sustained_current_v"]    = round(volt_at_sustain, 3)
        metrics["sustained_current_threshold_a"]  = round(sustain_thresh, 1)

        if sustain_sag_pct > SAG_SUSTAIN_WARN:
            deduction += 8.0
            issues.append(_issue(
                "warning", "PWR-003",
                f"Sustained voltage sag during high-current phases: {sustain_sag_pct:.1f}% "
                f"(threshold {SAG_SUSTAIN_WARN}%). Average bus voltage drops to "
                f"{volt_at_sustain:.2f} V during periods above {sustain_thresh:.0f} A. "
                "Indicates the battery is consistently sagging under operational load. "
                "Check battery condition and ensure current draw is within battery rating.",
                value=round(sustain_sag_pct, 2),
                threshold=SAG_SUSTAIN_WARN,
            ))
    else:
        metrics["sustained_sag_pct"] = None

    return issues, deduction, metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "score":     100.0,
        "grade":     "A",
        "available": False,
        "issues":    [],
        "metrics":   {"unavailable_reason": reason},
        "summary":   f"Power rail analysis unavailable: {reason}",
    }
