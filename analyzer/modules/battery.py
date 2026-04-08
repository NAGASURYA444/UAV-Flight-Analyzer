"""
Battery & Power Analysis Module
================================
Analyses the BAT/BAT2 message stream to produce a health assessment of the
power system during the flight.

Checks performed
----------------
1.  Starting voltage — was the battery fully charged before flight?
2.  Ending voltage — how much reserve was left at landing / disarm?
3.  Voltage curve shape — healthy = smooth gradual sag; bad = sudden cliff drop
4.  Voltage sag under load — estimates internal resistance from sag events
5.  Average & peak current draw — checked against max continuous rating
6.  Capacity consumed — mAh used vs total capacity
7.  Estimated internal resistance — rising value indicates ageing battery
8.  Low battery failsafe — was a failsafe triggered?

Scoring deductions (from 100)
------------------------------
-10 to -20   Battery not fully charged at start (per % below full)
-20 to -30   Critically low ending voltage (< critical threshold)
-10          Low ending voltage warning (< min threshold)
-10 to -20   High voltage sag (> 2V for 6S)
-15          Average current exceeds rated C-rating
-5           Peak current exceeds rated C-rating momentarily
-10          Capacity consumed > 90% (very little reserve)
-5           Capacity consumed > 80%
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)


def analyse(result: ParseResult, profile: DroneProfile, arm_time: float = 0.0, disarm_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Run battery analysis.

    Parameters
    ----------
    result      : ParseResult from bin_parser
    profile     : DroneProfile with battery thresholds
    arm_time    : Timestamp (s) when the drone was armed (from flight_overview)

    Returns a standardised result dict.
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    # Prefer BAT, fall back to BAT2
    bat_df = result.get("BAT")
    if bat_df is None:
        bat_df = result.get("BAT2")

    # Multi-instance BAT logging (e.g. ArduPlane with two power monitors):
    # use only Instance 0 (primary battery) — same pattern as multi-IMU fix.
    if bat_df is not None and "Instance" in bat_df.columns:
        bat_df = bat_df[bat_df["Instance"] == 0].copy().reset_index(drop=True)

    if bat_df is None or len(bat_df) < 5:
        return {
            "score": 50.0,
            "grade": "C",
            "available": False,
            "issues": [_issue("warning", "BAT-000",
                              "No battery telemetry (BAT/BAT2) found in log.")],
            "metrics": {},
            "summary": "Battery data not available in this log.",
        }

    # Restrict analysis to the armed window
    end_time = disarm_time if disarm_time is not None else float(bat_df["timestamp"].max())
    flight_bat = bat_df[
        (bat_df["timestamp"] >= arm_time) &
        (bat_df["timestamp"] <= end_time)
    ].copy()
    if len(flight_bat) < 5:
        flight_bat = bat_df.copy()   # use entire log if arm event not reliable

    # ── Auto-correct cell count if profile is wrong ───────────────────────────
    # Detect LiPo cell count from first voltage reading.  If the per-cell
    # voltage implied by the profile's cell_count is outside the physical LiPo
    # range (3.2–4.35 V/cell), auto-correct and log a warning so users know.
    if "Volt" in flight_bat.columns and len(flight_bat) > 0:
        first_v = float(flight_bat["Volt"].iloc[0])
        profile = _maybe_fix_cell_count(profile, first_v)

    # ── Column availability ───────────────────────────────────────────────────
    has_volt = "Volt" in flight_bat.columns
    has_curr = "Curr" in flight_bat.columns
    has_curr_tot = "CurrTot" in flight_bat.columns
    has_temp = "Temp" in flight_bat.columns

    # ── 1. Voltage at start and end ───────────────────────────────────────────
    start_voltage = None
    end_voltage = None

    if has_volt:
        start_voltage = float(flight_bat["Volt"].iloc[0])

        # Use the last voltage reading that is above a sane minimum.
        # The very last record(s) in the log are often post-shutdown near-zero
        # artifacts as the flight controller powers down — exclude those.
        min_sane_voltage = profile.battery.cell_count * 2.5   # 2.5 V/cell absolute floor
        sane_volts = flight_bat[flight_bat["Volt"] > min_sane_voltage]["Volt"]
        end_voltage = float(sane_volts.iloc[-1]) if len(sane_volts) > 0 else float(flight_bat["Volt"].iloc[-1])

        metrics["start_voltage_v"] = round(start_voltage, 3)
        metrics["end_voltage_v"] = round(end_voltage, 3)
        metrics["full_voltage_v"] = profile.battery.full_voltage
        metrics["min_voltage_v"] = profile.min_pack_voltage
        metrics["critical_voltage_v"] = profile.critical_pack_voltage

        # Per-cell values for easy interpretation
        n = profile.battery.cell_count
        metrics["start_cell_voltage_v"] = round(start_voltage / n, 3)
        metrics["end_cell_voltage_v"] = round(end_voltage / n, 3)

        # Check start voltage
        full_v = profile.battery.full_voltage
        charge_pct = min(100.0, (start_voltage / full_v) * 100.0)
        metrics["start_charge_pct"] = round(charge_pct, 1)

        # Detect profile cell count mismatch — check per-cell voltage against
        # LiPo physical range (3.20–4.35 V/cell).
        # Note: _maybe_fix_cell_count() already patched profile.battery.cell_count
        # if it was wrong, so n is already the corrected value here.
        # We only warn when the ORIGINAL profile had a mismatch (cell_count was
        # patched = profile.battery.cell_count differs from the YAML value).
        start_cell_v = start_voltage / n
        if not (3.20 <= start_cell_v <= 4.35):
            estimated_cells = round(start_voltage / 4.2)
            issues.append(_issue(
                "warning", "BAT-009",
                f"Battery start voltage ({start_voltage:.2f} V) gives {start_cell_v:.2f} V/cell "
                f"for the configured {n}S — outside the valid LiPo range (3.2–4.35 V/cell). "
                f"Profile 'cell_count' may be wrong; estimated actual: ~{estimated_cells}S. "
                "Update cell_count, nominal_voltage, and full_voltage in the drone profile yaml.",
                value=round(start_cell_v, 3), threshold=4.35,
            ))

        if charge_pct < 95.0:
            deficit = 100.0 - charge_pct
            sev = "warning" if charge_pct >= 85 else "critical"
            issues.append(_issue(
                sev, "BAT-001",
                f"Battery was NOT fully charged at arm — {charge_pct:.1f}% of full voltage "
                f"({start_voltage:.2f} V vs expected {full_v:.2f} V). "
                "Flying on a partially charged battery reduces endurance and increases stress.",
                value=start_voltage, threshold=full_v,
            ))

        # Check end voltage
        crit_v = profile.critical_pack_voltage
        min_v = profile.min_pack_voltage

        if end_voltage <= crit_v:
            issues.append(_issue(
                "critical", "BAT-002",
                f"Battery at CRITICAL voltage at landing: {end_voltage:.2f} V "
                f"(threshold: {crit_v:.2f} V / {profile.battery.critical_cell_voltage:.2f} V/cell). "
                "Risk of cell damage and uncontrolled descent.",
                value=end_voltage, threshold=crit_v,
            ))
        elif end_voltage <= min_v:
            issues.append(_issue(
                "warning", "BAT-002",
                f"Low battery voltage at landing: {end_voltage:.2f} V "
                f"(warning threshold: {min_v:.2f} V / {profile.battery.min_cell_voltage:.2f} V/cell).",
                value=end_voltage, threshold=min_v,
            ))

        # ── 2. Voltage sag analysis ───────────────────────────────────────────
        sag_result = _internal_resistance_analysis(flight_bat, profile)
        metrics.update(sag_result["metrics"])
        issues.extend(sag_result["issues"])

        # ── 3. Voltage curve shape (VTOL-aware) ───────────────────────────────
        curve_result = _voltage_curve_analysis(flight_bat, profile)
        metrics.update(curve_result["metrics"])
        issues.extend(curve_result["issues"])

    # ── 4. Current draw ───────────────────────────────────────────────────────
    if has_curr:
        current_result = _current_analysis(flight_bat, profile)
        metrics.update(current_result["metrics"])
        issues.extend(current_result["issues"])

    # ── 5. Capacity consumed ──────────────────────────────────────────────────
    if has_curr_tot:
        cap_result = _capacity_analysis(flight_bat, profile)
        # Some ArduPlane configs log a CurrTot column that stays at 0 (not wired).
        # If consumed is 0 but current data exists, fall back to integration.
        if cap_result["metrics"].get("capacity_consumed_mah", 0) == 0 \
                and has_curr and "timestamp" in flight_bat.columns:
            cap_result = _capacity_from_integration(flight_bat, profile)
        metrics.update(cap_result["metrics"])
        issues.extend(cap_result["issues"])
    elif has_curr and "timestamp" in flight_bat.columns:
        # Integrate current over time to estimate consumed capacity
        cap_result = _capacity_from_integration(flight_bat, profile)
        metrics.update(cap_result["metrics"])
        issues.extend(cap_result["issues"])

    # ── 6. Temperature ────────────────────────────────────────────────────────
    if has_temp:
        temp_result = _temperature_analysis(flight_bat)
        metrics.update(temp_result["metrics"])
        issues.extend(temp_result["issues"])

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues, profile)
    grade = _grade(score)

    # Summary line
    parts = []
    if start_voltage is not None:
        parts.append(f"Start: {start_voltage:.2f} V")
    if end_voltage is not None:
        parts.append(f"End: {end_voltage:.2f} V")
    if "capacity_consumed_mah" in metrics:
        parts.append(f"Used: {metrics['capacity_consumed_mah']:.0f} mAh")
    if "avg_current_a" in metrics:
        parts.append(f"Avg current: {metrics['avg_current_a']:.1f} A")
    summary = "  |  ".join(parts) if parts else "Battery data present but limited."

    return {
        "score": round(score, 1),
        "grade": grade,
        "available": True,
        "issues": issues,
        "metrics": metrics,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis sub-functions
# ─────────────────────────────────────────────────────────────────────────────

def _internal_resistance_analysis(bat_df: pd.DataFrame, profile: DroneProfile) -> Dict:
    """
    Estimate battery internal resistance from current-voltage correlation.

    Method: at each high-current event, measure how much the voltage drops
    relative to the local baseline. R_internal = delta_V / delta_I.
    This correctly measures instantaneous sag under load — NOT total flight
    depletion — giving a true battery health indicator.

    LiPo internal resistance benchmarks (per cell):
        < 5 mOhm   Excellent (new battery)
        5–10 mOhm  Good
        10–20 mOhm Aging — monitor
        > 20 mOhm  Replace recommended
        > 30 mOhm  Critical — replace immediately
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    if "Curr" not in bat_df.columns or "Volt" not in bat_df.columns:
        return {"issues": issues, "metrics": metrics}

    curr = bat_df["Curr"].values.astype(float)
    volt = bat_df["Volt"].values.astype(float)

    if len(curr) < 30:
        return {"issues": issues, "metrics": metrics}

    window = 8
    r_estimates: List[float] = []

    for i in range(window, len(curr) - 2):
        i_baseline = float(np.mean(curr[i - window:i]))
        v_baseline = float(np.mean(volt[i - window:i]))
        delta_i = curr[i] - i_baseline
        delta_v = volt[i] - v_baseline

        # Only use events where current increased significantly (> 5A step)
        if delta_i > 5.0:
            r = -delta_v / delta_i   # R = -dV/dI  (voltage drops → negative dV)
            if 0.001 < r < 2.0:      # Sanity: 1 mOhm to 2 Ω
                r_estimates.append(r)

    if len(r_estimates) < 5:
        # Not enough current events (e.g. very smooth fixed-wing cruise) — skip
        metrics["internal_resistance_available"] = False
        return {"issues": issues, "metrics": metrics}

    r_ohm = float(np.median(r_estimates))
    r_mohm = r_ohm * 1000.0
    r_per_cell_mohm = r_mohm / profile.battery.cell_count

    metrics["internal_resistance_available"] = True
    metrics["internal_resistance_mohm"] = round(r_mohm, 2)
    metrics["internal_resistance_per_cell_mohm"] = round(r_per_cell_mohm, 2)
    metrics["r_sample_count"] = len(r_estimates)

    if r_per_cell_mohm > 30:
        issues.append(_issue(
            "critical", "BAT-003",
            f"Critical internal resistance: {r_per_cell_mohm:.1f} mOhm/cell "
            f"(total: {r_mohm:.1f} mOhm). Battery must be replaced — risk of cell failure in flight.",
            value=round(r_per_cell_mohm, 2), threshold=30,
        ))
    elif r_per_cell_mohm > 20:
        issues.append(_issue(
            "warning", "BAT-003",
            f"Elevated internal resistance: {r_per_cell_mohm:.1f} mOhm/cell "
            f"(total: {r_mohm:.1f} mOhm). Battery aging — check cell balance and plan replacement.",
            value=round(r_per_cell_mohm, 2), threshold=20,
        ))
    elif r_per_cell_mohm > 10:
        issues.append(_issue(
            "info", "BAT-003",
            f"Internal resistance: {r_per_cell_mohm:.1f} mOhm/cell — monitor trend over flights.",
            value=round(r_per_cell_mohm, 2), threshold=10,
        ))

    return {"issues": issues, "metrics": metrics}


def _voltage_curve_analysis(bat_df: pd.DataFrame, profile: DroneProfile) -> Dict:
    """
    Detect abrupt voltage drops NOT explained by current draw.
    For VTOL drones, drops coinciding with current spikes (transitions) are filtered out
    as they are normal hover↔cruise mode-transition behavior.
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    volts = bat_df["Volt"].values
    if len(volts) < 20:
        return {"issues": issues, "metrics": metrics}

    dv = np.diff(volts)
    dv_std = float(np.std(dv))
    drop_threshold = -3.0 * dv_std if dv_std > 0.001 else -0.5
    drop_indices = np.where(dv < drop_threshold)[0]

    # Filter drops that coincide with current spikes — these are normal IR drops
    # under load (high current → more voltage sag = expected physics).
    # A genuine loose connector or weak cell causes drops INDEPENDENT of current level.
    # This filtering applies to ALL drone types: VTOL mode transitions, fixed-wing
    # throttle changes, and multirotor waypoint maneuvers all create current transients.
    if "Curr" in bat_df.columns and len(drop_indices) > 0:
        curr = bat_df["Curr"].values.astype(float)
        curr_mean = float(np.mean(curr))
        curr_std = float(np.std(curr))
        curr_spike_thresh = curr_mean + curr_std

        genuine_drops = []
        for idx in drop_indices:
            # Check a small window around the drop for a current spike
            w_start = max(0, idx - 3)
            w_end = min(len(curr), idx + 4)
            if float(np.max(curr[w_start:w_end])) < curr_spike_thresh:
                genuine_drops.append(idx)   # drop without current spike = genuine anomaly

        n_drops = len(genuine_drops)
        n_filtered = len(drop_indices) - n_drops
        metrics["current_coincident_drops_filtered"] = n_filtered
    else:
        n_drops = int(len(drop_indices))

    metrics["sudden_voltage_drops"] = n_drops
    metrics["voltage_curve_std"] = round(dv_std, 5)

    if n_drops > 3:
        filter_note = " (current-coincident drops already excluded)"
        issues.append(_issue(
            "warning", "BAT-004",
            f"Irregular voltage curve: {n_drops} unexplained voltage drops detected{filter_note}. "
            "Possible loose battery connector, weak cell, or ESC brownout.",
            value=n_drops, threshold=3,
        ))

    return {"issues": issues, "metrics": metrics}


def _current_analysis(bat_df: pd.DataFrame, profile: DroneProfile) -> Dict:
    """Check average and peak current draw against the battery's C-rating."""
    issues: List[Dict] = []
    metrics: Dict = {}

    curr = bat_df["Curr"].values
    if len(curr) == 0:
        return {"issues": issues, "metrics": metrics}

    avg_curr = float(np.mean(curr))
    peak_curr = float(np.percentile(curr, 99))   # 99th percentile to avoid spikes
    max_rated = profile.battery.max_continuous_current_a

    metrics["avg_current_a"] = round(avg_curr, 2)
    metrics["peak_current_a"] = round(peak_curr, 2)
    metrics["max_rated_current_a"] = max_rated

    # C-rate = current / capacity_ah
    capacity_ah = profile.battery.capacity_mah / 1000.0
    avg_c_rate = avg_curr / capacity_ah if capacity_ah > 0 else 0.0
    peak_c_rate = peak_curr / capacity_ah if capacity_ah > 0 else 0.0
    metrics["avg_c_rate"] = round(avg_c_rate, 2)
    metrics["peak_c_rate"] = round(peak_c_rate, 2)

    if avg_curr > max_rated * 0.85:
        issues.append(_issue(
            "critical", "BAT-005",
            f"Average current draw ({avg_curr:.1f} A) is near or above the battery's "
            f"max continuous rating ({max_rated:.0f} A). Risk of overheating and damage.",
            value=round(avg_curr, 2), threshold=max_rated,
        ))
    elif avg_curr > max_rated * 0.70:
        issues.append(_issue(
            "warning", "BAT-005",
            f"Elevated average current draw ({avg_curr:.1f} A) — "
            f"{avg_curr / max_rated * 100:.0f}% of rated max ({max_rated:.0f} A).",
            value=round(avg_curr, 2), threshold=max_rated,
        ))

    if peak_curr > max_rated:
        issues.append(_issue(
            "warning", "BAT-006",
            f"Peak current ({peak_curr:.1f} A) exceeded battery rating ({max_rated:.0f} A). "
            "Transient overloads reduce battery life.",
            value=round(peak_curr, 2), threshold=max_rated,
        ))

    return {"issues": issues, "metrics": metrics}


def _capacity_analysis(bat_df: pd.DataFrame, profile: DroneProfile) -> Dict:
    """Analyse CurrTot (total consumed mAh) logged directly by ArduPilot."""
    issues: List[Dict] = []
    metrics: Dict = {}

    consumed = float(bat_df["CurrTot"].iloc[-1]) - float(bat_df["CurrTot"].iloc[0])
    consumed = max(0.0, consumed)
    total_cap = profile.battery.capacity_mah
    used_pct = (consumed / total_cap * 100.0) if total_cap > 0 else 0.0

    metrics["capacity_consumed_mah"] = round(consumed, 1)
    metrics["capacity_total_mah"] = total_cap
    metrics["capacity_used_pct"] = round(used_pct, 1)
    metrics["capacity_remaining_pct"] = round(100.0 - used_pct, 1)

    if used_pct > 90:
        issues.append(_issue(
            "critical", "BAT-007",
            f"Battery deeply discharged: {used_pct:.1f}% of capacity consumed. "
            "Flying to < 10% reserve is damaging for LiPo cells.",
            value=round(used_pct, 1), threshold=90,
        ))
    elif used_pct > 80:
        issues.append(_issue(
            "warning", "BAT-007",
            f"High capacity utilisation: {used_pct:.1f}% consumed. "
            "Consider landing at <= 80% for battery longevity.",
            value=round(used_pct, 1), threshold=80,
        ))

    return {"issues": issues, "metrics": metrics}


def _capacity_from_integration(bat_df: pd.DataFrame, profile: DroneProfile) -> Dict:
    """Estimate consumed capacity by integrating current over time (∫I dt)."""
    issues: List[Dict] = []
    metrics: Dict = {}

    try:
        ts = bat_df["timestamp"].values
        curr = bat_df["Curr"].values
        dt = np.diff(ts)
        i_avg = (curr[:-1] + curr[1:]) / 2.0          # trapezoidal
        consumed_as = float(np.sum(i_avg * dt))        # Ampere-seconds
        consumed_mah = consumed_as / 3.6               # convert to mAh

        total_cap = profile.battery.capacity_mah
        used_pct = (consumed_mah / total_cap * 100.0) if total_cap > 0 else 0.0

        metrics["capacity_consumed_mah"] = round(consumed_mah, 1)
        metrics["capacity_consumed_source"] = "integrated"
        metrics["capacity_total_mah"] = total_cap
        metrics["capacity_used_pct"] = round(used_pct, 1)
        metrics["capacity_remaining_pct"] = round(100.0 - used_pct, 1)

        if used_pct > 90:
            issues.append(_issue(
                "critical", "BAT-007",
                f"Estimated battery deeply discharged: {used_pct:.1f}% consumed (by current integration).",
                value=round(used_pct, 1), threshold=90,
            ))
        elif used_pct > 80:
            issues.append(_issue(
                "warning", "BAT-007",
                f"Estimated high capacity utilisation: {used_pct:.1f}% consumed.",
                value=round(used_pct, 1), threshold=80,
            ))
    except Exception as exc:
        logger.debug("Capacity integration failed: %s", exc)

    return {"issues": issues, "metrics": metrics}


def _temperature_analysis(bat_df: pd.DataFrame) -> Dict:
    """Check battery temperature during flight."""
    issues: List[Dict] = []
    metrics: Dict = {}

    temps = bat_df["Temp"].dropna()
    if len(temps) == 0:
        return {"issues": issues, "metrics": metrics}

    max_temp = float(temps.max())
    avg_temp = float(temps.mean())
    metrics["battery_max_temp_c"] = round(max_temp, 1)
    metrics["battery_avg_temp_c"] = round(avg_temp, 1)

    if max_temp > 60.0:
        issues.append(_issue(
            "critical", "BAT-008",
            f"Battery overheating: max temperature {max_temp:.1f} °C (limit: 60 °C). "
            "Risk of thermal runaway.",
            value=round(max_temp, 1), threshold=60,
        ))
    elif max_temp > 45.0:
        issues.append(_issue(
            "warning", "BAT-008",
            f"Elevated battery temperature: {max_temp:.1f} °C. Monitor thermal management.",
            value=round(max_temp, 1), threshold=45,
        ))

    return {"issues": issues, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict], profile: DroneProfile) -> float:
    score = 100.0

    # Start charge
    charge_pct = metrics.get("start_charge_pct", 100.0)
    if charge_pct < 95:
        score -= min(15.0, (95 - charge_pct) * 0.5)

    # End voltage
    end_v = metrics.get("end_voltage_v")
    if end_v is not None:
        crit_v = profile.critical_pack_voltage
        min_v = profile.min_pack_voltage
        if end_v <= crit_v:
            score -= 25.0
        elif end_v <= min_v:
            score -= 10.0

    # Internal resistance (replaces old sag metric)
    r_per_cell = metrics.get("internal_resistance_per_cell_mohm", 0.0)
    if r_per_cell > 30:
        score -= 20.0
    elif r_per_cell > 20:
        score -= 10.0
    elif r_per_cell > 10:
        score -= 3.0

    # Capacity used
    used_pct = metrics.get("capacity_used_pct", 0.0)
    if used_pct > 90:
        score -= 10.0
    elif used_pct > 80:
        score -= 5.0

    # Current
    avg_curr = metrics.get("avg_current_a", 0.0)
    max_rated = profile.battery.max_continuous_current_a
    if max_rated > 0 and avg_curr > max_rated * 0.85:
        score -= 15.0

    # Additional penalty for issues NOT already covered by the metric-based
    # deductions above (BAT-001/002/003/005/006/007 are all explicitly scored
    # through their corresponding metrics; only BAT-004/008/009 reach here).
    _EXPLICIT_CODES = {"BAT-001", "BAT-002", "BAT-003", "BAT-005", "BAT-006", "BAT-007"}
    for issue in issues:
        if issue.get("code") in _EXPLICIT_CODES:
            continue
        if issue.get("severity") == "critical":
            score -= 10.0
        elif issue.get("severity") == "warning":
            score -= 3.0

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cell_count(voltage: float) -> Optional[int]:
    """
    Infer LiPo/LiHV cell count from pack voltage.
    Checks standard cell counts 1S–14S.
    Valid per-cell range: 3.20 V (discharged) to 4.35 V (LiHV full).
    Returns None if no cell count gives a plausible per-cell voltage.
    """
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]:
        v_per_cell = voltage / n
        if 3.20 <= v_per_cell <= 4.35:
            return n
    return None


def _maybe_fix_cell_count(profile: DroneProfile, start_voltage: float) -> DroneProfile:
    """
    If the profile's cell_count gives an implausible per-cell voltage,
    auto-detect the correct count and patch the profile in-place.
    Returns the (possibly patched) profile.
    """
    import copy
    n = profile.battery.cell_count
    v_per_cell = start_voltage / n if n > 0 else 0.0
    if 3.20 <= v_per_cell <= 4.35:
        return profile   # Profile is correct — nothing to do

    detected = _detect_cell_count(start_voltage)
    if detected is None or detected == n:
        return profile   # Cannot determine better value

    # Patch a copy so we don't mutate the shared profile object
    profile = copy.deepcopy(profile)
    logger.warning(
        "Profile cell_count=%d gives %.2f V/cell from %.2f V pack — "
        "physically implausible. Auto-correcting to %dS.",
        n, v_per_cell, start_voltage, detected,
    )
    profile.battery.cell_count = detected
    profile.battery.nominal_voltage = round(detected * 3.7, 1)
    profile.battery.full_voltage = round(detected * 4.2, 1)
    return profile


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
