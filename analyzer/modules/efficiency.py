"""
Power Efficiency Analysis Module
==================================
Computes energy and power efficiency metrics from battery current/voltage data
and GPS distance, with VTOL-aware hover vs cruise phase separation.

Checks performed
----------------
1.  Energy consumed  — Wh and mAh actually used during flight
2.  Distance flown   — GPS-derived total distance (km)
3.  Wh/km            — primary efficiency metric (lower = better)
4.  mAh/km           — secondary efficiency metric
5.  Average power    — Watts consumed during flight
6.  Wh/min           — energy rate (useful for loiter / hover missions)
7.  VTOL phase split — hover vs cruise time, energy, and per-phase efficiency

Scoring (from 100)
------------------
Efficiency is scored relative to what the drone profile expects.
The profile supplies  expected_range_km  and the battery capacity (Wh).
Expected Wh/km = total_Wh / expected_range_km.

Deductions
----------
-10   Wh/km > 120 % of expected (flight less efficient than baseline)
-20   Wh/km > 150 % of expected (significantly inefficient)
- 5   Average power > 90 % of rated max power
-10   Average power > rated max power
- 5   No distance data (GPS unavailable)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run power efficiency analysis.

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

    # ── Get battery data ──────────────────────────────────────────────────────
    bat_df = result.get("BAT")
    if bat_df is None:
        bat_df = result.get("BAT2")
    # Multi-instance BAT logging: use only Instance 0 (primary battery).
    if bat_df is not None and "Instance" in bat_df.columns:
        bat_df = bat_df[bat_df["Instance"] == 0].copy().reset_index(drop=True)

    if bat_df is None or bat_df.empty:
        return _unavailable("No BAT messages found — efficiency analysis unavailable")

    end_time = disarm_time if disarm_time else bat_df["timestamp"].max()
    flight_bat = bat_df[
        (bat_df["timestamp"] >= arm_time) & (bat_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_bat) < 10:
        return _unavailable(f"Too few BAT samples in armed window ({len(flight_bat)})")

    # ── Identify voltage / current columns ───────────────────────────────────
    volt_col = _find_col(flight_bat, ["Volt", "VoltR", "V"])
    curr_col = _find_col(flight_bat, ["Curr", "CurrTot", "I"])
    if volt_col is None or curr_col is None:
        return _unavailable(f"Voltage or current column missing from BAT. "
                            f"Available: {list(flight_bat.columns)}")

    # ── Compute energy (Wh) via trapezoidal integration ──────────────────────
    timestamps   = flight_bat["timestamp"].values
    volts        = flight_bat[volt_col].values.astype(float)
    raw_currents = flight_bat[curr_col].values.astype(float)

    volts = np.clip(volts, 0, None)   # negative voltage is unphysical

    # Use raw (unclipped) currents for both energy and mAh.
    # Clipping negative sensor noise to 0 inflates both metrics on low-current
    # flights where noise magnitude approaches the signal (e.g. fixed-wing at 2A).
    # Raw currents let noise cancel over time in the trapezoidal integral.
    power_w  = volts * raw_currents             # instantaneous watts (may be slightly negative)
    dt_h     = np.diff(timestamps) / 3600.0    # time steps in hours
    energy_wh = float(np.sum(0.5 * (power_w[:-1] + power_w[1:]) * dt_h))
    energy_wh = max(0.0, energy_wh)            # total energy can't be negative

    mah_consumed = float(np.sum(0.5 * (raw_currents[:-1] + raw_currents[1:]) * dt_h * 1000))
    mah_consumed = max(0.0, mah_consumed)      # total charge can't be negative

    # Flight duration (hours and minutes)
    flight_dur_s   = float(timestamps[-1] - timestamps[0])
    flight_dur_min = flight_dur_s / 60.0
    flight_dur_h   = flight_dur_s / 3600.0

    avg_power_w = energy_wh / flight_dur_h if flight_dur_h > 0 else 0.0
    wh_per_min  = energy_wh / flight_dur_min if flight_dur_min > 0 else 0.0

    metrics.update({
        "energy_wh":        round(energy_wh, 2),
        "mah_consumed":     round(mah_consumed, 0),
        "avg_power_w":      round(avg_power_w, 1),
        "wh_per_min":       round(wh_per_min, 3),
        "flight_dur_min":   round(flight_dur_min, 1),
    })

    # ── GPS distance ─────────────────────────────────────────────────────────
    distance_km, gps_available = _gps_distance(result, arm_time, end_time)
    metrics["distance_km"]   = round(distance_km, 3) if distance_km is not None else None
    metrics["gps_available"] = gps_available

    # ── Efficiency metrics ────────────────────────────────────────────────────
    if distance_km and distance_km > 0.1:
        wh_per_km  = energy_wh  / distance_km
        mah_per_km = mah_consumed / distance_km
    else:
        wh_per_km  = None
        mah_per_km = None

    metrics["wh_per_km"]  = round(wh_per_km,  2) if wh_per_km  is not None else None
    metrics["mah_per_km"] = round(mah_per_km, 1) if mah_per_km is not None else None

    # ── VTOL phase split ──────────────────────────────────────────────────────
    if profile.type in ("vtol", "fixed_wing"):
        phase_metrics = _vtol_phase_split(result, flight_bat, volt_col, curr_col, arm_time, end_time)
        metrics.update(phase_metrics)

    # ── Expected efficiency baseline ─────────────────────────────────────────
    total_wh_capacity = (
        profile.battery.capacity_mah / 1000.0 *
        profile.battery.nominal_voltage
    )
    expected_range_km = max(profile.expected_range_km, 0.1)
    expected_wh_per_km = total_wh_capacity / expected_range_km
    metrics["expected_wh_per_km"] = round(expected_wh_per_km, 2)

    max_power_w = (
        profile.battery.max_continuous_current_a *
        profile.battery.nominal_voltage
    )
    metrics["max_rated_power_w"] = round(max_power_w, 0)

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = 100.0

    if not gps_available or distance_km is None or distance_km <= 0.1:
        score -= 5
        issues.append({
            "severity": "info",
            "code": "EFF-001",
            "message": "GPS distance unavailable — Wh/km efficiency cannot be computed.",
        })
    else:
        ratio = wh_per_km / expected_wh_per_km if expected_wh_per_km > 0 else 1.0
        if ratio > 1.5:
            score -= 20
            issues.append({
                "severity": "warning",
                "code": "EFF-002",
                "message": f"Flight significantly less efficient than baseline: "
                           f"{wh_per_km:.1f} Wh/km vs expected {expected_wh_per_km:.1f} Wh/km "
                           f"({ratio*100:.0f}% of baseline)",
                "value": round(wh_per_km, 2),
                "threshold": round(expected_wh_per_km * 1.5, 2),
            })
        elif ratio > 1.2:
            score -= 10
            issues.append({
                "severity": "info",
                "code": "EFF-002",
                "message": f"Flight slightly less efficient than baseline: "
                           f"{wh_per_km:.1f} Wh/km vs expected {expected_wh_per_km:.1f} Wh/km "
                           f"({ratio*100:.0f}% of baseline)",
                "value": round(wh_per_km, 2),
                "threshold": round(expected_wh_per_km * 1.2, 2),
            })

    if max_power_w > 0:
        power_ratio = avg_power_w / max_power_w
        if power_ratio > 1.0:
            score -= 10
            issues.append({
                "severity": "warning",
                "code": "EFF-003",
                "message": f"Average power {avg_power_w:.0f} W exceeds rated max {max_power_w:.0f} W. "
                           "Check current sensor calibration or battery sizing.",
                "value": round(avg_power_w, 0),
                "threshold": round(max_power_w, 0),
            })
        elif power_ratio > 0.9:
            score -= 5
            issues.append({
                "severity": "info",
                "code": "EFF-003",
                "message": f"Average power {avg_power_w:.0f} W near rated max {max_power_w:.0f} W "
                           f"({power_ratio*100:.0f}%). Consider higher capacity battery.",
                "value": round(avg_power_w, 0),
                "threshold": round(max_power_w * 0.9, 0),
            })

    score = max(0.0, min(100.0, score))

    from analyzer.scoring.engine import _grade
    grade = _grade(score)

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if energy_wh is not None and energy_wh >= 0:
        parts.append(f"{energy_wh:.1f} Wh used")
    if distance_km is not None and distance_km > 0.1:
        parts.append(f"{distance_km:.2f} km")
    if wh_per_km is not None and wh_per_km > 0:
        parts.append(f"{wh_per_km:.1f} Wh/km")
    parts.append(f"{avg_power_w:.0f} W avg")

    # VTOL summary addition
    hover_pct = metrics.get("hover_time_pct")
    if hover_pct is not None:
        parts.append(f"Hover {hover_pct:.0f}%")

    summary = " | ".join(parts) if parts else "No efficiency data"

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─── GPS distance helper ──────────────────────────────────────────────────────

def _gps_distance(
    result: ParseResult,
    arm_time: float,
    end_time: float,
) -> Tuple[Optional[float], bool]:
    """Return (distance_km, gps_available)."""
    gps_df = result.get("GPS")
    if gps_df is None or gps_df.empty:
        return None, False

    lat_col = _find_col(gps_df, ["Lat", "lat"])
    lon_col = _find_col(gps_df, ["Lng", "Lon", "lng", "lon"])
    if lat_col is None or lon_col is None:
        return None, False

    flight_gps = gps_df[
        (gps_df["timestamp"] >= arm_time) & (gps_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_gps) < 2:
        return None, False

    # Filter to 3D fix only if Status column present
    if "Status" in flight_gps.columns:
        flight_gps = flight_gps[flight_gps["Status"] >= 3]
    if len(flight_gps) < 2:
        return None, False

    lats = flight_gps[lat_col].values.astype(float)
    lons = flight_gps[lon_col].values.astype(float)

    # Check if coords are in degrees×1e7 (ArduPilot raw integer form)
    if np.max(np.abs(lats)) > 1000:
        lats = lats / 1e7
        lons = lons / 1e7

    distance_m = _haversine_total(lats, lons)
    return distance_m / 1000.0, True


def _haversine_total(lats: np.ndarray, lons: np.ndarray) -> float:
    """Sum of haversine distances between consecutive GPS points (metres)."""
    R = 6371000.0  # Earth radius metres
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lons[1:] - lons[:-1])

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c

    # Filter implausible jumps (> 500 m between samples — likely GPS artefact)
    distances = distances[distances < 500]
    return float(np.sum(distances))


# ─── VTOL phase split ─────────────────────────────────────────────────────────

_HOVER_MODES = {"QHOVER", "QLOITER", "QLAND", "QRTL", "QSTABILIZE", "QAUTOTUNE",
                "STABILIZE", "LOITER", "ALT_HOLD", "GUIDED", "AUTO"}
_CRUISE_MODES = {"AUTO", "CRUISE", "FBWA", "FBWB", "GUIDED", "RTL",
                 "FLY_BY_WIRE_A", "FLY_BY_WIRE_B", "MANUAL", "STABILIZE", "ACRO"}

# Modes that are unambiguously hover on a VTOL
_VTOL_HOVER = {"QHOVER", "QLOITER", "QLAND", "QRTL", "QSTABILIZE", "QAUTOTUNE", "QACRO"}
# Modes that are unambiguously cruise on a VTOL / fixed-wing
# Note: ArduPlane logs "FLY_BY_WIRE_A" / "FLY_BY_WIRE_B" (not the short form FBWA/FBWB)
_VTOL_CRUISE = {"AUTO", "CRUISE", "FBWA", "FBWB",
                "FLY_BY_WIRE_A", "FLY_BY_WIRE_B", "MANUAL", "STABILIZE", "ACRO",
                "GUIDED", "RTL", "LOITER", "AUTOTUNE", "CIRCLE", "TAKEOFF",
                "AVOID_ADSB", "THERMAL"}


def _vtol_phase_split(
    result: ParseResult,
    flight_bat: pd.DataFrame,
    volt_col: str,
    curr_col: str,
    arm_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """
    Split energy consumption into hover and cruise phases using mode data.
    Returns a dict of phase metrics to merge into the main metrics dict.
    """
    mode_df = result.get("MODE")
    if mode_df is None or mode_df.empty:
        return {"phase_split_available": False}

    # Build a mode timeline covering the flight window
    flight_modes = mode_df[
        (mode_df["timestamp"] >= arm_time) & (mode_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_modes) == 0:
        return {"phase_split_available": False}

    # For each BAT sample, determine the mode at that timestamp
    mode_times  = flight_modes["timestamp"].values
    mode_names  = flight_modes["mode_name"].values if "mode_name" in flight_modes.columns \
                  else np.array(["UNKNOWN"] * len(flight_modes))

    bat_ts    = flight_bat["timestamp"].values
    _volts    = np.clip(flight_bat[volt_col].values.astype(float), 0, None)
    _currents = flight_bat[curr_col].values.astype(float)   # raw, matches main energy calculation
    bat_power = _volts * _currents

    # Assign each BAT sample to a mode via searchsorted
    idx = np.searchsorted(mode_times, bat_ts, side="right") - 1
    idx = np.clip(idx, 0, len(mode_names) - 1)
    bat_modes = mode_names[idx]

    # Use N-1 intervals (same as main trapezoidal energy calculation).
    # Each interval is assigned the mode of its start sample.
    # This ensures hover_energy + cruise_energy == total energy_wh.
    dt_s      = np.diff(bat_ts)                                    # N-1 intervals (seconds)
    interval_modes  = bat_modes[:-1]                               # mode at start of each interval
    interval_energy = 0.5 * (bat_power[:-1] + bat_power[1:]) * dt_s / 3600  # trapezoidal Wh

    hover_mask  = np.isin(interval_modes, list(_VTOL_HOVER))
    cruise_mask = np.isin(interval_modes, list(_VTOL_CRUISE))

    hover_time_s  = float(np.sum(dt_s[hover_mask]))
    cruise_time_s = float(np.sum(dt_s[cruise_mask]))
    total_time_s  = hover_time_s + cruise_time_s if (hover_time_s + cruise_time_s) > 0 \
                    else float(bat_ts[-1] - bat_ts[0])

    hover_energy_wh  = float(np.sum(interval_energy[hover_mask]))
    cruise_energy_wh = float(np.sum(interval_energy[cruise_mask]))

    hover_pct  = hover_time_s  / total_time_s * 100 if total_time_s > 0 else 0.0
    cruise_pct = cruise_time_s / total_time_s * 100 if total_time_s > 0 else 0.0

    # Hover avg power
    hover_avg_w  = hover_energy_wh  / (hover_time_s  / 3600) if hover_time_s  > 0 else 0.0
    cruise_avg_w = cruise_energy_wh / (cruise_time_s / 3600) if cruise_time_s > 0 else 0.0

    return {
        "phase_split_available": True,
        "hover_time_s":          round(hover_time_s, 1),
        "cruise_time_s":         round(cruise_time_s, 1),
        "hover_time_pct":        round(hover_pct, 1),
        "cruise_time_pct":       round(cruise_pct, 1),
        "hover_energy_wh":       round(hover_energy_wh, 2),
        "cruise_energy_wh":      round(cruise_energy_wh, 2),
        "hover_avg_power_w":     round(hover_avg_w, 1),
        "cruise_avg_power_w":    round(cruise_avg_w, 1),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "score":     50.0,
        "grade":     "C",
        "available": False,
        "issues":    [],
        "metrics":   {"unavailable_reason": reason},
        "summary":   f"Efficiency analysis unavailable: {reason}",
    }
