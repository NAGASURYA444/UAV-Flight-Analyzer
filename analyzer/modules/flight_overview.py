"""
Flight Overview Module
======================
Extracts top-level flight facts from the log:

* Arm / disarm timestamps
* Flight phase sequence  (ARM → TAKEOFF → CRUISE/MISSION → LANDING → DISARM)
* Total armed time and actual airborne time
* Max altitude, max speed, total distance
* Endurance analysis  — actual vs expected, with root-cause inference
* Mode change timeline
* Safety event summary

Scoring contribution
--------------------
score = 100, then deductions:
  -5  to -15   endurance deficit > 15% (proportional to deficit)
  -10           no GPS data (cannot verify key metrics)
  -5            excessive mode changes (>8 in flight = instability / RC issues)
  -5            very short flight (< 2 min, not representative / premature abort)
  -10           arm → immediate failsafe / crash
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# EV event IDs for arm/disarm
EV_ARMED = 10
EV_DISARMED = 11

# ArduCopter mode number for RTL and LAND
MODE_RTL = 6
MODE_LAND = 9
MODE_AUTO = 3


@dataclass
class FlightPhase:
    name: str
    start_s: float
    end_s: float
    mode: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> Dict:
        return {
            "phase": self.name,
            "mode": self.mode,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
        }


def analyse(result: ParseResult, profile: DroneProfile) -> Dict[str, Any]:
    """
    Run flight overview analysis.

    Returns a standardised result dict:
    {
        score          : float   (0-100)
        grade          : str     (A/B/C/D/F)
        available      : bool
        issues         : list[dict]
        metrics        : dict
        summary        : str
    }
    """
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    # ── Arm / disarm times ────────────────────────────────────────────────────
    arm_time, disarm_time = _get_arm_disarm(result)
    if arm_time is None:
        # Fallback: use log time range
        arm_time, disarm_time = result.time_range

    armed_duration_s = max(0.0, (disarm_time or result.time_range[1]) - arm_time)
    armed_duration_min = armed_duration_s / 60.0

    metrics["arm_time_s"] = round(arm_time, 2)
    metrics["disarm_time_s"] = round(disarm_time, 2) if disarm_time else None
    metrics["armed_duration_s"] = round(armed_duration_s, 2)
    metrics["armed_duration_min"] = round(armed_duration_min, 2)

    # ── GPS-derived metrics ───────────────────────────────────────────────────
    gps_df = result.get("GPS")
    if gps_df is None:
        gps_df = result.get("GPS2")
    has_gps = gps_df is not None and len(gps_df) > 5

    if has_gps:
        gps_flight = gps_df[
            (gps_df["timestamp"] >= arm_time) &
            (gps_df["timestamp"] <= (disarm_time or gps_df["timestamp"].max()))
        ].copy()

        max_alt_m = _max_altitude(gps_flight)
        max_speed_ms = _max_speed(gps_flight)
        distance_km = _total_distance(gps_flight)
        home_lat, home_lon = _home_position(gps_flight)

        metrics["max_altitude_msl_m"] = round(max_alt_m, 1)
        metrics["max_altitude_m"] = round(max_alt_m, 1)   # kept for backward compat
        metrics["max_speed_ms"] = round(max_speed_ms, 2)
        metrics["distance_km"] = round(distance_km, 3)
        metrics["home_lat"] = home_lat
        metrics["home_lon"] = home_lon
    else:
        metrics["max_altitude_msl_m"] = None
        metrics["max_altitude_m"] = None
        metrics["max_speed_ms"] = None
        metrics["distance_km"] = None
        issues.append(_issue(
            "warning", "OVR-001",
            "No GPS data found — altitude, speed and distance metrics unavailable.",
        ))

    # ── AGL altitude from barometer ───────────────────────────────────────────
    # BARO Alt is raw MSL but zeroed to ground at startup → relative reading = AGL
    agl_result = _agl_altitude(result, arm_time, disarm_time)
    metrics.update(agl_result)

    # ── Airborne time (altitude-based) ────────────────────────────────────────
    airborne_s = _estimate_airborne_time(result, arm_time, disarm_time)
    airborne_min = airborne_s / 60.0
    metrics["airborne_s"] = round(airborne_s, 2)
    metrics["airborne_min"] = round(airborne_min, 2)

    # ── Flight phases ─────────────────────────────────────────────────────────
    phases = _detect_phases(result, arm_time, disarm_time)
    metrics["phases"] = [p.to_dict() for p in phases]
    metrics["phase_sequence"] = " > ".join(p.name for p in phases)

    # ── Mode changes ─────────────────────────────────────────────────────────
    mode_changes = _mode_changes(result, arm_time, disarm_time)
    metrics["mode_changes"] = mode_changes
    metrics["mode_change_count"] = len(mode_changes)

    if len(mode_changes) > 8:
        issues.append(_issue(
            "warning", "OVR-002",
            f"Excessive mode changes during flight ({len(mode_changes)}). "
            "May indicate RC interference, instability, or pilot intervention.",
            value=len(mode_changes), threshold=8,
        ))

    # ── Endurance analysis ────────────────────────────────────────────────────
    endurance_result = _endurance_analysis(
        actual_min=airborne_min,
        expected_min=profile.expected_endurance_min,
        result=result,
        profile=profile,
        arm_time=arm_time,
        disarm_time=disarm_time,
        metrics=metrics,
    )
    metrics["endurance"] = endurance_result
    issues.extend(endurance_result.get("issues", []))

    # ── Safety events ─────────────────────────────────────────────────────────
    safety_events = _extract_safety_events(result)
    metrics["safety_events"] = safety_events
    for ev in safety_events:
        issues.append(_issue(
            ev["severity"], ev["code"], ev["message"],
            timestamp=ev.get("timestamp"),
        ))

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues, profile, airborne_min)
    grade = _grade(score)

    alt_str = ""
    if metrics.get("max_altitude_msl_m") is not None:
        alt_str = f", max alt {metrics['max_altitude_msl_m']:.0f} m MSL"
        if metrics.get("max_altitude_agl_m") is not None:
            alt_str += f" / {metrics['max_altitude_agl_m']:.0f} m AGL"
    elif metrics.get("max_altitude_agl_m") is not None:
        alt_str = f", max alt {metrics['max_altitude_agl_m']:.0f} m AGL"

    summary = (
        f"Flight: {_fmt_duration(airborne_s)} airborne"
        + (f", {metrics['distance_km']:.2f} km" if metrics.get("distance_km") else "")
        + alt_str
        + f"  |  Expected endurance: {profile.expected_endurance_min:.0f} min"
        + f", actual: {airborne_min:.1f} min"
    )

    return {
        "score": round(score, 1),
        "grade": grade,
        "available": True,
        "issues": issues,
        "metrics": metrics,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endurance analysis
# ─────────────────────────────────────────────────────────────────────────────

def _endurance_analysis(
    actual_min: float,
    expected_min: float,
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float,
    disarm_time: Optional[float],
    metrics: Dict,
) -> Dict[str, Any]:
    """
    Compare actual vs expected endurance and infer probable root causes.
    """
    delta_min = actual_min - expected_min
    delta_pct = (delta_min / expected_min * 100.0) if expected_min > 0 else 0.0

    root_causes: List[str] = []
    end_issues: List[Dict] = []

    if delta_pct < -10:
        # ── Root cause inference ──────────────────────────────────────────────

        # 1. Short mission plan?  (drone spent < 40% of time in AUTO mode)
        mode_df = result.get("MODE")
        auto_fraction = 0.0
        if mode_df is not None and len(mode_df) > 0:
            # reset_index so positional .iloc access is safe
            flight_modes = mode_df[mode_df["timestamp"] >= arm_time].reset_index(drop=True)
            if len(flight_modes) > 1:
                total = flight_modes["timestamp"].iloc[-1] - flight_modes["timestamp"].iloc[0]
                auto_time = 0.0
                for idx in range(len(flight_modes)):
                    row = flight_modes.iloc[idx]
                    # Use mode_name (works for both ArduCopter AUTO=3 and ArduPlane AUTO=10)
                    mode_name_val = str(row.get("mode_name", row.get("Mode", "")))
                    t_start = row["timestamp"]
                    if idx + 1 < len(flight_modes):
                        t_end = flight_modes.iloc[idx + 1]["timestamp"]
                    else:
                        t_end = disarm_time or t_start
                    if mode_name_val == "AUTO":
                        auto_time += t_end - t_start
                auto_fraction = auto_time / total if total > 0 else 0.0

        if auto_fraction < 0.3 and actual_min < expected_min * 0.7:
            root_causes.append("Short or incomplete mission plan (low AUTO mode time)")

        # 2. High average power draw?
        bat_df = result.get("BAT")
        if bat_df is None:
            bat_df = result.get("BAT2")
        avg_current = None
        if bat_df is not None and "Curr" in bat_df.columns:
            flight_bat = bat_df[bat_df["timestamp"] >= arm_time]
            if len(flight_bat) > 0:
                avg_current = flight_bat["Curr"].mean()
                expected_avg_current = (profile.battery.capacity_mah / 1000.0) / (expected_min / 60.0)
                if avg_current > expected_avg_current * 1.25:
                    root_causes.append(
                        f"High average current draw ({avg_current:.1f} A vs "
                        f"expected ~{expected_avg_current:.1f} A) — possible heavy payload or high wind"
                    )

        # 3. Early battery failsafe?
        err_df = result.get("ERR")
        if err_df is not None and "Subsys" in err_df.columns:
            bat_fs = err_df[err_df["Subsys"] == 6]  # Subsys 6 = battery failsafe
            if len(bat_fs) > 0:
                root_causes.append("Battery failsafe triggered — flight cut short")

        RTL_MODE_NAMES = {"RTL", "QRTL", "SMART_RTL"}

        # 4. Early RTL (before 50% of expected time)?
        if mode_df is not None and len(mode_df) > 0:
            fm = mode_df[mode_df["timestamp"] >= arm_time].reset_index(drop=True)
            early_cutoff = arm_time + (expected_min * 60 * 0.5)
            rtl_early = fm[
                (fm["mode_name"].isin(RTL_MODE_NAMES)) &
                (fm["timestamp"] < early_cutoff)
            ]
            if len(rtl_early) > 0:
                root_causes.append("Early RTL activated (before 50% of expected flight time)")

        # 5. Mission completed normally? (RTL/QRTL after >50% of expected flight time
        #    with most of the flight in AUTO = normal mission completion, not a deficit)
        if not root_causes and mode_df is not None and len(mode_df) > 0:
            fm = mode_df[mode_df["timestamp"] >= arm_time].reset_index(drop=True)
            late_cutoff = arm_time + (expected_min * 60 * 0.5)
            late_rtl = fm[
                (fm["mode_name"].isin(RTL_MODE_NAMES)) &
                (fm["timestamp"] >= late_cutoff)
            ]
            if len(late_rtl) > 0 and auto_fraction > 0.5:
                # Distinguish genuine mission completion from operator/GCS recall.
                # If battery still has > 40% remaining, the drone wasn't energy-limited
                # → RTL was likely a deliberate recall, not energy exhaustion or short plan.
                remaining_pct = 0.0  # default: assume battery used (conservative)
                if avg_current is not None and avg_current > 0 and profile.battery.capacity_mah > 0:
                    consumed_mah = avg_current * (actual_min / 60.0) * 1000.0
                    remaining_pct = max(
                        0.0,
                        (profile.battery.capacity_mah - consumed_mah)
                        / profile.battery.capacity_mah * 100.0,
                    )
                if remaining_pct >= 40.0:
                    root_causes.append(
                        f"RTL triggered with ~{remaining_pct:.0f}% battery remaining — "
                        "likely operator recall or GCS command. No energy issue detected."
                    )
                    # Mark as intentional so severity and score are not penalised
                    end_issues.append(_issue(
                        "info", "OVR-003",
                        f"Flight shorter than expected ({actual_min:.1f} min vs "
                        f"{expected_min:.0f} min expected, {delta_pct:+.1f}%) — "
                        f"intentional early return with ~{remaining_pct:.0f}% battery remaining. "
                        "No health concern.",
                        value=actual_min, threshold=expected_min,
                    ))
                    return {
                        "actual_min": round(actual_min, 2),
                        "expected_min": expected_min,
                        "delta_min": round(delta_min, 2),
                        "delta_pct": round(delta_pct, 1),
                        "root_causes": root_causes,
                        "intentional_short": True,
                        "issues": end_issues,
                    }
                else:
                    root_causes.append(
                        "Mission plan completed before expected endurance — "
                        "mission distance was shorter than maximum range."
                    )

        # 6. Underpowered / heavy payload (using RCOU data)
        rcou_df = result.get("RCOU")
        avg_throttle_pct = None
        if rcou_df is not None:
            motor_channels = [f"C{ch}" for ch in profile.motors.channels]
            avail_cols = [c for c in motor_channels if c in rcou_df.columns]
            if avail_cols:
                flight_rcou = rcou_df[rcou_df["timestamp"] >= arm_time]
                if len(flight_rcou) > 10:
                    min_pwm = profile.motors.min_pwm
                    pwm_range = profile.motors.max_pwm - min_pwm
                    # Use first motor for fixed_wing, mean of all for multirotor
                    if profile.type == "fixed_wing":
                        pwm_vals = flight_rcou[avail_cols[0]].values.astype(float)
                        thr = np.clip((pwm_vals - min_pwm) / pwm_range * 100.0, 0.0, 100.0)
                        airborne = thr[thr > 10.0]
                        avg_throttle_pct = float(np.mean(airborne)) if len(airborne) > 0 else None
                    else:
                        pwm_matrix = flight_rcou[avail_cols].values.astype(float)
                        thr_matrix = np.clip((pwm_matrix - min_pwm) / pwm_range * 100.0, 0.0, 100.0)
                        in_flight = np.max(thr_matrix, axis=1) > 20.0
                        if in_flight.sum() > 10:
                            avg_throttle_pct = float(np.mean(np.mean(thr_matrix[in_flight], axis=1)))

        if avg_throttle_pct is not None:
            expected_hover = profile.motors.hover_throttle_pct
            if avg_throttle_pct > expected_hover + 20:
                root_causes.append(
                    f"High average throttle ({avg_throttle_pct:.0f}% vs expected ~{expected_hover:.0f}%) "
                    "— possible heavy payload, strong headwind, or propulsion efficiency loss"
                )

        # 7. High altitude penalty (using already-computed metrics)
        mean_agl = metrics.get("mean_altitude_agl_m")
        if mean_agl is not None and mean_agl > profile.max_altitude_m * 0.8:
            root_causes.append(
                f"Mean flight altitude ({mean_agl:.0f} m AGL avg) exceeds profile maximum "
                f"({profile.max_altitude_m:.0f} m) — thinner air at higher altitude increases power consumption"
            )

        # 8. Headwind detection (GPS speed + throttle correlation)
        gps_df_rc = result.get("GPS")
        if gps_df_rc is not None and avg_throttle_pct is not None and avg_throttle_pct > 70:
            spd_col = "Spd" if "Spd" in gps_df_rc.columns else ("GndSpd" if "GndSpd" in gps_df_rc.columns else None)
            if spd_col:
                flight_gps_rc = gps_df_rc[gps_df_rc["timestamp"] >= arm_time]
                if len(flight_gps_rc) > 10:
                    avg_spd = float(flight_gps_rc[spd_col].mean())
                    if avg_spd < profile.max_speed_ms * 0.5:
                        root_causes.append(
                            f"Possible strong headwind — low average ground speed ({avg_spd:.1f} m/s, "
                            f"expected >{profile.max_speed_ms * 0.5:.1f} m/s) combined with high throttle"
                        )

        if not root_causes:
            root_causes.append("Unknown — review battery curve and mission plan")

        severity = "critical" if delta_pct < -30 else "warning"
        end_issues.append(_issue(
            severity, "OVR-003",
            f"Endurance deficit: expected {expected_min:.0f} min, "
            f"actual {actual_min:.1f} min ({delta_pct:+.1f}%). "
            f"Probable cause(s): {'; '.join(root_causes)}",
            value=actual_min, threshold=expected_min,
        ))

    elif delta_pct > 0:
        end_issues.append(_issue(
            "info", "OVR-004",
            f"Flight exceeded expected endurance: {actual_min:.1f} min vs "
            f"{expected_min:.0f} min expected ({delta_pct:+.1f}%).",
        ))

    return {
        "actual_min": round(actual_min, 2),
        "expected_min": expected_min,
        "delta_min": round(delta_min, 2),
        "delta_pct": round(delta_pct, 1),
        "root_causes": root_causes,
        "intentional_short": False,
        "issues": end_issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_phases(
    result: ParseResult,
    arm_time: float,
    disarm_time: Optional[float],
) -> List[FlightPhase]:
    """Detect flight phases from mode changes and altitude profile."""
    phases: List[FlightPhase] = []
    end_time = disarm_time or result.time_range[1]

    mode_df = result.get("MODE")
    if mode_df is None or len(mode_df) == 0:
        phases.append(FlightPhase("ARM", arm_time, arm_time))
        phases.append(FlightPhase("FLIGHT", arm_time, end_time))
        phases.append(FlightPhase("DISARM", end_time, end_time))
        return phases

    # Filter modes during armed window
    flight_modes = mode_df[
        (mode_df["timestamp"] >= arm_time) &
        (mode_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    phases.append(FlightPhase("ARM", arm_time, arm_time))

    # Build phase list from mode transitions
    for i, row in flight_modes.iterrows():
        t_start = row["timestamp"]
        t_end = (
            flight_modes.loc[i + 1, "timestamp"]
            if i + 1 < len(flight_modes)
            else end_time
        )
        mode_num = row.get("Mode", -1)
        mode_name = row.get("mode_name", str(mode_num))

        # Map mode to phase label
        # Use mode_name string (works for both ArduCopter and ArduPlane AUTO)
        # ArduCopter AUTO = mode 3 | ArduPlane AUTO = mode 10 — both have mode_name "AUTO"
        if mode_num == MODE_LAND or mode_name in ("LAND", "QLAND"):
            phase_label = "LANDING"
        elif mode_num == MODE_RTL or mode_name in ("RTL", "QRTL", "SMART_RTL"):
            phase_label = "RTH"
        elif mode_name == "AUTO":
            phase_label = "MISSION"
        else:
            phase_label = "FLIGHT"

        # Skip duplicate consecutive phases
        if phases and phases[-1].name == phase_label:
            phases[-1].end_s = t_end
        else:
            phases.append(FlightPhase(phase_label, t_start, t_end, mode=mode_name))

    phases.append(FlightPhase("DISARM", end_time, end_time))
    return phases


# ─────────────────────────────────────────────────────────────────────────────
# AGL altitude from barometer
# ─────────────────────────────────────────────────────────────────────────────

def _agl_altitude(
    result: ParseResult,
    arm_time: float,
    disarm_time: Optional[float],
) -> Dict[str, Any]:
    """
    Compute AGL (Above Ground Level) altitude metrics from the barometer.

    ArduPilot's BARO.Alt is absolute MSL altitude, but since the autopilot
    homes its altitude reference at startup (ground level), subtracting the
    ground reading gives a true AGL value for the flight.
    """
    baro_df = result.get("BARO")
    if baro_df is None:
        baro_df = result.get("BARO2")
    if baro_df is None or "Alt" not in baro_df.columns or "timestamp" not in baro_df.columns:
        return {"max_altitude_agl_m": None, "mean_altitude_agl_m": None}

    end_time = disarm_time or result.time_range[1]
    flight_baro = baro_df[
        (baro_df["timestamp"] >= arm_time) &
        (baro_df["timestamp"] <= end_time)
    ].copy()

    if len(flight_baro) < 4:
        return {"max_altitude_agl_m": None, "mean_altitude_agl_m": None}

    # Ground level = median of first 5 s after arm (drone still on the ground)
    ground_window = flight_baro[flight_baro["timestamp"] < arm_time + 5.0]
    ground_alt = (
        float(ground_window["Alt"].median())
        if len(ground_window) > 0
        else float(flight_baro["Alt"].iloc[0])
    )

    agl = flight_baro["Alt"] - ground_alt
    airborne_agl = agl[agl > 1.5]   # only count as airborne when >1.5 m AGL

    max_agl = float(agl.max()) if len(agl) > 0 else 0.0
    mean_agl = float(airborne_agl.mean()) if len(airborne_agl) > 0 else 0.0

    return {
        "max_altitude_agl_m": round(max_agl, 1),
        "mean_altitude_agl_m": round(mean_agl, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _max_altitude(gps_df: pd.DataFrame) -> float:
    col = "Alt" if "Alt" in gps_df.columns else ("Altitude" if "Altitude" in gps_df.columns else None)
    if col is None or len(gps_df) == 0:
        return 0.0
    return float(gps_df[col].max())


def _max_speed(gps_df: pd.DataFrame) -> float:
    col = "Spd" if "Spd" in gps_df.columns else ("GndSpd" if "GndSpd" in gps_df.columns else None)
    if col is None or len(gps_df) == 0:
        return 0.0
    return float(gps_df[col].max())


def _total_distance(gps_df: pd.DataFrame) -> float:
    """Sum of haversine distances between consecutive GPS positions (km)."""
    if len(gps_df) < 2:
        return 0.0
    lat_col = "Lat" if "Lat" in gps_df.columns else None
    lng_col = "Lng" if "Lng" in gps_df.columns else ("Lon" if "Lon" in gps_df.columns else None)
    if lat_col is None or lng_col is None:
        return 0.0

    lats = gps_df[lat_col].values
    lons = gps_df[lng_col].values

    # Haversine vectorised
    R = 6_371_000  # metres
    lat1, lat2 = np.radians(lats[:-1]), np.radians(lats[1:])
    lon1, lon2 = np.radians(lons[:-1]), np.radians(lons[1:])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_m = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    # Filter out GPS jumps > 500 m between consecutive samples
    dist_m = dist_m[dist_m < 500]
    return float(dist_m.sum()) / 1000.0


def _home_position(gps_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    lat_col = "Lat" if "Lat" in gps_df.columns else None
    lng_col = "Lng" if "Lng" in gps_df.columns else ("Lon" if "Lon" in gps_df.columns else None)
    if lat_col is None or lng_col is None or len(gps_df) == 0:
        return None, None
    # Use the first GPS record after arm as home
    row = gps_df.iloc[0]
    return float(row[lat_col]), float(row[lng_col])


# ─────────────────────────────────────────────────────────────────────────────
# Arm / disarm detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_arm_disarm(result: ParseResult) -> Tuple[Optional[float], Optional[float]]:
    """Find arm and disarm timestamps from EV messages, falling back to log bounds."""
    ev_df = result.get("EV")
    arm_time: Optional[float] = None
    disarm_time: Optional[float] = None

    if ev_df is not None and "Id" in ev_df.columns and "timestamp" in ev_df.columns:
        arm_rows = ev_df[ev_df["Id"] == EV_ARMED]
        disarm_rows = ev_df[ev_df["Id"] == EV_DISARMED]
        if len(arm_rows) > 0:
            arm_time = float(arm_rows["timestamp"].iloc[0])
        if len(disarm_rows) > 0:
            disarm_time = float(disarm_rows["timestamp"].iloc[-1])

    # Fallback to log bounds if events not found
    if arm_time is None:
        arm_time = result.time_range[0]
    if disarm_time is None:
        disarm_time = result.time_range[1]

    return arm_time, disarm_time


# ─────────────────────────────────────────────────────────────────────────────
# Airborne time estimation
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_airborne_time(
    result: ParseResult,
    arm_time: float,
    disarm_time: Optional[float],
) -> float:
    """
    Estimate actual airborne time using barometer altitude.
    Falls back to full armed duration if barometer data unavailable.
    """
    end_time = disarm_time or result.time_range[1]

    baro_df = result.get("BARO")
    if baro_df is None:
        baro_df = result.get("BARO2")
    if baro_df is None or "Alt" not in baro_df.columns or "timestamp" not in baro_df.columns:
        return max(0.0, end_time - arm_time)

    flight_baro = baro_df[
        (baro_df["timestamp"] >= arm_time) &
        (baro_df["timestamp"] <= end_time)
    ].copy()

    if len(flight_baro) < 4:
        return max(0.0, end_time - arm_time)

    # Ground altitude = median of first 5 seconds
    ground_window = flight_baro[
        flight_baro["timestamp"] < arm_time + 5.0
    ]
    ground_alt = (
        float(ground_window["Alt"].median())
        if len(ground_window) > 0
        else float(flight_baro["Alt"].iloc[0])
    )
    AIRBORNE_THRESHOLD_M = 1.5  # consider airborne if > 1.5 m above ground

    airborne_mask = (flight_baro["Alt"] - ground_alt) > AIRBORNE_THRESHOLD_M
    if airborne_mask.sum() == 0:
        return 0.0

    # Sum time intervals where airborne
    ts = flight_baro["timestamp"].values
    intervals = np.diff(ts)
    airborne_flags = airborne_mask.values[:-1]
    return float(np.sum(intervals[airborne_flags]))


# ─────────────────────────────────────────────────────────────────────────────
# Mode change timeline
# ─────────────────────────────────────────────────────────────────────────────

def _mode_changes(
    result: ParseResult,
    arm_time: float,
    disarm_time: Optional[float],
) -> List[Dict]:
    mode_df = result.get("MODE")
    if mode_df is None or len(mode_df) == 0:
        return []
    end_time = disarm_time or result.time_range[1]
    flight_modes = mode_df[
        (mode_df["timestamp"] >= arm_time) &
        (mode_df["timestamp"] <= end_time)
    ]
    changes = []
    last_mode = None
    last_ts = None
    for _, row in flight_modes.iterrows():
        ts = round(float(row["timestamp"]), 2)
        mode = row.get("mode_name", str(row.get("Mode", "?")))
        # Skip duplicate entries (same mode at same or nearly-same timestamp)
        if mode == last_mode and last_ts is not None and abs(ts - last_ts) < 1.0:
            continue
        changes.append({
            "timestamp_s": ts,
            "mode": mode,
            "reason": row.get("Rsn", None),
        })
        last_mode = mode
        last_ts = ts
    return changes


# ─────────────────────────────────────────────────────────────────────────────
# Safety events
# ─────────────────────────────────────────────────────────────────────────────

def _extract_safety_events(result: ParseResult) -> List[Dict]:
    events: List[Dict] = []

    err_df = result.get("ERR")
    if err_df is not None and "Subsys" in err_df.columns and "ECode" in err_df.columns:
        # Only actual error events (ECode != 0)
        errors = err_df[err_df["ECode"] != 0]

        # Aggregate by (Subsys, ECode) — emit ONE summary issue per unique error type.
        # Emitting one issue per row causes massive score collapse when a sensor
        # fires hundreds of repeated events (e.g. 718 radio-loss ERR events).
        grouped = errors.groupby(["Subsys", "ECode"])
        for (subsys_id, ecode), grp in grouped:
            subsys_name = grp["subsys_name"].iloc[0] if "subsys_name" in grp.columns else str(subsys_id)
            count = len(grp)
            first_ts = float(grp["timestamp"].iloc[0])
            sev = "critical" if subsys_id in (5, 6, 15, 16) else "warning"
            count_str = f" (x{count})" if count > 1 else ""
            events.append({
                "severity": sev,
                "code": f"ERR-{subsys_id:02d}",
                "message": f"Error — Subsystem: {subsys_name}, Code: {int(ecode)}{count_str}",
                "timestamp": round(first_ts, 2),
            })

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(
    metrics: Dict,
    issues: List[Dict],
    profile: DroneProfile,
    airborne_min: float,
) -> float:
    score = 100.0

    # Endurance deficit — skip deduction for intentional early returns
    endo = metrics.get("endurance", {})
    delta_pct = endo.get("delta_pct", 0.0)
    if delta_pct < -10 and not endo.get("intentional_short", False):
        deduction = min(20.0, abs(delta_pct) * 0.4)
        score -= deduction

    # No GPS
    if metrics.get("distance_km") is None:
        score -= 10.0

    # Excessive mode changes
    mode_count = metrics.get("mode_change_count", 0)
    if mode_count > 8:
        score -= min(10.0, (mode_count - 8) * 1.5)

    # Very short flight
    if 0 < airborne_min < 2.0:
        score -= 5.0

    # Safety events
    for issue in issues:
        if issue.get("severity") == "critical":
            score -= 15.0
        elif issue.get("severity") == "warning":
            score -= 5.0

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Cross-module endurance enrichment
# ─────────────────────────────────────────────────────────────────────────────

def post_analyse(
    overview_result: Dict[str, Any],
    module_results: Dict[str, Dict],
    profile: DroneProfile,
) -> Dict[str, Any]:
    """
    Phase 4: Enrich endurance root-cause analysis with cross-module data.
    Call this AFTER all modules have run, passing the full module_results dict.
    Updates the overview result in-place and returns it.
    """
    endurance = overview_result.get("metrics", {}).get("endurance", {})
    if not endurance or endurance.get("delta_pct", 0) >= -10:
        return overview_result  # no deficit — nothing to enrich

    root_causes = endurance.get("root_causes", [])
    new_causes: List[str] = []

    # ── Check battery remaining — if > 40% is left, the flight was NOT energy-limited.
    # Energy-waste causes (PID, battery IR, efficiency) are irrelevant in that case and
    # should not be added as endurance root causes (they are real issues but are NOT why
    # the flight ended short).
    bat = module_results.get("battery", {})
    bat_remaining_pct = 0.0
    if bat.get("available"):
        bat_remaining_pct = bat.get("metrics", {}).get("capacity_remaining_pct", 0.0)
    energy_limited = (bat_remaining_pct < 40.0)

    # ── Efficiency: Wh/km vs baseline ────────────────────────────────────────
    eff = module_results.get("efficiency", {})
    if energy_limited and eff.get("available"):
        eff_m = eff.get("metrics", {})
        wh_per_km = eff_m.get("wh_per_km")
        expected_wh = eff_m.get("expected_wh_per_km")
        if wh_per_km and expected_wh and expected_wh > 0 and wh_per_km > expected_wh * 1.3:
            pct_worse = (wh_per_km / expected_wh - 1.0) * 100.0
            new_causes.append(
                f"Poor energy efficiency: {wh_per_km:.1f} Wh/km vs expected {expected_wh:.1f} Wh/km "
                f"({pct_worse:+.0f}%) — investigate drag, payload weight, or wind"
            )

    # ── Control: PID oscillation wastes energy ───────────────────────────────
    ctl = module_results.get("control", {})
    if energy_limited and ctl.get("available"):
        ctl_m = ctl.get("metrics", {})
        roll_osc = ctl_m.get("roll_osc_rms_deg", 0.0)
        pitch_osc = ctl_m.get("pitch_osc_rms_deg", 0.0)
        max_osc = max(roll_osc, pitch_osc)
        osc_threshold = 6.0 if profile.type in ("fixed_wing", "vtol") else 3.0
        if max_osc > osc_threshold:
            new_causes.append(
                f"PID oscillation ({max_osc:.1f}° RMS) causes constant attitude corrections, "
                "wasting energy — retune PID gains or run AutoTune"
            )

    # ── Battery: degraded pack reduces usable capacity ───────────────────────
    if bat.get("available"):
        bat_m = bat.get("metrics", {})
        r_per_cell = bat_m.get("internal_resistance_per_cell_mohm", 0.0)
        if energy_limited and r_per_cell and r_per_cell > 15.0:
            new_causes.append(
                f"Elevated battery internal resistance ({r_per_cell:.0f} mOhm/cell) "
                "reduces usable capacity and increases voltage sag under load"
            )
        charge_pct = bat_m.get("start_charge_pct", 100.0)
        if charge_pct < 90.0:
            new_causes.append(
                f"Battery started at only {charge_pct:.0f}% charge — "
                f"approximately {100.0 - charge_pct:.0f}% less energy available than expected"
            )
        # If flight was NOT energy-limited but ended very short, note the battery state.
        # Guard: skip if _endurance_analysis already mentioned battery remaining (e.g.
        # "RTL triggered with ~59% battery remaining") to avoid a duplicate root cause.
        if not energy_limited and bat_remaining_pct >= 40.0:
            already_noted = any("battery remaining" in c for c in root_causes)
            if not already_noted:
                new_causes.append(
                    f"Aircraft landed with ~{bat_remaining_pct:.0f}% battery remaining — "
                    "flight was cut short intentionally, not due to energy constraints"
                )

    # ── VTOL: excessive hover fraction ───────────────────────────────────────
    if profile.type == "vtol" and eff.get("available"):
        eff_m = eff.get("metrics", {})
        hover_pct = eff_m.get("hover_time_pct", 0.0)
        if hover_pct and hover_pct > 25.0:
            hover_w = eff_m.get("hover_avg_power_w", 0.0)
            cruise_w = eff_m.get("cruise_avg_power_w", 0.0)
            if hover_w and cruise_w and hover_w > cruise_w * 1.5:
                new_causes.append(
                    f"Excessive hover time ({hover_pct:.0f}% of flight) at "
                    f"{hover_w:.0f} W avg vs {cruise_w:.0f} W cruise — "
                    "hover consumes significantly more power than cruise"
                )

    # ── Merge (no duplicates) ─────────────────────────────────────────────────
    for cause in new_causes:
        if cause not in root_causes:
            root_causes.append(cause)

    endurance["root_causes"] = root_causes

    # ── Re-evaluate intentional_short after enrichment ────────────────────────
    # post_analyse may have added "aircraft landed with ~N% battery remaining —
    # flight was cut short intentionally" as a new cause.  If so, the flight is
    # now definitively classified as intentional — downgrade OVR-003 to info and
    # mark intentional_short=True so _compute_score skips the deduction.
    intentional = endurance.get("intentional_short", False) or any(
        "cut short intentionally" in c for c in root_causes
    )
    endurance["intentional_short"] = intentional

    # Update the OVR-003 issue: message, severity, and score-deduction flag
    for issue in overview_result.get("issues", []):
        if issue.get("code") == "OVR-003":
            if intentional:
                issue["severity"] = "info"
                issue["message"] = (
                    f"Flight shorter than expected ({endurance['actual_min']:.1f} min vs "
                    f"{endurance['expected_min']:.0f} min expected, {endurance['delta_pct']:+.1f}%) — "
                    f"intentional early return. {'; '.join(root_causes)}"
                )
            else:
                issue["message"] = (
                    f"Endurance deficit: expected {endurance['expected_min']:.0f} min, "
                    f"actual {endurance['actual_min']:.1f} min ({endurance['delta_pct']:+.1f}%). "
                    f"Probable cause(s): {'; '.join(root_causes)}"
                )
            break

    # Re-compute score now that intentional_short may have changed
    overview_result["score"] = round(_compute_score(
        overview_result.get("metrics", {}),
        overview_result.get("issues", []),
        profile,
        overview_result.get("metrics", {}).get("airborne_s", 0.0) / 60.0,
    ), 1)
    overview_result["grade"] = _grade(overview_result["score"])

    return overview_result


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


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


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
