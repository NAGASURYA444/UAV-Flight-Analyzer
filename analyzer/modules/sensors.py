"""
Sensor Health Module — Phase 2
================================
Checks the health of all onboard sensors using telemetry data from the log.

Sensors checked
---------------
1. GPS        — fix quality, satellite count, HDOP, position consistency
2. EKF / XKF  — innovation levels, test ratios, filter flags
               (NKF4 = ArduCopter, XKF4 = ArduPlane/VTOL)
3. IMU        — temperature stability, multi-IMU agreement (if IMU2 present)
4. Magnetometer — field strength consistency, interference spikes
5. Barometer  — altitude consistency, pressure stability

Each sub-sensor produces a 0-100 sub-score.
Overall sensor score = weighted average of available sub-scores.

Scoring weights
---------------
GPS : 0.35   (navigation-critical)
EKF : 0.30   (filter health directly affects flight safety)
IMU : 0.20
MAG : 0.10
BAR : 0.05
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

SUB_WEIGHTS = {"gps": 0.35, "ekf": 0.30, "imu": 0.20, "mag": 0.10, "baro": 0.05}

# ArduPilot GPS fix types
GPS_FIX_NAMES = {0: "No Fix", 1: "Dead Reckoning", 2: "2D Fix",
                 3: "3D Fix", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}


def analyse(result: ParseResult, profile: DroneProfile, arm_time: float = 0.0) -> Dict[str, Any]:
    """Run sensor health analysis. Returns standardised result dict."""
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}
    sub_scores: Dict[str, float] = {}

    # ── GPS ───────────────────────────────────────────────────────────────────
    gps_df = result.get("GPS")
    if gps_df is None:
        gps_df = result.get("GPS2")
    if gps_df is not None and len(gps_df) > 5:
        gps_r = _gps_analysis(gps_df, profile, arm_time)
        metrics["gps"] = gps_r["metrics"]
        issues.extend(gps_r["issues"])
        sub_scores["gps"] = gps_r["score"]
    else:
        issues.append(_issue("warning", "SEN-001", "No GPS data found in log."))

    # ── EKF / XKF ─────────────────────────────────────────────────────────────
    ekf_df = result.get("XKF4")
    ekf_type = "XKF4"
    if ekf_df is None:
        ekf_df = result.get("NKF4")
        ekf_type = "NKF4"
    if ekf_df is None:
        ekf_df = result.get("EKF4")
        ekf_type = "EKF4"
    if ekf_df is None:
        ekf_type = None
    if ekf_df is not None and len(ekf_df) > 5:
        ekf_r = _ekf_analysis(ekf_df, ekf_type, arm_time)
        metrics["ekf"] = ekf_r["metrics"]
        issues.extend(ekf_r["issues"])
        sub_scores["ekf"] = ekf_r["score"]
    else:
        issues.append(_issue("info", "SEN-002",
                             "No EKF health data (NKF4/XKF4) found in log."))

    # ── IMU ───────────────────────────────────────────────────────────────────
    imu_df = result.get("IMU")
    imu2_df = result.get("IMU2")
    if imu_df is not None and len(imu_df) > 10:
        imu_r = _imu_analysis(imu_df, imu2_df, arm_time)
        metrics["imu"] = imu_r["metrics"]
        issues.extend(imu_r["issues"])
        sub_scores["imu"] = imu_r["score"]
    else:
        issues.append(_issue("info", "SEN-003", "No IMU data found in log."))

    # ── Magnetometer ──────────────────────────────────────────────────────────
    mag_df = result.get("MAG")
    if mag_df is None:
        mag_df = result.get("MAG2")
    if mag_df is not None and len(mag_df) > 10:
        mag_r = _mag_analysis(mag_df, arm_time)
        metrics["mag"] = mag_r["metrics"]
        issues.extend(mag_r["issues"])
        sub_scores["mag"] = mag_r["score"]

    # ── Barometer ─────────────────────────────────────────────────────────────
    baro_df = result.get("BARO")
    if baro_df is None:
        baro_df = result.get("BARO2")
    if baro_df is not None and len(baro_df) > 10:
        baro_r = _baro_analysis(baro_df, gps_df, arm_time, drone_type=profile.type)
        metrics["baro"] = baro_r["metrics"]
        issues.extend(baro_r["issues"])
        sub_scores["baro"] = baro_r["score"]

    # ── Overall sensor score ──────────────────────────────────────────────────
    if sub_scores:
        total_w = sum(SUB_WEIGHTS.get(k, 0.1) for k in sub_scores)
        score = sum(sub_scores[k] * SUB_WEIGHTS.get(k, 0.1) for k in sub_scores) / total_w
    else:
        score = 50.0

    score = max(0.0, min(100.0, score))
    grade = _grade(score)
    metrics["sub_scores"] = {k: round(v, 1) for k, v in sub_scores.items()}

    gps_m = metrics.get("gps", {})
    summary_parts = []
    if "fix_pct_good" in gps_m:
        summary_parts.append(f"GPS fix: {gps_m['fix_pct_good']:.0f}% good")
    if "avg_satellites" in gps_m:
        summary_parts.append(f"Sats: {gps_m['avg_satellites']:.0f}")
    if "avg_hdop" in gps_m:
        summary_parts.append(f"HDOP: {gps_m['avg_hdop']:.2f}")
    ekf_m = metrics.get("ekf", {})
    if "ekf_healthy_pct" in ekf_m:
        summary_parts.append(f"EKF health: {ekf_m['ekf_healthy_pct']:.0f}%")

    summary = "  |  ".join(summary_parts) if summary_parts else "Sensor data analysed."

    return {
        "score": round(score, 1),
        "grade": grade,
        "available": True,
        "issues": issues,
        "metrics": metrics,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPS analysis
# ─────────────────────────────────────────────────────────────────────────────

def _gps_analysis(gps_df: pd.DataFrame, profile: DroneProfile, arm_time: float) -> Dict:
    issues: List[Dict] = []
    metrics: Dict = {}

    flight_gps = gps_df[gps_df["timestamp"] >= arm_time].copy() if "timestamp" in gps_df.columns else gps_df.copy()
    if len(flight_gps) == 0:
        flight_gps = gps_df.copy()

    score = 100.0

    # Fix type
    if "Status" in flight_gps.columns:
        fix_types = flight_gps["Status"].values
        min_fix = profile.gps.min_fix_type
        good_fix_pct = float(np.mean(fix_types >= min_fix) * 100.0)
        min_fix_seen = int(fix_types.min())
        max_fix_seen = int(fix_types.max())
        metrics["fix_pct_good"] = round(good_fix_pct, 1)
        metrics["min_fix_type"] = min_fix_seen
        metrics["max_fix_type"] = max_fix_seen
        metrics["fix_type_name"] = GPS_FIX_NAMES.get(max_fix_seen, str(max_fix_seen))

        if good_fix_pct < 95:
            sev = "critical" if good_fix_pct < 80 else "warning"
            issues.append(_issue(
                sev, "SEN-010",
                f"GPS fix quality: only {good_fix_pct:.0f}% of flight with acceptable fix "
                f"(>= {GPS_FIX_NAMES.get(min_fix, str(min_fix))}). "
                "Poor GPS fix increases navigation error and EKF uncertainty.",
                value=round(good_fix_pct, 1), threshold=95,
            ))
            score -= (100 - good_fix_pct) * 0.5

    # Satellite count
    sat_col = "NSats" if "NSats" in flight_gps.columns else ("Satellites" if "Satellites" in flight_gps.columns else None)
    if sat_col:
        sats = flight_gps[sat_col].values
        avg_sats = float(np.mean(sats))
        min_sats = int(sats.min())
        metrics["avg_satellites"] = round(avg_sats, 1)
        metrics["min_satellites"] = min_sats
        metrics["required_satellites"] = profile.gps.min_satellites

        if avg_sats < profile.gps.min_satellites:
            issues.append(_issue(
                "warning", "SEN-011",
                f"Low average satellite count: {avg_sats:.0f} "
                f"(profile minimum: {profile.gps.min_satellites}). "
                "Fly with clearer sky view or wait for better GPS coverage.",
                value=round(avg_sats, 1), threshold=profile.gps.min_satellites,
            ))
            score -= 10.0

        if min_sats < 6:
            issues.append(_issue(
                "critical", "SEN-012",
                f"GPS satellite count dropped to {min_sats} during flight — "
                "dangerously low for reliable navigation.",
                value=min_sats, threshold=6,
            ))
            score -= 20.0

    # HDOP
    hdop_col = "HDop" if "HDop" in flight_gps.columns else ("HDOP" if "HDOP" in flight_gps.columns else None)
    if hdop_col:
        hdop = flight_gps[hdop_col].values
        avg_hdop = float(np.mean(hdop))
        max_hdop = float(np.percentile(hdop, 95))
        metrics["avg_hdop"] = round(avg_hdop, 3)
        metrics["p95_hdop"] = round(max_hdop, 3)
        metrics["hdop_threshold"] = profile.gps.max_hdop

        if avg_hdop > profile.gps.max_hdop:
            issues.append(_issue(
                "warning", "SEN-013",
                f"Elevated HDOP: avg {avg_hdop:.2f} (threshold: {profile.gps.max_hdop:.1f}). "
                "Horizontal position accuracy degraded.",
                value=round(avg_hdop, 3), threshold=profile.gps.max_hdop,
            ))
            score -= 8.0

    # Position jumps
    lat_col = "Lat" if "Lat" in flight_gps.columns else None
    lng_col = "Lng" if "Lng" in flight_gps.columns else ("Lon" if "Lon" in flight_gps.columns else None)
    if lat_col and lng_col and len(flight_gps) > 2:
        jumps = _count_position_jumps(
            flight_gps[lat_col].values,
            flight_gps[lng_col].values,
            profile.gps.max_position_jump_m,
        )
        metrics["position_jumps"] = jumps
        if jumps > 3:
            issues.append(_issue(
                "warning", "SEN-014",
                f"{jumps} GPS position jumps > {profile.gps.max_position_jump_m:.0f} m detected. "
                "Possible multipath interference or GPS glitches.",
                value=jumps, threshold=3,
            ))
            score -= min(15.0, jumps * 2.0)

    return {"issues": issues, "metrics": metrics, "score": max(0.0, min(100.0, score))}


# ─────────────────────────────────────────────────────────────────────────────
# EKF analysis
# ─────────────────────────────────────────────────────────────────────────────

def _ekf_analysis(ekf_df: pd.DataFrame, ekf_type: Optional[str], arm_time: float) -> Dict:
    """
    Check EKF health via innovation ratios and filter flags.
    High innovations mean the filter is struggling to reconcile sensor data.
    """
    issues: List[Dict] = []
    metrics: Dict = {}
    score = 100.0

    flight_ekf = ekf_df[ekf_df["timestamp"] >= arm_time].copy() if "timestamp" in ekf_df.columns else ekf_df.copy()
    if len(flight_ekf) == 0:
        flight_ekf = ekf_df.copy()

    metrics["ekf_type"] = ekf_type or "unknown"

    # Velocity innovation test ratio (SVT) — key health indicator
    # Values: 0 = perfect, 0.5 = borderline, > 1.0 = filter failed/rejected
    svt_col = "SVT" if "SVT" in flight_ekf.columns else None
    if svt_col:
        svt = flight_ekf[svt_col].dropna().values
        avg_svt = float(np.mean(svt))
        max_svt = float(np.percentile(svt, 99))
        high_svt_pct = float(np.mean(svt > 0.5) * 100)

        metrics["avg_velocity_innovation"] = round(avg_svt, 4)
        metrics["max_velocity_innovation"] = round(max_svt, 4)
        metrics["pct_high_innovation"] = round(high_svt_pct, 1)

        if max_svt > 1.0:
            issues.append(_issue(
                "critical", "SEN-020",
                f"EKF velocity innovation exceeded 1.0 (max: {max_svt:.2f}). "
                "Filter rejected velocity data — possible GPS / IMU inconsistency.",
                value=round(max_svt, 3), threshold=1.0,
            ))
            score -= 25.0
        elif max_svt > 0.5:
            issues.append(_issue(
                "warning", "SEN-020",
                f"EKF velocity innovation elevated (max: {max_svt:.2f}, avg: {avg_svt:.3f}). "
                "Sensor disagreement detected — check GPS and IMU calibration.",
                value=round(max_svt, 3), threshold=0.5,
            ))
            score -= 10.0

    # Filter status flags (FS)
    if "FS" in flight_ekf.columns:
        fs_vals = flight_ekf["FS"].dropna().values
        # Non-zero FS = filter fault flags
        fault_pct = float(np.mean(fs_vals != 0) * 100.0)
        healthy_pct = 100.0 - fault_pct
        metrics["ekf_healthy_pct"] = round(healthy_pct, 1)
        metrics["ekf_fault_pct"] = round(fault_pct, 1)

        if fault_pct > 10:
            sev = "critical" if fault_pct > 25 else "warning"
            issues.append(_issue(
                sev, "SEN-021",
                f"EKF reported fault flags for {fault_pct:.1f}% of flight. "
                "Navigation reliability compromised — check sensor calibration.",
                value=round(fault_pct, 1), threshold=10,
            ))
            score -= min(30.0, fault_pct * 0.8)
    else:
        metrics["ekf_healthy_pct"] = None

    # Height innovation (SH)
    if "SH" in flight_ekf.columns:
        sh = flight_ekf["SH"].dropna().values
        avg_sh = float(np.mean(np.abs(sh)))
        metrics["avg_height_innovation"] = round(avg_sh, 4)
        if avg_sh > 0.5:
            issues.append(_issue(
                "warning", "SEN-022",
                f"EKF height innovation elevated (avg: {avg_sh:.3f}). "
                "Barometer and GPS altitude may be inconsistent.",
                value=round(avg_sh, 3), threshold=0.5,
            ))
            score -= 8.0

    return {"issues": issues, "metrics": metrics, "score": max(0.0, min(100.0, score))}


# ─────────────────────────────────────────────────────────────────────────────
# IMU analysis
# ─────────────────────────────────────────────────────────────────────────────

def _imu_analysis(imu_df: pd.DataFrame, imu2_df: Optional[pd.DataFrame], arm_time: float) -> Dict:
    issues: List[Dict] = []
    metrics: Dict = {}
    score = 100.0

    # Use primary IMU only (index 0) — multi-IMU logs interleave IMU0/1/2 records
    imu_primary = imu_df[imu_df["I"] == 0] if "I" in imu_df.columns else imu_df
    flight_imu = imu_primary[imu_primary["timestamp"] >= arm_time].copy() if "timestamp" in imu_primary.columns else imu_primary.copy()

    # Temperature stability — column name varies by ArduPilot version
    temp_col = next((c for c in ["Temp", "T", "temp"] if c in flight_imu.columns), None)
    if temp_col is not None:
        flight_imu = flight_imu.rename(columns={temp_col: "Temp"})
    if "Temp" in flight_imu.columns:
        temps = flight_imu["Temp"].dropna().values
        if len(temps) > 0:
            temp_min = float(temps.min())
            temp_max = float(temps.max())
            temp_range = temp_max - temp_min
            metrics["imu_temp_min_c"] = round(temp_min, 1)
            metrics["imu_temp_max_c"] = round(temp_max, 1)
            metrics["imu_temp_range_c"] = round(temp_range, 1)

            if temp_max > 85:
                issues.append(_issue(
                    "critical", "SEN-030",
                    f"IMU overheating: {temp_max:.0f} °C. Risk of sensor drift and damage.",
                    value=round(temp_max, 1), threshold=85,
                ))
                score -= 20.0
            elif temp_max > 70:
                issues.append(_issue(
                    "warning", "SEN-030",
                    f"High IMU temperature: {temp_max:.0f} °C. Monitor thermal management.",
                    value=round(temp_max, 1), threshold=70,
                ))
                score -= 8.0

            if temp_range > 30:
                issues.append(_issue(
                    "info", "SEN-031",
                    f"Large IMU temperature range ({temp_range:.0f} °C during flight). "
                    "Ensure IMU heater is functioning for consistent calibration.",
                    value=round(temp_range, 1), threshold=30,
                ))

    # Multi-IMU consistency (if IMU2 available)
    if imu2_df is not None and len(imu2_df) > 10:
        flight_imu2 = imu2_df[imu2_df["timestamp"] >= arm_time].copy() if "timestamp" in imu2_df.columns else imu2_df.copy()

        for axis in ["X", "Y", "Z"]:
            col = f"Acc{axis}"
            if col in flight_imu.columns and col in flight_imu2.columns:
                # Align by index length
                n = min(len(flight_imu), len(flight_imu2))
                diff = (flight_imu[col].values[:n] - flight_imu2[col].values[:n])
                rms_diff = float(np.sqrt(np.mean(diff ** 2)))
                metrics[f"imu_imu2_rms_diff_{axis.lower()}"] = round(rms_diff, 4)

                if rms_diff > 3.0:
                    issues.append(_issue(
                        "warning", "SEN-032",
                        f"IMU1 vs IMU2 disagreement on {axis}-axis: {rms_diff:.2f} m/s² RMS. "
                        "Possible vibration isolation issue or IMU miscalibration.",
                        value=round(rms_diff, 3), threshold=3.0,
                    ))
                    score -= 10.0

        metrics["dual_imu"] = True
    else:
        metrics["dual_imu"] = False

    return {"issues": issues, "metrics": metrics, "score": max(0.0, min(100.0, score))}


# ─────────────────────────────────────────────────────────────────────────────
# Magnetometer analysis
# ─────────────────────────────────────────────────────────────────────────────

def _mag_analysis(mag_df: pd.DataFrame, arm_time: float) -> Dict:
    issues: List[Dict] = []
    metrics: Dict = {}
    score = 100.0

    flight_mag = mag_df[mag_df["timestamp"] >= arm_time].copy() if "timestamp" in mag_df.columns else mag_df.copy()
    if len(flight_mag) < 10:
        flight_mag = mag_df.copy()

    for axis in ["X", "Y", "Z"]:
        col = f"Mag{axis}" if f"Mag{axis}" in flight_mag.columns else (
              f"mag{axis}" if f"mag{axis}" in flight_mag.columns else None)
        if col is None:
            continue
        vals = flight_mag[col].dropna().values
        metrics[f"mag_{axis.lower()}_mean"] = round(float(np.mean(vals)), 1)
        metrics[f"mag_{axis.lower()}_std"] = round(float(np.std(vals)), 1)

    # Compute field magnitude and check consistency
    mag_cols = [f"Mag{ax}" for ax in ["X", "Y", "Z"] if f"Mag{ax}" in flight_mag.columns]
    if len(mag_cols) == 3:
        field_mag = np.sqrt(
            flight_mag[mag_cols[0]] ** 2 +
            flight_mag[mag_cols[1]] ** 2 +
            flight_mag[mag_cols[2]] ** 2
        ).values
        avg_field = float(np.mean(field_mag))
        std_field = float(np.std(field_mag))
        cv = std_field / avg_field if avg_field > 0 else 0.0

        metrics["mag_field_strength_mean"] = round(avg_field, 1)
        metrics["mag_field_strength_std"] = round(std_field, 1)
        metrics["mag_field_cv"] = round(cv, 4)

        if cv > 0.20:
            issues.append(_issue(
                "warning", "SEN-040",
                f"Magnetometer field strength inconsistency: CV={cv:.2f}. "
                "Possible magnetic interference — check for nearby metal or motor wiring.",
                value=round(cv, 3), threshold=0.20,
            ))
            score -= 15.0
        elif cv > 0.10:
            issues.append(_issue(
                "info", "SEN-040",
                f"Minor magnetometer variation: CV={cv:.2f}. Monitor over subsequent flights.",
                value=round(cv, 3), threshold=0.10,
            ))

        # Detect interference spikes
        spike_threshold = avg_field + 3 * std_field
        spike_count = int(np.sum(field_mag > spike_threshold))
        metrics["mag_interference_spikes"] = spike_count
        if spike_count > 5:
            issues.append(_issue(
                "warning", "SEN-041",
                f"{spike_count} magnetometer interference spikes detected. "
                "Check compass mount separation from power wires and motors.",
                value=spike_count, threshold=5,
            ))
            score -= 10.0

    return {"issues": issues, "metrics": metrics, "score": max(0.0, min(100.0, score))}


# ─────────────────────────────────────────────────────────────────────────────
# Barometer analysis
# ─────────────────────────────────────────────────────────────────────────────

def _baro_analysis(baro_df: pd.DataFrame, gps_df: Optional[pd.DataFrame], arm_time: float,
                   drone_type: str = "multirotor") -> Dict:
    issues: List[Dict] = []
    metrics: Dict = {}
    score = 100.0

    flight_baro = baro_df[baro_df["timestamp"] >= arm_time].copy() if "timestamp" in baro_df.columns else baro_df.copy()

    if "Alt" in flight_baro.columns:
        alt = flight_baro["Alt"].values
        alt_std = float(np.std(alt))
        metrics["baro_alt_std"] = round(alt_std, 3)

        # For fixed-wing and VTOL aircraft the altitude changes a lot during a
        # survey/transit mission — overall std dev is not a baro health indicator.
        # Instead check short-term noise (std of 2nd-order differences = rapid spikes).
        if drone_type in ("vtol", "fixed_wing"):
            if len(alt) > 10:
                noise = float(np.std(np.diff(np.diff(alt))))
                metrics["baro_noise_m"] = round(noise, 3)
                if noise > 3.0:
                    issues.append(_issue(
                        "info", "SEN-050",
                        f"Barometer short-term noise: {noise:.2f} m (2nd-order diff std). "
                        "Check baro foam seal and vibration isolation.",
                        value=round(noise, 2), threshold=3.0,
                    ))
                    score -= 5.0
        else:
            # Multirotor: overall std dev is valid (should hover at constant altitude)
            if alt_std > 5.0:
                issues.append(_issue(
                    "info", "SEN-050",
                    f"Barometer altitude noise: {alt_std:.1f} m std dev. "
                    "Check baro is shielded from prop wash and sunlight.",
                    value=round(alt_std, 2), threshold=5.0,
                ))
                score -= 5.0

    if "Press" in flight_baro.columns and drone_type not in ("vtol", "fixed_wing"):
        # Fixed-wing and VTOL aircraft change airspeed frequently — dynamic pressure
        # on the baro port produces hundreds of normal pressure steps per minute.
        # The 50 Pa threshold is too sensitive and generates false positives.
        press = flight_baro["Press"].values
        press_spikes = int(np.sum(np.abs(np.diff(press)) > 50))
        metrics["baro_pressure_spikes"] = press_spikes
        if press_spikes > 5:
            issues.append(_issue(
                "info", "SEN-051",
                f"{press_spikes} sudden barometer pressure changes detected. "
                "Possible airflow interference on baro port.",
                value=press_spikes, threshold=5,
            ))
            score -= 5.0

    return {"issues": issues, "metrics": metrics, "score": max(0.0, min(100.0, score))}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_position_jumps(lats: np.ndarray, lons: np.ndarray, threshold_m: float) -> int:
    R = 6_371_000
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    dlat = np.radians(lats[1:] - lats[:-1])
    dlon = np.radians(lons[1:] - lons[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return int(np.sum(dist > threshold_m))


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _issue(severity, code, message, value=None, threshold=None, timestamp=None):
    d = {"severity": severity, "code": code, "message": message}
    if value is not None: d["value"] = value
    if threshold is not None: d["threshold"] = threshold
    if timestamp is not None: d["timestamp_s"] = timestamp
    return d
