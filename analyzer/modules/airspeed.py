"""
Airspeed Health Module
=======================
Phase 4.4 — Airspeed envelope analysis using GPS groundspeed as a proxy

Notes
-----
- None of the current logs contain an ARSP (airspeed sensor) message, so GPS
  groundspeed (GSPD column of GPS log) is used as a proxy for airspeed.
- This is an acceptable approximation in low-wind conditions and for cruising
  flight.  It is explicitly flagged as a proxy in all issue messages.
- The module is NOT applicable to multirotors (available=False, score=100).
  Multirotors do not have a minimum airspeed envelope.
- For VTOL (quadplane), only fixed-wing cruise segments are analysed.
  Q-modes (QHOVER, QLOITER, QLAND, QRTL, QSTABILIZE) are excluded.

Speed thresholds (derived from profile)
----------------------------------------
  stall_proxy_ms    = max_speed_ms * 0.40   (40% of cruise speed)
  overspeed_ms      = max_speed_ms * 1.25   (25% above max speed)

Scoring (from 100)
------------------
  -20   > 5% of cruise samples below stall proxy (stall risk)
  -10   > 2% of cruise samples above overspeed threshold
   -5   airspeed variability > 30% of mean (high turbulence / instability)

Issue codes
-----------
  ASP-001  Info — GPS groundspeed used as airspeed proxy (always, when available)
  ASP-002  Stall risk: > 5% time below minimum airspeed proxy
  ASP-003  Overspeed: > 2% time above maximum airspeed
  ASP-004  High airspeed variability (std/mean > 0.30)

Availability
------------
  available=False (score 100) when:
  - Profile type is 'multirotor'
  - No GPS data in log
  - No cruise-phase GPS samples found (e.g., only Q-mode samples for VTOL)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# Q-mode names that are NOT fixed-wing cruise (VTOL hover/landing modes)
_Q_MODES = {
    "QSTABILIZE", "QHOVER", "QLOITER", "QLAND", "QRTL",
    "QAUTOTUNE", "QACRO",
}

# Fixed-wing cruise modes — only these are analysed for fixed_wing type
# Excludes MANUAL/STABILIZE (takeoff roll), LAND, CIRCLE (loiter), etc.
# Note: parser maps ArduPlane mode 5 → "FLY_BY_WIRE_A", mode 6 → "FLY_BY_WIRE_B"
_FW_CRUISE_MODES = {
    "AUTO", "GUIDED", "FLY_BY_WIRE_B", "CRUISE", "FLY_BY_WIRE_A",
    "LOITER", "AUTOTUNE", "RTL",
}

# Stall proxy = this fraction of max_speed_ms
STALL_PROXY_FRAC  = 0.40
# Overspeed threshold = this fraction of max_speed_ms
OVERSPEED_FRAC    = 1.25

# Scoring thresholds
STALL_TIME_WARN_PCT    = 5.0    # % of samples below stall proxy -> penalty
OVERSPEED_TIME_PCT     = 2.0    # % of samples above overspeed -> penalty
VARIABILITY_WARN_RATIO = 0.30   # std/mean above this -> variability warning


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Airspeed Health — Phase 4.4."""

    issues:  List[Dict] = []
    metrics: Dict[str, Any] = {}

    def _unavailable(reason: str) -> Dict[str, Any]:
        return {
            "score": 100.0,
            "grade": "A",
            "available": False,
            "issues": [],
            "metrics": {"unavailable_reason": reason},
            "summary": f"Airspeed: {reason}",
        }

    # ── Multirotors do not have an airspeed envelope ──────────────────────────
    if profile.type == "multirotor":
        return _unavailable("Not applicable for multirotor — no minimum airspeed envelope.")

    # ── Check for dedicated airspeed sensor (ARSP) first ─────────────────────
    arsp_df = result.get("ARSP")
    arsp_col = None
    using_arsp = False
    if arsp_df is not None and not arsp_df.empty:
        arsp_col = next((c for c in ("Airspeed", "airspeed", "IAS", "TAS")
                         if c in arsp_df.columns), None)
        if arsp_col is not None:
            using_arsp = True
            logger.info("Dedicated airspeed sensor (ARSP) detected — using ARSP.%s", arsp_col)

    # ── GPS data (fallback when no ARSP) ─────────────────────────────────────
    mode_df = result.get("MODE")

    if using_arsp:
        # Use ARSP sensor data
        speed_df = arsp_df.copy()
        spd_col  = arsp_col
        speed_source = "Dedicated airspeed sensor (ARSP)"
    else:
        gps_df = result.get("GPS")
        if gps_df is None or gps_df.empty:
            return _unavailable("No GPS data available in this log.")
        spd_col = next((c for c in ("Spd", "GSpd", "spd", "gspd", "GSpeed")
                        if c in gps_df.columns), None)
        if spd_col is None:
            return _unavailable("GPS speed column not found (expected 'Spd' or 'GSpd').")
        speed_df = gps_df.copy()
        speed_source = "GPS groundspeed (proxy)"

    # ── Restrict to armed window ──────────────────────────────────────────────
    if "timestamp" in speed_df.columns:
        speed_df = speed_df[speed_df["timestamp"] >= arm_time]
        if disarm_time is not None:
            speed_df = speed_df[speed_df["timestamp"] <= disarm_time]

    # ── Filter to cruise-phase only ────────────────────────────────────────────
    if mode_df is not None and not mode_df.empty:
        if profile.type == "vtol":
            speed_df = _filter_fw_cruise(speed_df, mode_df)
        elif profile.type == "fixed_wing":
            speed_df = _filter_fw_modes_only(speed_df, mode_df)

    if speed_df.empty:
        return _unavailable("No cruise-phase samples found after mode filtering.")

    speeds = speed_df[spd_col].dropna().values.astype(float)
    # Filter out implausible values (sensor noise, pre-arm zeros)
    speeds = speeds[speeds > 0.5]
    if len(speeds) < 10:
        return _unavailable("Insufficient airspeed samples for analysis (< 10 samples).")

    # ── Thresholds ────────────────────────────────────────────────────────────
    max_spd       = float(profile.max_speed_ms)
    stall_proxy   = max_spd * STALL_PROXY_FRAC
    overspeed_thr = max_spd * OVERSPEED_FRAC

    # ── Compute statistics ────────────────────────────────────────────────────
    mean_spd = float(np.mean(speeds))
    std_spd  = float(np.std(speeds))
    min_spd  = float(np.min(speeds))
    max_meas = float(np.max(speeds))

    n_total       = len(speeds)
    n_below_stall  = int(np.sum(speeds < stall_proxy))
    n_above_ospeed = int(np.sum(speeds > overspeed_thr))

    stall_pct     = n_below_stall  / n_total * 100.0
    overspeed_pct = n_above_ospeed / n_total * 100.0
    variability   = std_spd / mean_spd if mean_spd > 0.5 else 0.0

    metrics.update({
        "speed_source":      speed_source,
        "using_arsp":        using_arsp,
        "mean_speed_ms":     round(mean_spd, 2),
        "std_speed_ms":      round(std_spd, 2),
        "min_speed_ms":      round(min_spd, 2),
        "max_speed_ms":      round(max_meas, 2),
        "stall_proxy_ms":    round(stall_proxy, 2),
        "overspeed_thr_ms":  round(overspeed_thr, 2),
        "stall_time_pct":    round(stall_pct, 1),
        "overspeed_pct":     round(overspeed_pct, 1),
        "variability_ratio": round(variability, 3),
        "sample_count":      n_total,
    })

    # ASP-001: info notice — GPS proxy OR ARSP detected
    if using_arsp:
        issues.append({
            "severity": "info",
            "code": "ASP-001",
            "message": (
                f"Dedicated airspeed sensor (ARSP) detected and used. "
                f"Mean {mean_spd:.1f} m/s  |  min {min_spd:.1f} m/s  |  max {max_meas:.1f} m/s  "
                f"({n_total} cruise samples)."
            ),
        })
    else:
        issues.append({
            "severity": "info",
            "code": "ASP-001",
            "message": (
                f"Airspeed analysis uses GPS groundspeed as a proxy (no ARSP sensor detected). "
                f"Mean {mean_spd:.1f} m/s  |  min {min_spd:.1f} m/s  |  max {max_meas:.1f} m/s  "
                f"({n_total} cruise samples). Install a dedicated airspeed sensor (ARSP) "
                f"for more accurate stall and overspeed protection."
            ),
        })

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = 100.0
    deductions: List[str] = []

    if stall_pct > STALL_TIME_WARN_PCT:
        # When groundspeed variability is very high (std/mean > 0.50), wind effects are
        # likely dominating the GPS groundspeed — headwind alone can push groundspeed below
        # the stall proxy even at a perfectly safe airspeed.  Downgrade to warning and
        # reduce the score penalty proportionally so the score reflects measurement confidence.
        gusty_proxy = (variability > 0.50)
        score -= 5 if gusty_proxy else 20
        deductions.append(f"stall risk {stall_pct:.1f}%")
        if gusty_proxy:
            stall_sev = "warning"
            wind_note = (
                f" NOTE: GPS groundspeed variability is very high (std/mean = {variability:.2f}), "
                f"suggesting significant wind effects. Headwind gusts may reduce groundspeed below "
                f"the proxy threshold without the aircraft actually approaching stall. "
                f"Install a dedicated airspeed sensor (ARSP) for accurate stall detection."
            )
        else:
            stall_sev = "critical"
            wind_note = ""
        issues.append({
            "severity": stall_sev,
            "code": "ASP-002",
            "message": (
                f"Stall risk: {stall_pct:.1f}% of cruise time below minimum airspeed proxy "
                f"({stall_proxy:.1f} m/s). Check ARSPD_FBW_MIN and approach speeds.{wind_note}"
            ),
            "value": round(stall_pct, 1),
            "threshold": STALL_TIME_WARN_PCT,
        })

    if overspeed_pct > OVERSPEED_TIME_PCT:
        score -= 10
        deductions.append(f"overspeed {overspeed_pct:.1f}%")
        issues.append({
            "severity": "warning",
            "code": "ASP-003",
            "message": (
                f"Overspeed: {overspeed_pct:.1f}% of cruise time above {overspeed_thr:.1f} m/s "
                f"({OVERSPEED_FRAC*100:.0f}% of max speed). Check ARSPD_FBW_MAX."
            ),
            "value": round(overspeed_pct, 1),
            "threshold": OVERSPEED_TIME_PCT,
        })

    if variability > VARIABILITY_WARN_RATIO:
        score -= 5
        deductions.append(f"high variability ({variability:.2f})")
        issues.append({
            "severity": "warning",
            "code": "ASP-004",
            "message": (
                f"High airspeed variability (std/mean = {variability:.2f}). "
                f"May indicate turbulence, gusts, or throttle hunting. "
                f"Review TECS_THR_DAMP or L1 controller settings."
            ),
            "value": round(variability, 3),
            "threshold": VARIABILITY_WARN_RATIO,
        })

    score = max(0.0, min(100.0, score))
    grade = _grade(score)

    # ── Summary ───────────────────────────────────────────────────────────────
    src_tag = "ARSP sensor" if using_arsp else "GPS proxy"
    summary = f"Mean {mean_spd:.1f} m/s  |  min {min_spd:.1f} m/s  |  max {max_meas:.1f} m/s  [{src_tag}]"
    if deductions:
        summary += "  [" + ", ".join(deductions) + "]"

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_fw_modes_only(gps_df, mode_df) -> "pd.DataFrame":
    """
    Return only GPS rows that fall within known fixed-wing cruise modes
    (AUTO, GUIDED, FBWB, CRUISE, FBWA, LOITER, etc.).
    Excludes MANUAL, STABILIZE, LAND, CIRCLE modes used during takeoff/landing.
    """
    if "timestamp" not in gps_df.columns or "timestamp" not in mode_df.columns:
        return gps_df

    mode_sorted = mode_df.sort_values("timestamp").reset_index(drop=True)
    gps_ts = gps_df["timestamp"].values
    mode_ts = mode_sorted["timestamp"].values
    mode_names = mode_sorted["mode_name"].values if "mode_name" in mode_sorted.columns else None
    if mode_names is None:
        return gps_df

    import numpy as np
    indices = np.searchsorted(mode_ts, gps_ts, side="right") - 1
    indices = np.clip(indices, 0, len(mode_names) - 1)
    active_modes = mode_names[indices]
    mask = np.array([m in _FW_CRUISE_MODES for m in active_modes])
    return gps_df[mask].reset_index(drop=True)


def _filter_fw_cruise(gps_df, mode_df) -> "pd.DataFrame":
    """
    Return only GPS rows that fall within fixed-wing cruise segments
    (i.e., not in any Q-mode).
    """
    if "timestamp" not in gps_df.columns or "timestamp" not in mode_df.columns:
        return gps_df   # can't filter — return all

    mode_sorted = mode_df.sort_values("timestamp").reset_index(drop=True)
    gps_ts = gps_df["timestamp"].values

    # For each GPS sample, find the active mode via searchsorted
    mode_ts = mode_sorted["timestamp"].values
    mode_names = mode_sorted["mode_name"].values if "mode_name" in mode_sorted.columns else None
    if mode_names is None:
        return gps_df

    # searchsorted gives the insertion point — subtract 1 for the current mode
    import numpy as np
    indices = np.searchsorted(mode_ts, gps_ts, side="right") - 1
    indices = np.clip(indices, 0, len(mode_names) - 1)

    active_modes = mode_names[indices]
    mask = np.array([m not in _Q_MODES for m in active_modes])
    return gps_df[mask].reset_index(drop=True)


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"
