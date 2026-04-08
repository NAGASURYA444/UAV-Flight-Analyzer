"""
Landing Quality & Wind Analysis Module
========================================
Phase 4.2 — Final approach quality and ambient wind estimation

Landing Quality:
  - Detected from final BARO altitude descent to ground level
  - Descent rate at touchdown (hard / rough / normal)
  - Horizontal speed at touchdown (multirotor: should be near zero)
  - Bounce detection (post-touchdown altitude reversal; multirotor/vtol)
  - Approach stability (ATT tracking error during final approach)

Wind Analysis (best-available method, in priority order):
  1. EKF wind estimate  — XKF2 Vwn / Vwe (direct, most accurate)
  2. Crab angle method  — GPS ground track vs ATT heading (fixed_wing / vtol)
  3. Hover tilt method  — mean roll/pitch lean during near-stationary hover (multirotor)

Issue codes
-----------
  LND-001  Hard landing: descent rate > 3.0 m/s at touchdown
  LND-002  Rough landing: descent rate 1.5–3.0 m/s
  LND-003  High horizontal speed at touchdown (multirotor, > 2.0 m/s)
  LND-004  Landing bounce detected after touchdown
  LND-005  Unstable approach: ATT tracking error elevated during approach

  WND-001  Info — estimated wind speed and direction (always, when available)
  WND-002  Strong wind detected (> 10 m/s)
  WND-003  Gusty conditions (wind speed std > 3 m/s)

Scoring deductions (from 100)
------------------------------
  -20  Hard landing
  -10  Rough landing
  -10  High-speed touchdown (multirotor only)
   -5  Bounce
   -5  Unstable approach
   -5  Strong wind (WND-002)
   -5  Gusty (WND-003)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
HARD_LANDING_MS       = 3.0    # m/s descent rate → hard landing
ROUGH_LANDING_MS      = 1.5    # m/s descent rate → rough landing
COPTER_MAX_HORIZ_MS   = 2.0    # m/s horizontal speed at touchdown (multirotor)
BOUNCE_AGL_M          = 1.5    # m altitude rise after touchdown = bounce
APPROACH_UNSTABLE_DEG = 5.0    # roll/pitch RMS during approach
TOUCHDOWN_MARGIN_M    = 8.0    # m above landed altitude to call "touchdown"
APPROACH_AGL_M        = 30.0   # m above landed alt where approach window starts
LOOK_BACK_S           = 180.0  # seconds before disarm to search for landing
STRONG_WIND_MS        = 10.0   # m/s mean wind → WND-002
GUSTY_WIND_STD_MS     = 3.0    # m/s wind std → WND-003
MIN_CRUISE_SPD_MS     = 5.0    # minimum ground speed for crab angle computation
EV_LAND_ID            = 18     # ArduPilot EV message Id for "Land Complete"


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Landing Quality & Wind Analysis — Phase 4.2."""

    issues:  List[Dict] = []
    metrics: Dict[str, Any] = {}

    baro_df = result.get("BARO")
    gps_df  = result.get("GPS")
    att_df  = result.get("ATT")
    xkf2_df = result.get("XKF2")
    ev_df   = result.get("EV")

    end_time = disarm_time if disarm_time is not None else (
        float(baro_df["timestamp"].max()) if baro_df is not None else arm_time + 600.0
    )

    # ── Landing Quality ───────────────────────────────────────────────────────
    lnd_metrics, lnd_issues = _analyse_landing(
        baro_df, gps_df, att_df, ev_df, profile, arm_time, end_time
    )
    metrics.update(lnd_metrics)
    issues.extend(lnd_issues)

    # ── Wind Analysis ─────────────────────────────────────────────────────────
    wnd_metrics, wnd_issues = _analyse_wind(
        xkf2_df, gps_df, att_df, profile, arm_time, end_time
    )
    metrics.update(wnd_metrics)
    issues.extend(wnd_issues)

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues)
    grade = _grade(score)

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if metrics.get("landing_detected"):
        dr = metrics.get("descent_rate_ms")
        parts.append(f"Descent rate: {dr:.1f} m/s" if dr is not None else "Landing detected")
    else:
        parts.append("Landing not detected")

    ws = metrics.get("wind_speed_ms")
    wdir = metrics.get("wind_direction_deg")
    if ws is not None:
        dir_str = f" from {wdir:.0f}°" if wdir is not None else ""
        parts.append(f"Wind: {ws:.1f} m/s{dir_str}")
    else:
        parts.append("Wind: N/A")

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
# Landing Quality
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_landing(
    baro_df:  Optional[pd.DataFrame],
    gps_df:   Optional[pd.DataFrame],
    att_df:   Optional[pd.DataFrame],
    ev_df:    Optional[pd.DataFrame],
    profile:  DroneProfile,
    arm_time: float,
    end_time: float,
) -> Tuple[Dict, List[Dict]]:

    issues: List[Dict] = []
    metrics: Dict[str, Any] = {
        "landing_detected":       False,
        "touchdown_s":            None,
        "approach_start_s":       None,
        "descent_rate_ms":        None,
        "horizontal_speed_ms":    None,
        "bounce_detected":        False,
        "approach_roll_rms_deg":  None,
        "approach_pitch_rms_deg": None,
    }

    if baro_df is None or len(baro_df) < 20:
        return metrics, issues

    # Ground altitude reference (BARO reading just after arming)
    ground_alt = _get_ground_alt(baro_df, arm_time)
    metrics["ground_alt_m"] = round(ground_alt, 1)

    # Detect approach window and touchdown timestamp
    approach_start, touchdown_s = _detect_landing_window(
        baro_df, ev_df, arm_time, end_time
    )
    if touchdown_s is None:
        return metrics, issues

    metrics["landing_detected"] = True
    metrics["touchdown_s"]      = round(touchdown_s, 2)
    metrics["approach_start_s"] = round(approach_start, 2) if approach_start is not None else None

    # ── Descent rate at touchdown ──────────────────────────────────────────────
    descent_rate = _compute_descent_rate(
        baro_df,
        approach_start if approach_start is not None else (touchdown_s - 10.0),
        touchdown_s,
    )
    if descent_rate is not None:
        descent_rate = max(0.0, descent_rate)
        metrics["descent_rate_ms"] = round(descent_rate, 2)

        if descent_rate >= HARD_LANDING_MS:
            issues.append(_issue(
                "critical", "LND-001",
                f"Hard landing: descent rate {descent_rate:.1f} m/s at touchdown "
                f"(threshold {HARD_LANDING_MS:.1f} m/s). "
                "Inspect frame, landing gear, and motor mounts for damage.",
                value=round(descent_rate, 2),
                threshold=HARD_LANDING_MS,
                timestamp=touchdown_s,
            ))
        elif descent_rate >= ROUGH_LANDING_MS:
            issues.append(_issue(
                "warning", "LND-002",
                f"Rough landing: descent rate {descent_rate:.1f} m/s at touchdown "
                f"(threshold {ROUGH_LANDING_MS:.1f} m/s). "
                "Check landing gear and frame integrity.",
                value=round(descent_rate, 2),
                threshold=ROUGH_LANDING_MS,
                timestamp=touchdown_s,
            ))

    # ── Horizontal speed at touchdown ─────────────────────────────────────────
    if gps_df is not None:
        horiz_speed = _compute_horizontal_speed(gps_df, touchdown_s)
        if horiz_speed is not None:
            metrics["horizontal_speed_ms"] = round(horiz_speed, 2)
            if profile.type == "multirotor" and horiz_speed > COPTER_MAX_HORIZ_MS:
                issues.append(_issue(
                    "warning", "LND-003",
                    f"High horizontal speed at touchdown: {horiz_speed:.1f} m/s "
                    f"(multirotor should land with < {COPTER_MAX_HORIZ_MS:.1f} m/s). "
                    "Review landing approach path or LAND_SPEED_HIGH parameter.",
                    value=round(horiz_speed, 2),
                    threshold=COPTER_MAX_HORIZ_MS,
                    timestamp=touchdown_s,
                ))

    # ── Bounce detection (multirotor / vtol only) ─────────────────────────────
    if profile.type in ("multirotor", "vtol"):
        bounced = _detect_bounce(baro_df, touchdown_s, end_time)
        metrics["bounce_detected"] = bounced
        if bounced:
            issues.append(_issue(
                "warning", "LND-004",
                "Landing bounce detected — aircraft left the ground after initial touchdown. "
                "Reduce final descent speed (LAND_SPEED) or check landing surface conditions.",
                timestamp=touchdown_s,
            ))

    # ── Approach stability ────────────────────────────────────────────────────
    if att_df is not None and approach_start is not None:
        roll_rms, pitch_rms = _approach_stability(att_df, approach_start, touchdown_s)
        if roll_rms is not None:
            metrics["approach_roll_rms_deg"]  = round(roll_rms,  2)
            metrics["approach_pitch_rms_deg"] = round(pitch_rms, 2)
            max_rms = max(roll_rms, pitch_rms)
            if max_rms > APPROACH_UNSTABLE_DEG:
                issues.append(_issue(
                    "warning", "LND-005",
                    f"Unstable approach: attitude error RMS {max_rms:.1f}° during final approach "
                    f"(threshold {APPROACH_UNSTABLE_DEG:.1f}°). "
                    "Check PID tuning, reduce approach speed, or review wind conditions at landing.",
                    value=round(max_rms, 2),
                    threshold=APPROACH_UNSTABLE_DEG,
                ))

    return metrics, issues


def _get_ground_alt(baro_df: pd.DataFrame, arm_time: float) -> float:
    """BARO altitude reference at ground level (just after arming)."""
    near_arm = baro_df[baro_df["timestamp"] >= arm_time].head(5)
    if len(near_arm) == 0:
        return float(baro_df["Alt"].iloc[0])
    return float(near_arm["Alt"].mean())


def _detect_landing_window(
    baro_df:  pd.DataFrame,
    ev_df:    Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find (approach_start_s, touchdown_s).

    Method 1 (primary): ArduPilot EV message Id=18 (Land Complete).
    Method 2 (fallback): adaptive BARO descent detection — uses minimum BARO
    altitude in the last 30 s as the landed reference, which is robust to
    baro drift accumulated over a long flight.
    """
    # ── Method 1: EV land event ───────────────────────────────────────────────
    if ev_df is not None and len(ev_df) > 0 and "Id" in ev_df.columns:
        land_evs = ev_df[
            (ev_df["Id"] == EV_LAND_ID) &
            (ev_df["timestamp"] >= arm_time) &
            (ev_df["timestamp"] <= end_time)
        ]
        if len(land_evs) > 0:
            touchdown_s = float(land_evs["timestamp"].iloc[-1])
            approach_start = _find_approach_start_baro(baro_df, touchdown_s, end_time)
            return approach_start, touchdown_s

    # ── Method 2: Adaptive BARO descent ──────────────────────────────────────
    if baro_df is None or len(baro_df) < 10:
        return None, None

    seg = baro_df[
        (baro_df["timestamp"] >= max(arm_time, end_time - LOOK_BACK_S)) &
        (baro_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(seg) < 10:
        return None, None

    alts = seg["Alt"].values.astype(float)
    ts   = seg["timestamp"].values.astype(float)

    # Adaptive ground reference: min altitude in last 30 s before disarm.
    # This accounts for baro drift accumulated over a long flight so the
    # threshold stays relative to the actual landing altitude, not arm time.
    last_30_mask = ts >= (end_time - 30.0)
    if np.sum(last_30_mask) >= 3:
        landed_alt = float(np.min(alts[last_30_mask]))
    else:
        landed_alt = float(np.min(alts[-max(3, len(alts) // 10):]))

    agl = alts - landed_alt   # AGL relative to actual landing point

    below = agl < TOUCHDOWN_MARGIN_M   # 8 m above landed altitude
    if not np.any(below):
        return None, None

    # Find last downward crossing of the threshold
    touchdown_idx = None
    for i in range(len(agl) - 1, 0, -1):
        if below[i] and not below[i - 1]:
            touchdown_idx = i
            break

    if touchdown_idx is None:
        touchdown_idx = int(np.argmax(below))

    touchdown_s = float(ts[touchdown_idx])

    # Approach start: last time AGL was above APPROACH_AGL_M before touchdown
    approach_start: Optional[float] = None
    for i in range(touchdown_idx, -1, -1):
        if agl[i] > APPROACH_AGL_M:
            approach_start = float(ts[i])
            break
    if approach_start is None:
        approach_start = float(ts[0])

    return approach_start, touchdown_s


def _find_approach_start_baro(
    baro_df:     Optional[pd.DataFrame],
    touchdown_s: float,
    end_time:    float,
) -> Optional[float]:
    """Find approach start from BARO data given a known touchdown time."""
    if baro_df is None:
        return None
    seg = baro_df[
        (baro_df["timestamp"] >= max(0, touchdown_s - LOOK_BACK_S)) &
        (baro_df["timestamp"] <= touchdown_s)
    ].copy().reset_index(drop=True)
    if len(seg) < 5:
        return None

    alts = seg["Alt"].values.astype(float)
    ts   = seg["timestamp"].values.astype(float)
    td_alt = float(alts[-1])   # baro at touchdown = reference

    for i in range(len(alts) - 1, -1, -1):
        if (alts[i] - td_alt) > APPROACH_AGL_M:
            return float(ts[i])
    return float(ts[0])


def _compute_descent_rate(
    baro_df:        pd.DataFrame,
    approach_start: float,
    touchdown_s:    float,
    window_s:       float = 10.0,
) -> Optional[float]:
    """Mean descent rate (m/s, positive = descending) in final window before touchdown."""
    seg = baro_df[
        (baro_df["timestamp"] >= max(approach_start, touchdown_s - window_s)) &
        (baro_df["timestamp"] <= touchdown_s)
    ].copy().reset_index(drop=True)

    if len(seg) < 4:
        return None

    # Prefer logged climb-rate column (CRt) if present
    if "CRt" in seg.columns:
        crt = seg["CRt"].dropna().values.astype(float)
        if len(crt) >= 3:
            # CRt may be in m/s or cm/s — detect by magnitude
            mean_crt = float(np.mean(np.abs(crt)))
            if mean_crt > 50:        # likely cm/s
                crt = crt / 100.0
            return float(-np.mean(crt))   # positive = descending

    # Fallback: differentiate altitude
    ts   = seg["timestamp"].values.astype(float)
    alts = seg["Alt"].values.astype(float)
    if len(ts) < 2:
        return None
    vz = np.diff(alts) / np.diff(ts)   # m/s, positive = ascending
    return float(-np.mean(vz))          # negate: positive = descending


def _compute_horizontal_speed(
    gps_df:      pd.DataFrame,
    touchdown_s: float,
    window_s:    float = 5.0,
) -> Optional[float]:
    """Mean GPS ground speed (m/s) around touchdown."""
    spd_col = _find_col(gps_df, ["Spd", "spd", "Speed", "speed"])
    if spd_col is None:
        return None
    seg = gps_df[
        (gps_df["timestamp"] >= touchdown_s - window_s) &
        (gps_df["timestamp"] <= touchdown_s + 2.0)
    ]
    if len(seg) < 2:
        return None
    return float(seg[spd_col].mean())


def _detect_bounce(
    baro_df:     pd.DataFrame,
    touchdown_s: float,
    end_time:    float,
) -> bool:
    """True if altitude rises significantly after initial touchdown.

    Uses relative altitude change (max - min) in the post-touchdown window,
    which is robust to baro drift — no fixed ground reference needed.
    """
    seg = baro_df[
        (baro_df["timestamp"] >= touchdown_s) &
        (baro_df["timestamp"] <= min(end_time, touchdown_s + 15.0))
    ].copy().reset_index(drop=True)

    if len(seg) < 5:
        return False

    alts    = seg["Alt"].values.astype(float)
    min_alt = float(np.min(alts))
    max_alt = float(np.max(alts))

    # Only flag if the minimum is not at the very end (requires post-min data)
    min_idx = int(np.argmin(alts))
    if min_idx >= len(alts) - 3:
        return False   # minimum at end of window — no subsequent rise to judge

    return (max_alt - min_alt) > BOUNCE_AGL_M


def _approach_stability(
    att_df:        pd.DataFrame,
    approach_start: float,
    touchdown_s:   float,
) -> Tuple[Optional[float], Optional[float]]:
    """Roll/pitch tracking error RMS during the approach window."""
    seg = att_df[
        (att_df["timestamp"] >= approach_start) &
        (att_df["timestamp"] <= touchdown_s)
    ]
    if len(seg) < 10:
        return None, None

    if "DesRoll" not in seg.columns or "DesPitch" not in seg.columns:
        return None, None

    roll_err  = (seg["Roll"]  - seg["DesRoll"]).abs().values.astype(float)
    pitch_err = (seg["Pitch"] - seg["DesPitch"]).abs().values.astype(float)

    return (
        float(np.sqrt(np.mean(roll_err  ** 2))),
        float(np.sqrt(np.mean(pitch_err ** 2))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wind Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_wind(
    xkf2_df:  Optional[pd.DataFrame],
    gps_df:   Optional[pd.DataFrame],
    att_df:   Optional[pd.DataFrame],
    profile:  DroneProfile,
    arm_time: float,
    end_time: float,
) -> Tuple[Dict, List[Dict]]:
    """Try wind estimation methods in priority order."""

    # Priority 1: EKF wind estimate (XKF2 — most accurate)
    if xkf2_df is not None and not xkf2_df.empty:
        vwn_col = _find_col(xkf2_df, ["Vwn", "VWN", "vwn"])
        vwe_col = _find_col(xkf2_df, ["Vwe", "VWE", "vwe"])
        if vwn_col and vwe_col:
            m, i = _wind_from_xkf2(xkf2_df, arm_time, end_time, vwn_col, vwe_col)
            if m.get("wind_available"):
                return m, i

    # Priority 2: Crab angle (fixed_wing / vtol — heading vs ground track)
    if profile.type in ("fixed_wing", "vtol") and gps_df is not None and att_df is not None:
        m, i = _wind_from_crab_angle(gps_df, att_df, arm_time, end_time)
        if m.get("wind_available"):
            return m, i

    # Priority 3: Hover tilt estimate (multirotor)
    if profile.type == "multirotor" and att_df is not None:
        m, i = _wind_from_hover_tilt(att_df, gps_df, arm_time, end_time)
        if m.get("wind_available"):
            return m, i

    return {"wind_available": False, "wind_method": "none"}, []


def _wind_from_xkf2(
    xkf2_df: pd.DataFrame,
    arm_time: float,
    end_time: float,
    vwn_col:  str,
    vwe_col:  str,
) -> Tuple[Dict, List[Dict]]:
    seg = xkf2_df[
        (xkf2_df["timestamp"] >= arm_time) &
        (xkf2_df["timestamp"] <= end_time)
    ].copy()
    if len(seg) < 10:
        return {"wind_available": False, "wind_method": "xkf2"}, []

    vwn = seg[vwn_col].values.astype(float)
    vwe = seg[vwe_col].values.astype(float)

    speeds = np.sqrt(vwn ** 2 + vwe ** 2)
    mean_vwn = float(np.mean(vwn))
    mean_vwe = float(np.mean(vwe))
    mean_dir = float((math.degrees(math.atan2(mean_vwe, mean_vwn)) + 360) % 360)

    metrics = {
        "wind_available":     True,
        "wind_method":        "EKF (XKF2)",
        "wind_speed_ms":      round(float(np.mean(speeds)), 1),
        "wind_speed_std_ms":  round(float(np.std(speeds)), 1),
        "wind_speed_max_ms":  round(float(np.max(speeds)), 1),
        "wind_direction_deg": round(mean_dir, 0),
    }
    return metrics, _build_wind_issues(metrics)


def _wind_from_crab_angle(
    gps_df:   pd.DataFrame,
    att_df:   pd.DataFrame,
    arm_time: float,
    end_time: float,
) -> Tuple[Dict, List[Dict]]:
    """Crosswind estimate from GPS ground track vs aircraft heading (fixed-wing)."""
    spd_col  = _find_col(gps_df, ["Spd",  "spd",  "Speed"])
    gcrs_col = _find_col(gps_df, ["GCrs", "Gcrs", "gcrs", "Crs"])
    yaw_col  = _find_col(att_df, ["Yaw",  "yaw"])

    if None in (spd_col, gcrs_col, yaw_col):
        return {"wind_available": False, "wind_method": "crab_angle"}, []

    gps_seg = gps_df[
        (gps_df["timestamp"] >= arm_time) &
        (gps_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    att_seg = att_df[
        (att_df["timestamp"] >= arm_time) &
        (att_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(gps_seg) < 20 or len(att_seg) < 20:
        return {"wind_available": False, "wind_method": "crab_angle"}, []

    # Interpolate ATT yaw onto GPS timestamps
    yaw_interp = np.interp(
        gps_seg["timestamp"].values,
        att_seg["timestamp"].values,
        att_seg[yaw_col].values.astype(float),
    )

    speeds = gps_seg[spd_col].values.astype(float)
    gcrs   = gps_seg[gcrs_col].values.astype(float)

    # Also compute yaw rate to filter out turns (high yaw rate = banked turn)
    yaw_rate = np.abs(np.gradient(yaw_interp, gps_seg["timestamp"].values))

    # Keep only straight cruise segments
    cruise_mask = (speeds >= MIN_CRUISE_SPD_MS) & (yaw_rate < 5.0)
    if np.sum(cruise_mask) < 10:
        return {"wind_available": False, "wind_method": "crab_angle"}, []

    crab  = _wrap_angle(yaw_interp[cruise_mask] - gcrs[cruise_mask])
    spd_c = speeds[cruise_mask]

    # Crosswind component: positive = wind from left of aircraft
    crosswind = spd_c * np.sin(np.radians(crab))
    mean_crab = float(np.mean(crab))
    mean_cw   = float(np.mean(crosswind))
    std_cw    = float(np.std(crosswind))
    approx_ws = float(np.mean(np.abs(crosswind)))   # magnitude as speed proxy

    metrics = {
        "wind_available":       True,
        "wind_method":          "crab angle (GPS track vs heading)",
        "wind_speed_ms":        round(approx_ws, 1),
        "wind_speed_std_ms":    round(std_cw, 1),
        "wind_crosswind_ms":    round(mean_cw, 1),
        "wind_crab_angle_deg":  round(mean_crab, 1),
    }
    return metrics, _build_wind_issues(metrics)


def _wind_from_hover_tilt(
    att_df:   pd.DataFrame,
    gps_df:   Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
) -> Tuple[Dict, List[Dict]]:
    """Approximate wind from mean roll/pitch lean during near-hover."""
    att_seg = att_df[
        (att_df["timestamp"] >= arm_time) &
        (att_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    roll_col  = _find_col(att_seg, ["Roll",  "roll"])
    pitch_col = _find_col(att_seg, ["Pitch", "pitch"])
    if roll_col is None or pitch_col is None or len(att_seg) < 20:
        return {"wind_available": False, "wind_method": "hover_tilt"}, []

    # Build low-speed hover mask using GPS ground speed if available
    hover_mask: Optional[np.ndarray] = None
    if gps_df is not None:
        spd_col = _find_col(gps_df, ["Spd", "spd"])
        if spd_col:
            gps_seg = gps_df[
                (gps_df["timestamp"] >= arm_time) &
                (gps_df["timestamp"] <= end_time)
            ].copy()
            if len(gps_seg) >= 5:
                spd_interp = np.interp(
                    att_seg["timestamp"].values,
                    gps_seg["timestamp"].values,
                    gps_seg[spd_col].values.astype(float),
                )
                hover_mask = spd_interp < 1.5   # near-stationary

    rolls  = att_seg[roll_col].values.astype(float)
    pitches= att_seg[pitch_col].values.astype(float)

    if hover_mask is not None and np.sum(hover_mask) >= 10:
        rolls   = rolls[hover_mask]
        pitches = pitches[hover_mask]

    if len(rolls) < 10:
        return {"wind_available": False, "wind_method": "hover_tilt"}, []

    mean_roll  = float(np.mean(rolls))
    mean_pitch = float(np.mean(pitches))
    tilt_deg   = math.sqrt(mean_roll ** 2 + mean_pitch ** 2)
    # Rough approximation: ~1° lean ≈ 0.5 m/s wind (varies with airframe weight/drag)
    est_ws = tilt_deg * 0.5

    metrics = {
        "wind_available":       True,
        "wind_method":          "hover tilt (approximate)",
        "wind_speed_ms":        round(est_ws, 1),
        "wind_speed_std_ms":    None,
        "hover_mean_roll_deg":  round(mean_roll, 1),
        "hover_mean_pitch_deg": round(mean_pitch, 1),
        "hover_tilt_deg":       round(tilt_deg, 1),
    }
    return metrics, _build_wind_issues(metrics)


def _build_wind_issues(m: Dict) -> List[Dict]:
    """Build WND issues from wind metrics dict."""
    issues: List[Dict] = []
    ws  = m.get("wind_speed_ms", 0.0) or 0.0
    std = m.get("wind_speed_std_ms") or 0.0
    wdir= m.get("wind_direction_deg")

    dir_str = f" from {wdir:.0f}°" if wdir is not None else ""
    method  = m.get("wind_method", "")

    issues.append(_issue(
        "info", "WND-001",
        f"Estimated wind speed: {ws:.1f} m/s{dir_str} [{method}].",
        value=round(ws, 1),
    ))

    if ws > STRONG_WIND_MS:
        issues.append(_issue(
            "warning", "WND-002",
            f"Strong wind detected: {ws:.1f} m/s (threshold {STRONG_WIND_MS:.0f} m/s). "
            "Verify aircraft is rated for these wind conditions and check motor/ESC temperature.",
            value=round(ws, 1),
            threshold=STRONG_WIND_MS,
        ))

    if std > GUSTY_WIND_STD_MS:
        issues.append(_issue(
            "warning", "WND-003",
            f"Gusty conditions: wind variability ±{std:.1f} m/s "
            f"(threshold ±{GUSTY_WIND_STD_MS:.1f} m/s). "
            "Wind gusts may have caused attitude disturbances during flight.",
            value=round(std, 1),
            threshold=GUSTY_WIND_STD_MS,
        ))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict]) -> float:
    codes = {iss["code"] for iss in issues}

    # If landing not detected: no landing deductions — wind only
    if not metrics.get("landing_detected", False):
        score = 100.0
        if "WND-002" in codes: score -= 5
        if "WND-003" in codes: score -= 5
        return max(0.0, min(100.0, score))

    score = 100.0
    if   "LND-001" in codes: score -= 20
    elif "LND-002" in codes: score -= 10
    if   "LND-003" in codes: score -= 10
    if   "LND-004" in codes: score -= 5
    if   "LND-005" in codes: score -= 5
    if   "WND-002" in codes: score -= 5
    if   "WND-003" in codes: score -= 5

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle array to ±180°."""
    return ((np.asarray(angle) + 180.0) % 360.0) - 180.0


def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _issue(
    severity:  str,
    code:      str,
    message:   str,
    value:     Any = None,
    threshold: Any = None,
    timestamp: Optional[float] = None,
) -> Dict:
    d: Dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if value     is not None: d["value"]       = value
    if threshold is not None: d["threshold"]   = threshold
    if timestamp is not None: d["timestamp_s"] = timestamp
    return d
