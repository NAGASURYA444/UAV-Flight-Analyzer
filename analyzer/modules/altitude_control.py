"""
Altitude Control Quality Module
================================
Phase 4.8 — Altitude tracking accuracy from CTUN / NTUN messages.

Data sources
-------------
ArduCopter (multirotor):
  CTUN — DAlt (desired alt, m), Alt (actual alt, m), DCRt (desired climb rate),
          CRt (actual climb rate).  Climb rates are in cm/s on older firmware
          and m/s on 4.3+; auto-detected by magnitude.

ArduPlane / QuadPlane (fixed_wing / vtol):
  NTUN — AltErr (altitude error m, positive = below target), TAlt (target alt).
          Altitude error is provided directly; no separate DAlt/Alt needed.
          Climb-rate tracking not available from NTUN — skipped for plane types.

Checks performed
-----------------
CTN-001  Altitude tracking error RMS
CTN-002  Climb-rate tracking error RMS  (multirotor only)
CTN-003  Peak altitude deviation
CTN-004  Altitude hold stability — std dev during hover/loiter modes

Returns available=False (score=100/A) when required data is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd

from analyzer.parser.bin_parser import ParseResult
from analyzer.config.drone_profile import DroneProfile

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_ALT_TRACK_WARN: Dict[str, float] = {"multirotor": 2.0,  "fixed_wing": 5.0,  "vtol": 5.0}
_ALT_TRACK_CRIT: Dict[str, float] = {"multirotor": 5.0,  "fixed_wing": 15.0, "vtol": 15.0}

_CRT_TRACK_WARN: Dict[str, float] = {"multirotor": 0.5,  "fixed_wing": 1.0,  "vtol": 1.0}
_CRT_TRACK_CRIT: Dict[str, float] = {"multirotor": 1.5,  "fixed_wing": 2.5,  "vtol": 2.5}

_ALT_PEAK_WARN:  Dict[str, float] = {"multirotor": 5.0,  "fixed_wing": 15.0, "vtol": 30.0}
_ALT_PEAK_CRIT:  Dict[str, float] = {"multirotor": 10.0, "fixed_wing": 25.0, "vtol": 60.0}
# VTOL survey missions fly between waypoints at varying altitudes.  TECS altitude
# errors during commanded steps (climb between legs) are expected and can reach
# 30-60 m before the controller converges.  Use higher thresholds so that only
# genuinely sustained or extreme deviations trigger CRITICAL for VTOL.

_HOLD_STD_WARN:  Dict[str, float] = {"multirotor": 1.0,  "fixed_wing": 2.0,  "vtol": 3.0}
_HOLD_STD_CRIT:  Dict[str, float] = {"multirotor": 3.0,  "fixed_wing": 6.0,  "vtol": 8.0}

# Minimum absolute outlier floor for plane/VTOL trimming.
# VTOL initial-cruise-climb after transition produces ~40-50 m spikes that are
# intentional (commanded altitude step), not controller failures.  A slightly
# tighter floor (40 m) trims these while still exposing genuine sustained errors.
# Fixed-wing uses 50 m — a commanded waypoint altitude change is typically larger.
_OUTLIER_FLOOR: Dict[str, float] = {"fixed_wing": 50.0, "vtol": 40.0}

# Seconds to exclude from the start of the armed window for multirotor.
# Initial climb from ground to first target altitude creates a natural altitude
# error transient that is not representative of in-flight altitude-hold quality.
_COPTER_STARTUP_EXCL_S = 30.0

# Seconds to skip at the start of EACH tracking-mode segment for fixed-wing / VTOL.
# Each time AUTO (or LOITER/CRUISE) mode is entered, NTUN.AltErr immediately
# reflects the gap between the commanded waypoint altitude and the current aircraft
# altitude.  TECS needs 30-60 s to converge; samples within that window are
# settling transients, not representative of steady-state tracking quality.
# Using a per-segment exclusion (rather than a one-time startup exclusion) is
# necessary because fixed-wing flights often switch between AUTO and MANUAL many
# times — every re-entry into AUTO resets the settling clock.
_FW_SEGMENT_EXCL_S: Dict[str, float] = {"fixed_wing": 30.0, "vtol": 60.0}

_MIN_SAMPLES = 30
_CRT_CMS_THRESHOLD = 50.0   # if peak |CRt| > this, data is in cm/s

_HOLD_MODES_COPTER: Set[str] = {
    "ALT_HOLD", "LOITER", "POSHOLD", "AUTO", "GUIDED", "BRAKE",
}
# Modes where altitude should be actively held (used for hold stability check)
_HOLD_MODES_PLANE: Set[str] = {
    "AUTO", "LOITER", "CRUISE", "QHOVER", "QLOITER",
    "FLY_BY_WIRE_B", "GUIDED",
}
# Modes used for altitude TRACKING analysis (exclude RTL/descent — intentional deviations)
_TRACK_MODES_PLANE: Set[str] = {
    "AUTO", "LOITER", "CRUISE", "QHOVER", "QLOITER",
    "FLY_BY_WIRE_B", "GUIDED",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _issue(severity: str, code: str, message: str,
           value: Optional[float] = None,
           threshold: Optional[float] = None,
           timestamp: Optional[float] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if value is not None:
        d["value"] = round(float(value), 3)
    if threshold is not None:
        d["threshold"] = round(float(threshold), 3)
    if timestamp is not None:
        d["timestamp_s"] = round(float(timestamp), 1)
    return d


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
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _scope(df: pd.DataFrame,
           arm_time: Optional[float],
           disarm_time: Optional[float]) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    if arm_time is not None:
        df = df[df["timestamp"] >= arm_time]
    if disarm_time is not None:
        df = df[df["timestamp"] <= disarm_time]
    return df.reset_index(drop=True)


def _mode_mask(ts_arr: np.ndarray,
               mode_df: Optional[pd.DataFrame],
               hold_modes: Set[str]) -> np.ndarray:
    """Return boolean array indicating samples in hold-mode phases."""
    if mode_df is None or mode_df.empty or "mode_name" not in mode_df.columns:
        return np.zeros(len(ts_arr), dtype=bool)
    if "timestamp" not in mode_df.columns:
        return np.zeros(len(ts_arr), dtype=bool)

    mode_ts    = mode_df["timestamp"].values
    mode_names = mode_df["mode_name"].values

    def _mode_at(t: float) -> str:
        idx = int(np.searchsorted(mode_ts, t, side="right")) - 1
        if idx < 0:
            return ""
        return str(mode_names[idx])

    return np.array([_mode_at(t) in hold_modes for t in ts_arr])


def _build_settled_mask(ts_arr: np.ndarray,
                        mode_df: Optional[pd.DataFrame],
                        track_modes: Set[str],
                        excl_s: float) -> np.ndarray:
    """
    Return boolean mask (same shape as ts_arr): True for samples that are at
    least *excl_s* seconds into a continuous tracking-mode segment.

    Each time the flight mode transitions INTO one of *track_modes*, the first
    *excl_s* seconds of that segment are excluded.  This removes the initial
    TECS / altitude-controller settling transient that occurs after any
    commanded altitude step-change (mode entry, waypoint altitude change, or
    VTOL→FW transition), so that only steady-state tracking data is analysed.
    """
    if excl_s <= 0.0 or mode_df is None or mode_df.empty \
            or "mode_name" not in mode_df.columns \
            or "timestamp" not in mode_df.columns:
        return np.ones(len(ts_arr), dtype=bool)

    mode_ts    = mode_df["timestamp"].values.astype(float)
    mode_names = mode_df["mode_name"].values

    # Collect the timestamp of each entry INTO a tracking mode
    entry_times: list = []
    prev_in = False
    for t, name in zip(mode_ts, mode_names):
        cur_in = str(name) in track_modes
        if cur_in and not prev_in:
            entry_times.append(float(t))
        prev_in = cur_in

    if not entry_times:
        return np.ones(len(ts_arr), dtype=bool)

    entry_arr = np.array(entry_times, dtype=float)
    result    = np.zeros(len(ts_arr), dtype=bool)

    for i, t in enumerate(ts_arr):
        # Most-recent tracking-mode entry at or before this sample
        idx = int(np.searchsorted(entry_arr, t, side="right")) - 1
        if idx >= 0 and (t - entry_arr[idx]) >= excl_s:
            result[i] = True

    return result


def _analyse_errors(alt_err: np.ndarray,
                    ts: np.ndarray,
                    drone_type: str,
                    issues: list,
                    metrics: dict,
                    score: float) -> float:
    """Shared CTN-001 and CTN-003 logic for any error array.

    For fixed-wing/VTOL the RMS is computed on a trimmed array that excludes
    outliers larger than 5× the median absolute error (or > 50 m).  This
    removes transition-climb spikes (intentional large altitude steps during
    VTOL↔FW transitions or initial climbs to waypoint altitude) while keeping
    genuine sustained tracking errors.  The raw peak is still reported for CTN-003.
    """
    is_plane = drone_type in ("fixed_wing", "vtol")

    # Trimmed RMS for plane/VTOL — exclude transition-climb outliers.
    # The same trimmed array is used for BOTH the RMS (CTN-001) and peak (CTN-003)
    # so that transition-climb spikes excluded from the RMS are also excluded from
    # the peak.  Without this, a VTOL initial-climb spike (e.g. 130 m) that is
    # correctly absent from the RMS would still trigger a false CTN-003 critical.
    if is_plane and len(alt_err) > 0:
        median_abs  = float(np.nanmedian(np.abs(alt_err)))
        floor       = _OUTLIER_FLOOR.get(drone_type, 50.0)
        outlier_thr = max(5.0 * median_abs, floor)
        trim_mask   = np.abs(alt_err) <= outlier_thr
        if trim_mask.sum() >= _MIN_SAMPLES:
            rms_arr    = alt_err[trim_mask]
            ts_trimmed = ts[trim_mask]          # keep timestamps aligned
        else:
            rms_arr    = alt_err
            ts_trimmed = ts
        metrics["alt_tracking_outlier_thr_m"]    = round(outlier_thr, 1)
        metrics["alt_tracking_trimmed_samples"]  = int(trim_mask.sum())
    else:
        rms_arr    = alt_err
        ts_trimmed = ts

    alt_rms  = float(np.sqrt(np.nanmean(rms_arr ** 2)))
    warn_thr = _ALT_TRACK_WARN[drone_type]
    crit_thr = _ALT_TRACK_CRIT[drone_type]

    metrics["alt_tracking_rms_m"]   = round(alt_rms, 2)
    metrics["alt_warn_threshold_m"] = warn_thr
    metrics["alt_crit_threshold_m"] = crit_thr

    if alt_rms >= crit_thr:
        issues.append(_issue(
            "critical", "CTN-001",
            f"Altitude tracking critical: RMS error {alt_rms:.1f} m (threshold {crit_thr:.0f} m). "
            "The flight controller is consistently failing to hold commanded altitude. "
            "Check altimeter calibration, baro shielding, and altitude controller gains "
            "(PSC_ACCZ_P/I for copter, TECS gains for fixed-wing).",
            value=alt_rms, threshold=crit_thr,
        ))
        score -= 25.0
    elif alt_rms >= warn_thr:
        issues.append(_issue(
            "warning", "CTN-001",
            f"Altitude tracking elevated: RMS error {alt_rms:.1f} m (threshold {warn_thr:.0f} m). "
            "Check barometer shielding from prop wash, verify baro calibration, "
            "and review altitude controller tuning.",
            value=alt_rms, threshold=warn_thr,
        ))
        score -= 10.0

    # CTN-003 peak deviation — use same trimmed data as RMS so that
    # transition-climb spikes (excluded from RMS) are also excluded from peak.
    peak_idx = int(np.nanargmax(np.abs(rms_arr)))
    peak_err = float(np.abs(rms_arr[peak_idx]))
    peak_ts  = float(ts_trimmed[peak_idx])
    p_warn   = _ALT_PEAK_WARN[drone_type]
    p_crit   = _ALT_PEAK_CRIT[drone_type]

    metrics["alt_peak_error_m"] = round(peak_err, 1)
    metrics["alt_peak_ts"]      = round(peak_ts, 1)

    if peak_err >= p_crit:
        issues.append(_issue(
            "critical", "CTN-003",
            f"Peak altitude deviation {peak_err:.1f} m at T+{peak_ts:.1f}s (threshold {p_crit:.0f} m). "
            "Large altitude excursion — check for wind gust, sensor glitch, "
            "or commanded altitude step-change at that timestamp.",
            value=peak_err, threshold=p_crit, timestamp=peak_ts,
        ))
        score -= 15.0
    elif peak_err >= p_warn:
        issues.append(_issue(
            "warning", "CTN-003",
            f"Peak altitude deviation {peak_err:.1f} m at T+{peak_ts:.1f}s (threshold {p_warn:.0f} m). "
            "Altitude excursion detected — review timestamp for wind, mode change, or sensor event.",
            value=peak_err, threshold=p_warn, timestamp=peak_ts,
        ))
        score -= 5.0

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    parse_result: ParseResult,
    drone_profile: DroneProfile,
    arm_time: Optional[float] = None,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyse CTUN/NTUN altitude controller performance.
    Returns: score, grade, available, issues, metrics, summary
    """
    drone_type = getattr(drone_profile, "type", "multirotor")
    is_plane   = drone_type in ("fixed_wing", "vtol")

    mode_df = parse_result.get("MODE")
    hold_modes = _HOLD_MODES_PLANE if is_plane else _HOLD_MODES_COPTER

    issues: list  = []
    metrics: dict = {}
    score = 100.0

    # ── ArduPlane / QuadPlane — use NTUN.AltErr ───────────────────────────────
    if is_plane:
        ntun_raw = parse_result.get("NTUN")
        if ntun_raw is None or ntun_raw.empty:
            return _unavailable(
                "NTUN messages not present in log. "
                "Altitude error data unavailable for fixed-wing/VTOL analysis."
            )

        ntun = _scope(ntun_raw, arm_time, disarm_time)
        if len(ntun) < _MIN_SAMPLES:
            return _unavailable(
                f"Insufficient NTUN samples in armed window ({len(ntun)} < {_MIN_SAMPLES})."
            )

        # ArduPlane column name varies by firmware version
        alt_err_col = None
        for candidate in ("AltErr", "AltE"):
            if candidate in ntun.columns:
                alt_err_col = candidate
                break
        if alt_err_col is None:
            return _unavailable(
                "NTUN does not contain AltErr/AltE column — "
                "altitude error data unavailable in this firmware version."
            )

        alt_err_all = ntun[alt_err_col].values.astype(float)
        ts_all      = ntun["timestamp"].values if "timestamp" in ntun.columns else np.arange(len(ntun))

        # Scope tracking analysis to mission-flight modes only (AUTO, LOITER, CRUISE…)
        # RTL / QRTL / QLAND have intentionally large altitude deviations — exclude them.
        track_mask = _mode_mask(ts_all, mode_df, _TRACK_MODES_PLANE)

        # Per-segment settling exclusion.
        # Each entry into a tracking mode starts with a TECS settling transient
        # (commanded altitude step-change).  Exclude the first N seconds of every
        # segment so that only steady-state tracking data is used for the RMS.
        excl_s        = _FW_SEGMENT_EXCL_S.get(drone_type, 30.0)
        settled_mask  = _build_settled_mask(ts_all, mode_df, _TRACK_MODES_PLANE, excl_s)
        final_mask    = track_mask & settled_mask

        if final_mask.sum() >= _MIN_SAMPLES:
            alt_err = alt_err_all[final_mask]
            ts      = ts_all[final_mask]
            metrics["tracking_mode_filtered"] = True
            metrics["segment_excl_s"]         = excl_s
        elif track_mask.sum() >= _MIN_SAMPLES:
            # All tracking-mode segments were too short to yield settled data
            # (e.g. many brief AUTO bursts with no segment > excl_s).  Fall back
            # to unfiltered tracking samples with a warning flag — the RMS will
            # be inflated by mode-entry transients but is the best available data.
            alt_err = alt_err_all[track_mask]
            ts      = ts_all[track_mask]
            metrics["tracking_mode_filtered"]     = True
            metrics["segment_excl_insufficient"]  = True
            logger.warning(
                "altitude_control: all tracking-mode segments < %.0f s; "
                "using unfiltered tracking data — RMS may be inflated by "
                "mode-entry transients.", excl_s
            )
        else:
            # Not enough tracking-mode samples at all — fall back to all armed samples
            alt_err = alt_err_all
            ts      = ts_all
            metrics["tracking_mode_filtered"] = False

        metrics["samples"]     = len(ntun)
        metrics["data_source"] = "NTUN (ArduPlane navigation tuning)"

        # Target altitude info (column name varies by firmware).
        # Filter physically unreasonable values: pre-arm frames or unit-mismatch
        # artefacts can produce values > 10 000 m; exclude them.
        for talt_col in ("TAlt", "TAW", "TAT"):
            if talt_col in ntun.columns:
                talt = ntun[talt_col].values.astype(float)
                talt = talt[(talt != 0) & (talt > -1000) & (talt < 10000)]
                if len(talt) > 0:
                    metrics["target_alt_mean_m"] = round(float(np.nanmean(talt)), 1)
                break

        score = _analyse_errors(alt_err, ts, drone_type, issues, metrics, score)

        # CTN-004 hold stability.
        # Apply the same per-segment settling exclusion used for the tracking RMS
        # (_HOLD_MODES_PLANE == _TRACK_MODES_PLANE for fixed-wing/VTOL, so
        # *settled_mask* already covers the hold modes — no need to rebuild it).
        in_hold = _mode_mask(ts_all, mode_df, hold_modes)
        in_hold_settled = in_hold & settled_mask
        hold_errs_all = alt_err_all[in_hold_settled]
        if len(hold_errs_all) >= _MIN_SAMPLES:
            # Trim outliers from hold analysis too
            median_hold = float(np.nanmedian(np.abs(hold_errs_all)))
            h_thr = max(5.0 * median_hold, 50.0)
            hold_trim = hold_errs_all[np.abs(hold_errs_all) <= h_thr]
            if len(hold_trim) >= _MIN_SAMPLES:
                hold_std = float(np.nanstd(hold_trim))
            else:
                hold_std = float(np.nanstd(hold_errs_all))
            h_warn = _HOLD_STD_WARN[drone_type]
            h_crit = _HOLD_STD_CRIT[drone_type]
            metrics["alt_hold_std_m"] = round(hold_std, 2)
            metrics["hold_samples"]   = int(in_hold_settled.sum())
            if hold_std >= h_crit:
                issues.append(_issue(
                    "critical", "CTN-004",
                    f"Altitude hold unstable: std dev {hold_std:.1f} m during hold modes "
                    f"(threshold {h_crit:.0f} m). Significant altitude oscillation — "
                    "check TECS gains and barometer isolation.",
                    value=hold_std, threshold=h_crit,
                ))
                score -= 10.0
            elif hold_std >= h_warn:
                issues.append(_issue(
                    "warning", "CTN-004",
                    f"Altitude hold slightly unstable: std dev {hold_std:.1f} m during hold modes "
                    f"(threshold {h_warn:.0f} m). Review TECS altitude hold gains.",
                    value=hold_std, threshold=h_warn,
                ))
                score -= 5.0
        else:
            metrics["alt_hold_std_m"] = None
            metrics["hold_samples"]   = int(in_hold_settled.sum())

    # ── ArduCopter — use CTUN.DAlt / CTUN.Alt ────────────────────────────────
    else:
        ctun_raw = parse_result.get("CTUN")
        if ctun_raw is None or ctun_raw.empty:
            return _unavailable(
                "CTUN messages not present in log. "
                "Check LOG_BITMASK includes CTUN logging (enabled by default in most builds)."
            )

        ctun = _scope(ctun_raw, arm_time, disarm_time)
        if len(ctun) < _MIN_SAMPLES:
            return _unavailable(
                f"Insufficient CTUN samples in armed window ({len(ctun)} < {_MIN_SAMPLES})."
            )

        cols = set(ctun.columns)
        if "DAlt" not in cols or "Alt" not in cols:
            return _unavailable(
                "CTUN does not contain DAlt/Alt columns — "
                "altitude tracking analysis unavailable in this firmware version."
            )

        dalt    = ctun["DAlt"].values.astype(float)
        alt     = ctun["Alt"].values.astype(float)
        ts      = ctun["timestamp"].values if "timestamp" in cols else np.arange(len(ctun))
        alt_err = alt - dalt

        # Exclude the initial takeoff transient window.
        # During the first ~30 s after arming the drone climbs from ground to its
        # first target altitude.  CTUN.DAlt is already at the target while Alt is
        # still rising, producing a natural altitude error spike that does NOT
        # reflect in-flight altitude-hold quality.
        if arm_time is not None:
            startup_end  = arm_time + _COPTER_STARTUP_EXCL_S
            startup_mask = ts > startup_end
            if startup_mask.sum() >= _MIN_SAMPLES:
                alt_err  = alt_err[startup_mask]
                ts       = ts[startup_mask]
                metrics["startup_excluded_s"] = _COPTER_STARTUP_EXCL_S

        metrics["samples"]      = len(ctun)
        metrics["data_source"]  = "CTUN (ArduCopter altitude controller)"
        metrics["alt_mean_m"]   = round(float(np.nanmean(alt)), 1)
        metrics["dalt_mean_m"]  = round(float(np.nanmean(dalt)), 1)

        score = _analyse_errors(alt_err, ts, drone_type, issues, metrics, score)

        # CTN-002 climb-rate tracking
        if "DCRt" in cols and "CRt" in cols:
            dcrt_s = ctun["DCRt"].copy()
            crt_s  = ctun["CRt"].copy()
            peak   = float(np.nanmax(np.abs(crt_s.values)))
            if peak > _CRT_CMS_THRESHOLD:
                dcrt_s = dcrt_s / 100.0
                crt_s  = crt_s  / 100.0
            crt_err  = (crt_s - dcrt_s).values.astype(float)
            crt_rms  = float(np.sqrt(np.nanmean(crt_err ** 2)))
            crt_warn = _CRT_TRACK_WARN[drone_type]
            crt_crit = _CRT_TRACK_CRIT[drone_type]
            metrics["crt_tracking_rms_ms"] = round(crt_rms, 3)
            metrics["crt_warn_threshold"]  = crt_warn
            metrics["crt_crit_threshold"]  = crt_crit
            if crt_rms >= crt_crit:
                issues.append(_issue(
                    "critical", "CTN-002",
                    f"Climb-rate tracking critical: RMS error {crt_rms:.2f} m/s "
                    f"(threshold {crt_crit:.1f} m/s). "
                    "Vertical velocity controller is significantly lagging. "
                    "Check PSC_VELZ_P and verify no throttle saturation.",
                    value=crt_rms, threshold=crt_crit,
                ))
                score -= 15.0
            elif crt_rms >= crt_warn:
                issues.append(_issue(
                    "warning", "CTN-002",
                    f"Climb-rate tracking elevated: RMS error {crt_rms:.2f} m/s "
                    f"(threshold {crt_warn:.1f} m/s). "
                    "Review PILOT_SPEED_UP/DN and PSC_VELZ_P gain.",
                    value=crt_rms, threshold=crt_warn,
                ))
                score -= 5.0
        else:
            metrics["crt_tracking_rms_ms"] = None

        # CTN-004 hold stability
        in_hold   = _mode_mask(ts, mode_df, hold_modes)
        hold_errs = alt_err[in_hold]
        if len(hold_errs) >= _MIN_SAMPLES:
            hold_std = float(np.nanstd(hold_errs))
            h_warn   = _HOLD_STD_WARN[drone_type]
            h_crit   = _HOLD_STD_CRIT[drone_type]
            metrics["alt_hold_std_m"] = round(hold_std, 2)
            metrics["hold_samples"]   = int(in_hold.sum())
            if hold_std >= h_crit:
                issues.append(_issue(
                    "critical", "CTN-004",
                    f"Altitude hold unstable: std dev {hold_std:.1f} m during hold modes "
                    f"(threshold {h_crit:.0f} m). Significant altitude oscillation — "
                    "check PSC_ACCZ_P/I, barometer isolation, and airframe resonance.",
                    value=hold_std, threshold=h_crit,
                ))
                score -= 10.0
            elif hold_std >= h_warn:
                issues.append(_issue(
                    "warning", "CTN-004",
                    f"Altitude hold slightly unstable: std dev {hold_std:.1f} m during hold modes "
                    f"(threshold {h_warn:.0f} m). Minor oscillation in hover/loiter — "
                    "check baro shielding and PSC_ACCZ tuning.",
                    value=hold_std, threshold=h_warn,
                ))
                score -= 5.0
        else:
            metrics["alt_hold_std_m"] = None
            metrics["hold_samples"]   = int(in_hold.sum())

    # ── Final score ───────────────────────────────────────────────────────────
    score = max(0.0, min(100.0, score))
    grade = _grade(score)

    alt_rms  = metrics.get("alt_tracking_rms_m", 0.0)
    peak_err = metrics.get("alt_peak_error_m", 0.0)

    if not issues:
        summary = (
            f"Altitude control healthy: tracking RMS {alt_rms:.1f} m, "
            f"peak deviation {peak_err:.1f} m."
        )
    else:
        critical_count = sum(1 for i in issues if i["severity"] == "critical")
        warning_count  = sum(1 for i in issues if i["severity"] == "warning")
        summary = (
            f"Altitude control: {critical_count} critical, {warning_count} warning — "
            f"tracking RMS {alt_rms:.1f} m, peak {peak_err:.1f} m."
        )

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }
