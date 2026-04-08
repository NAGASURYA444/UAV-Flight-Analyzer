"""
Vibration Analysis Module — Phase 2
=====================================
Analyses VIBE messages and IMU accelerometer data to assess mechanical health.

Two layers of analysis
-----------------------
1. VIBE-based  (always available if VIBE messages exist)
   - Mean / peak vibration per axis (X, Y, Z)  in m/s²
   - Accelerometer clipping counts
   - Vibration trend  (first-half vs second-half of flight)

2. IMU-based FFT  (requires IMU messages, depends on log rate)
   - Dominant frequency identification
   - Frequency band labelling:
       < 2 Hz   — slow oscillation / attitude wobble
       2–5 Hz   — PID rate oscillation
       5–15 Hz  — frame / structural resonance
       15–50 Hz — motor / propeller harmonics
       > 50 Hz  — ESC switching noise / EMI
   - Note: frequency ceiling = sample_rate / 2 (Nyquist).
     Standard ArduPilot logs at 25–50 Hz → max detectable ≈ 25 Hz.

Scoring deductions (from 100)
------------------------------
-10 to -20   Mean vibration > warn threshold
-15 to -25   Mean vibration > critical threshold
-10          Any clipping events (per IMU)
-10          Vibration trend increasing > 20% (mechanical wear indicator)
-5           FFT shows dominant PID oscillation band
-10          FFT shows strong resonance in structural band
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)


def analyse(result: ParseResult, profile: DroneProfile, arm_time: float = 0.0) -> Dict[str, Any]:
    """Run vibration analysis. Returns standardised result dict."""
    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    vibe_df = result.get("VIBE")
    if vibe_df is None or len(vibe_df) < 10:
        return {
            "score": 50.0, "grade": "C", "available": False,
            "issues": [_issue("info", "VIB-000",
                              "No VIBE messages found in log — vibration data unavailable.")],
            "metrics": {},
            "summary": "Vibration data not available.",
        }

    # Restrict to armed / airborne window
    flight_vibe = vibe_df[vibe_df["timestamp"] >= arm_time].copy()
    if len(flight_vibe) < 10:
        flight_vibe = vibe_df.copy()

    # ── 1. VIBE-based analysis ────────────────────────────────────────────────
    vibe_result = _vibe_level_analysis(flight_vibe, profile)
    metrics.update(vibe_result["metrics"])
    issues.extend(vibe_result["issues"])

    # ── 2. Clipping ───────────────────────────────────────────────────────────
    clip_result = _clipping_analysis(flight_vibe, result.duration_s)
    metrics.update(clip_result["metrics"])
    issues.extend(clip_result["issues"])

    # ── 3. Vibration trend ────────────────────────────────────────────────────
    trend_result = _trend_analysis(flight_vibe)
    metrics.update(trend_result["metrics"])
    issues.extend(trend_result["issues"])

    # ── 4. IMU FFT analysis ───────────────────────────────────────────────────
    imu_df = result.get("IMU")
    if imu_df is not None and len(imu_df) > 100:
        fft_result = _fft_analysis(imu_df, arm_time)
        metrics.update(fft_result["metrics"])
        issues.extend(fft_result["issues"])
        if not metrics.get("fft_available"):
            # Record which columns were actually present to aid diagnostics
            metrics["imu_columns_found"] = list(imu_df.columns)
    else:
        metrics["fft_available"] = False
        metrics["fft_unavailable_reason"] = (
            "No IMU messages in log" if imu_df is None
            else f"Too few IMU samples ({len(imu_df)})"
        )

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues, profile)
    grade = _grade(score)

    vx = metrics.get("mean_vibe_x", 0)
    vy = metrics.get("mean_vibe_y", 0)
    vz = metrics.get("mean_vibe_z", 0)
    clips = metrics.get("total_clipping_events", 0)
    summary = (
        f"Vibe X/Y/Z: {vx:.1f} / {vy:.1f} / {vz:.1f} m/s²  |  "
        f"Clipping: {clips}  |  "
        f"Trend: {metrics.get('vibration_trend', 'N/A')}"
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
# VIBE level analysis
# ─────────────────────────────────────────────────────────────────────────────

def _vibe_level_analysis(vibe_df: pd.DataFrame, profile: DroneProfile) -> Dict:
    issues: List[Dict] = []
    metrics: Dict = {}

    warn_thresh = profile.vibration.warn_threshold
    crit_thresh = profile.vibration.critical_threshold

    for axis in ["X", "Y", "Z"]:
        col = f"Vibe{axis}"
        if col not in vibe_df.columns:
            continue
        vals = vibe_df[col].dropna().values
        if len(vals) == 0:
            continue
        mean_v = float(np.mean(vals))
        peak_v = float(np.percentile(vals, 99))
        metrics[f"mean_vibe_{axis.lower()}"] = round(mean_v, 3)
        metrics[f"peak_vibe_{axis.lower()}"] = round(peak_v, 3)

    # Use the worst axis (usually Z for multirotors) for threshold checks
    mean_vals = [metrics.get(f"mean_vibe_{ax}", 0.0) for ax in ["x", "y", "z"]]
    peak_vals = [metrics.get(f"peak_vibe_{ax}", 0.0) for ax in ["x", "y", "z"]]
    worst_mean = max(mean_vals) if mean_vals else 0.0
    worst_peak = max(peak_vals) if peak_vals else 0.0

    metrics["worst_axis_mean"] = round(worst_mean, 3)
    metrics["worst_axis_peak"] = round(worst_peak, 3)
    metrics["warn_threshold"] = warn_thresh
    metrics["critical_threshold"] = crit_thresh

    if worst_mean >= crit_thresh:
        issues.append(_issue(
            "critical", "VIB-001",
            f"Critical vibration level: {worst_mean:.1f} m/s² mean "
            f"(threshold: {crit_thresh:.0f} m/s²). "
            "Severe mechanical issue — inspect props, motors, and mounts immediately.",
            value=round(worst_mean, 2), threshold=crit_thresh,
        ))
    elif worst_mean >= warn_thresh:
        issues.append(_issue(
            "warning", "VIB-001",
            f"Elevated vibration: {worst_mean:.1f} m/s² mean "
            f"(threshold: {warn_thresh:.0f} m/s²). "
            "Check propeller balance and motor mount tightness.",
            value=round(worst_mean, 2), threshold=warn_thresh,
        ))

    if worst_peak >= crit_thresh * 1.5:
        issues.append(_issue(
            "warning", "VIB-002",
            f"High vibration peaks detected: {worst_peak:.1f} m/s² "
            "(99th percentile). Intermittent mechanical events observed.",
            value=round(worst_peak, 2), threshold=round(crit_thresh * 1.5, 1),
        ))

    return {"issues": issues, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Clipping analysis
# ─────────────────────────────────────────────────────────────────────────────

def _clipping_analysis(vibe_df: pd.DataFrame, flight_duration_s: float) -> Dict:
    """
    Accelerometer clipping = hard vibration that exceeded the sensor range.
    Any sustained clipping is a serious concern.
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    total_clips = 0
    for col in ["Clip0", "Clip1", "Clip2"]:
        if col not in vibe_df.columns:
            continue
        clip_series = vibe_df[col].dropna()
        if len(clip_series) > 1:
            # Count increments (clips are cumulative counters)
            clip_count = int(max(0, clip_series.iloc[-1] - clip_series.iloc[0]))
            metrics[f"{col.lower()}_events"] = clip_count
            total_clips += clip_count

    metrics["total_clipping_events"] = total_clips
    flight_min = flight_duration_s / 60.0
    clips_per_min = total_clips / flight_min if flight_min > 0 else 0.0
    metrics["clipping_per_min"] = round(clips_per_min, 2)

    if total_clips > 0:
        sev = "critical" if clips_per_min > 10 else "warning"
        issues.append(_issue(
            sev, "VIB-003",
            f"Accelerometer clipping: {total_clips} events "
            f"({clips_per_min:.1f}/min). "
            "Vibration exceeded sensor range — severe mechanical issue. "
            "Check for loose props, motor bearing failure, or frame crack.",
            value=total_clips, threshold=0,
        ))

    return {"issues": issues, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Vibration trend
# ─────────────────────────────────────────────────────────────────────────────

def _trend_analysis(vibe_df: pd.DataFrame) -> Dict:
    """
    Compare vibration in the first half vs second half of flight.
    Increasing vibration = bearing wear / prop damage developing during flight.
    """
    issues: List[Dict] = []
    metrics: Dict = {}

    if len(vibe_df) < 20:
        metrics["vibration_trend"] = "insufficient_data"
        return {"issues": issues, "metrics": metrics}

    mid = len(vibe_df) // 2

    # Use Z-axis as primary (most sensitive for multirotors)
    col = "VibeZ" if "VibeZ" in vibe_df.columns else (
          "VibeX" if "VibeX" in vibe_df.columns else None)

    if col is None:
        metrics["vibration_trend"] = "no_data"
        return {"issues": issues, "metrics": metrics}

    first_half_mean = float(vibe_df[col].iloc[:mid].mean())
    second_half_mean = float(vibe_df[col].iloc[mid:].mean())

    if first_half_mean > 0:
        trend_pct = (second_half_mean - first_half_mean) / first_half_mean * 100.0
    else:
        trend_pct = 0.0

    metrics["vibration_trend_pct"] = round(trend_pct, 1)
    metrics["first_half_mean"] = round(first_half_mean, 3)
    metrics["second_half_mean"] = round(second_half_mean, 3)

    if trend_pct > 50:
        metrics["vibration_trend"] = "rapidly_increasing"
        issues.append(_issue(
            "critical", "VIB-004",
            f"Vibration increased {trend_pct:.0f}% from first to second half of flight. "
            "Possible prop damage or bearing failure developing mid-flight.",
            value=round(trend_pct, 1), threshold=50,
        ))
    elif trend_pct > 20:
        metrics["vibration_trend"] = "increasing"
        issues.append(_issue(
            "warning", "VIB-004",
            f"Vibration trending up {trend_pct:.0f}% over the course of the flight. "
            "Monitor over next flights — could indicate early bearing wear.",
            value=round(trend_pct, 1), threshold=20,
        ))
    elif trend_pct < -20:
        metrics["vibration_trend"] = "decreasing"
    else:
        metrics["vibration_trend"] = "stable"

    return {"issues": issues, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# IMU FFT analysis
# ─────────────────────────────────────────────────────────────────────────────

def _fft_analysis(imu_df: pd.DataFrame, arm_time: float) -> Dict:
    """
    FFT on IMU accelerometer data to identify dominant vibration frequencies.
    Frequency resolution is limited by the log sample rate.
    """
    from scipy import signal as scipy_signal

    issues: List[Dict] = []
    metrics: Dict = {}

    # ArduPilot logs AccZ (standard) — some firmware versions use different names
    acc_col = None
    for candidate in ["AccZ", "accz", "AZ", "az"]:
        if candidate in imu_df.columns:
            acc_col = candidate
            break

    if acc_col is None or "timestamp" not in imu_df.columns:
        metrics["fft_available"] = False
        metrics["fft_unavailable_reason"] = (
            f"AccZ column not found. Available columns: {list(imu_df.columns)}"
            if acc_col is None else "timestamp column missing"
        )
        return {"issues": issues, "metrics": metrics}

    # Filter to primary IMU only (index 0) — Cube Orange logs IMU0/1/2 together,
    # which would produce zero dt_median and kill the FFT sample rate estimate.
    imu_primary = imu_df[imu_df["I"] == 0] if "I" in imu_df.columns else imu_df

    flight_imu = imu_primary[imu_primary["timestamp"] >= arm_time].copy()
    if len(flight_imu) < 64:
        metrics["fft_available"] = False
        metrics["fft_unavailable_reason"] = f"Too few IMU0 samples after arm ({len(flight_imu)})"
        return {"issues": issues, "metrics": metrics}

    # Estimate sample rate
    ts = flight_imu["timestamp"].values
    dt_median = float(np.median(np.diff(ts)))
    if dt_median <= 0:
        metrics["fft_available"] = False
        metrics["fft_unavailable_reason"] = "Zero sample interval (duplicate timestamps)"
        return {"issues": issues, "metrics": metrics}

    sample_rate = 1.0 / dt_median
    nyquist = sample_rate / 2.0
    metrics["fft_sample_rate_hz"] = round(sample_rate, 1)
    metrics["fft_nyquist_hz"] = round(nyquist, 1)

    acc_z = flight_imu[acc_col].values.astype(float)

    # Remove DC offset and detrend
    acc_z = acc_z - np.mean(acc_z)
    acc_z = acc_z - np.polyval(np.polyfit(np.arange(len(acc_z)), acc_z, 1),
                                np.arange(len(acc_z)))

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(len(acc_z))
    acc_windowed = acc_z * window

    # FFT
    fft_mag = np.abs(np.fft.rfft(acc_windowed))
    freqs = np.fft.rfftfreq(len(acc_windowed), d=dt_median)

    # Find peaks — restrict to >= 1 Hz to avoid near-DC attitude oscillation artifacts
    # (slow flight manoeuvres create large low-frequency components even after detrend)
    try:
        search_mask = freqs >= 1.0
        search_fft = fft_mag.copy()
        search_fft[~search_mask] = 0.0
        search_max = float(np.max(search_fft)) if np.any(search_mask) else 0.0
        peak_indices, _ = scipy_signal.find_peaks(
            search_fft, height=search_max * 0.10, distance=3
        )
        peak_freqs = freqs[peak_indices]
        peak_mags = fft_mag[peak_indices]
        sorted_peaks = sorted(zip(peak_freqs.tolist(), peak_mags.tolist()),
                               key=lambda x: -x[1])
        top_peaks = sorted_peaks[:5]
    except Exception:
        top_peaks = []

    dominant_frequencies = [
        {
            "freq_hz": round(f, 2),
            "label": _label_frequency(f),
            "relative_magnitude": round(m / (fft_mag.max() + 1e-10), 3),
        }
        for f, m in top_peaks
    ]

    metrics["fft_available"] = True
    metrics["dominant_frequencies"] = dominant_frequencies

    # Check for PID oscillation band (2–5 Hz)
    pid_energy = float(np.sum(fft_mag[(freqs >= 2.0) & (freqs <= 5.0)]))
    total_energy = float(np.sum(fft_mag[freqs > 0.5]))
    pid_fraction = pid_energy / (total_energy + 1e-10)
    metrics["pid_band_energy_fraction"] = round(pid_fraction, 3)

    if pid_fraction > 0.35:
        issues.append(_issue(
            "warning", "VIB-005",
            f"FFT shows {pid_fraction * 100:.0f}% of vibration energy in PID band (2–5 Hz). "
            "Possible PID rate gain too high — consider reducing Rate P/D gains.",
            value=round(pid_fraction * 100, 1), threshold=35,
        ))

    # Check structural resonance band (5–15 Hz, capped at Nyquist)
    struct_upper = min(15.0, nyquist)
    struct_energy = float(np.sum(fft_mag[(freqs >= 5.0) & (freqs <= struct_upper)]))
    struct_fraction = struct_energy / (total_energy + 1e-10)
    metrics["structural_band_energy_fraction"] = round(struct_fraction, 3)

    if struct_fraction > 0.40:
        issues.append(_issue(
            "warning", "VIB-006",
            f"FFT shows {struct_fraction * 100:.0f}% of vibration energy in structural band "
            f"(5–{struct_upper:.0f} Hz). "
            "Possible frame resonance — check arm tightness and add vibration damping.",
            value=round(struct_fraction * 100, 1), threshold=40,
        ))

    return {"issues": issues, "metrics": metrics}


def _label_frequency(freq_hz: float) -> str:
    if freq_hz < 2.0:
        return "Slow oscillation / attitude wobble"
    elif freq_hz < 5.0:
        return "PID rate oscillation"
    elif freq_hz < 15.0:
        return "Frame / structural resonance"
    elif freq_hz < 50.0:
        return "Motor / propeller harmonic"
    else:
        return "ESC noise / high-frequency EMI"


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict], profile: DroneProfile) -> float:
    score = 100.0

    warn_thresh = profile.vibration.warn_threshold
    crit_thresh = profile.vibration.critical_threshold
    worst_mean = metrics.get("worst_axis_mean", 0.0)

    if worst_mean >= crit_thresh:
        score -= 25.0
    elif worst_mean >= warn_thresh:
        score -= min(20.0, (worst_mean - warn_thresh) / (crit_thresh - warn_thresh) * 20.0)

    total_clips = metrics.get("total_clipping_events", 0)
    if total_clips > 0:
        score -= min(20.0, total_clips * 2.0)

    trend_pct = metrics.get("vibration_trend_pct", 0.0)
    if trend_pct > 50:
        score -= 15.0
    elif trend_pct > 20:
        score -= 8.0

    pid_frac = metrics.get("pid_band_energy_fraction", 0.0)
    if pid_frac > 0.35:
        score -= 8.0

    struct_frac = metrics.get("structural_band_energy_fraction", 0.0)
    if struct_frac > 0.40:
        score -= 10.0

    # All vibration issue codes (VIB-001/003/004/005/006) are already captured by
    # the worst_mean, total_clips, trend_pct, pid_frac, and struct_frac deductions
    # above.  An issue-severity loop here would double-penalise those codes.

    return max(0.0, min(100.0, score))


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
