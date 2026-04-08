"""
PID Tuning Quality Assessment Module
======================================
Phase 4.5 — Analyses ArduPilot PID rate controller log messages (PIDR, PIDP,
PIDY) to detect common tuning problems before they cause flight instability or
hardware wear.

Log messages analysed
---------------------
PIDR  — Roll  rate PID  (ArduCopter & ArduPlane)
PIDP  — Pitch rate PID  (ArduCopter & ArduPlane)
PIDY  — Yaw   rate PID  (ArduCopter; ArduPlane if logged)

All PID messages share these key columns (availability varies by firmware):
  Tar / Des  — target/desired rate (same units as Act)
  Act        — actual rate measured by IMU
  Err        — tracking error  (Tar − Act)
  P          — proportional term contribution to output
  I          — integral term contribution to output
  D          — derivative term contribution to output
  FF         — feedforward contribution (optional)
  Out        — total PID output

Checks performed
----------------
1.  I-term dominance   — mean(|I|) / (mean(|P|)+mean(|I|)+mean(|D|)+ε) per axis
    → sustained high ratio means P/D gains are too low or a persistent
      mechanical offset (worn prop, payload imbalance) is present.
2.  D-term noise index — std(D) / (mean(|Out|)+ε) per axis
    → high ratio means the derivative term is amplifying sensor noise;
      notch filter / gyro LP cutoff adjustment is needed.
3.  Normalised rate error — rms(Err) / (rms(Tar)+ε) per axis
    → measures how closely the actual rate tracks the demanded rate;
      high ratio signals poor overall loop performance.
4.  I-term steady bias  — |mean(I)| / (std(I)+ε) per axis
    → integrator consistently non-zero in one direction indicates a trim
      offset (centre-of-gravity, motor/ESC imbalance, compass for yaw).
5.  Roll/Pitch P-gain asymmetry
    → mean(|P|) ratio between roll and pitch axes; large imbalance flags
      asymmetric airframe stiffness or uneven payload.

Scoring (from 100)
------------------
Per axis — roll and pitch scored at full weight; yaw at half-weight:
  −15  I-term dominance critical  (i_ratio > 0.65)
  −10  I-term dominance warning   (i_ratio > 0.45)   [cap −20 across axes]
  −12  D-term noise critical      (d_noise > 0.80)
  − 8  D-term noise warning       (d_noise > 0.40)   [cap −16 across axes]
  −10  Rate error critical        (err_norm > 0.40)
  − 5  Rate error warning         (err_norm > 0.25)  [cap −15 across axes]
  − 5  I-term steady bias warning (i_bias  > 1.5)    [cap −10 across axes]
  − 3  Roll/Pitch P-gain asymmetry (ratio > 2.0)
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

I_RATIO_WARN  = 0.45   # I-term fraction of total PID output — warning
I_RATIO_CRIT  = 0.65   # I-term fraction of total PID output — critical
D_NOISE_WARN  = 0.40   # D-term std / output mean — warning
D_NOISE_CRIT  = 0.80   # D-term std / output mean — critical
ERR_NORM_WARN = 0.25   # rms(Err) / rms(Tar)     — warning
ERR_NORM_CRIT = 0.40   # rms(Err) / rms(Tar)     — critical
I_BIAS_WARN   = 1.5    # |mean(I)| / std(I)       — integrator bias warning
P_ASYM_WARN   = 2.0    # roll/pitch mean(|P|) ratio — asymmetry warning

# Per-category score deduction caps (prevent one bad axis from swamping score)
_I_DOM_CAP    = 20.0
_D_NOISE_CAP  = 16.0
_ERR_CAP      = 15.0
_BIAS_CAP     = 10.0


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    PID Tuning Quality analysis.

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
    end_time = disarm_time if disarm_time is not None else float("inf")

    # ── Load and filter PID DataFrames ────────────────────────────────────────
    pidr_df = _filter_window(result.get("PIDR"), arm_time, end_time)
    pidp_df = _filter_window(result.get("PIDP"), arm_time, end_time)
    pidy_df = _filter_window(result.get("PIDY"), arm_time, end_time)

    roll_ok  = pidr_df is not None and len(pidr_df) >= 30
    pitch_ok = pidp_df is not None and len(pidp_df) >= 30
    yaw_ok   = pidy_df is not None and len(pidy_df) >= 30

    if not roll_ok and not pitch_ok:
        logger.warning(
            "PIDR/PIDP messages not found (or < 30 samples) — PID tuning analysis unavailable."
        )
        return _unavailable(
            "No PID rate messages (PIDR/PIDP) found in log. "
            "Enable rate PID logging via LOG_BITMASK (bit 6)."
        )

    metrics["axes_available"] = (
        (["roll"] if roll_ok else []) +
        (["pitch"] if pitch_ok else []) +
        (["yaw"] if yaw_ok else [])
    )

    # ── Compute per-axis statistics ───────────────────────────────────────────
    roll_stats  = _axis_stats(pidr_df) if roll_ok  else None
    pitch_stats = _axis_stats(pidp_df) if pitch_ok else None
    yaw_stats   = _axis_stats(pidy_df) if yaw_ok   else None

    metrics["pid_roll"]  = roll_stats  if roll_stats  else {}
    metrics["pid_pitch"] = pitch_stats if pitch_stats else {}
    metrics["pid_yaw"]   = yaw_stats   if yaw_stats   else {}

    # ── Score each axis ───────────────────────────────────────────────────────
    score = 100.0

    # Accumulators for capped category totals
    i_dom_total  = 0.0
    d_noise_total = 0.0
    err_total    = 0.0
    bias_total   = 0.0

    for axis_name, stats, weight in [
        ("roll",  roll_stats,  1.0),
        ("pitch", pitch_stats, 1.0),
        ("yaw",   yaw_stats,   0.5),   # yaw scored at half-weight
    ]:
        if stats is None:
            continue
        ax_issues, ax_i_dom, ax_d_noise, ax_err, ax_bias = _score_axis(
            stats, axis_name, profile
        )
        issues.extend(ax_issues)
        i_dom_total   += ax_i_dom   * weight
        d_noise_total += ax_d_noise * weight
        err_total     += ax_err     * weight
        bias_total    += ax_bias    * weight

    # Apply capped category deductions
    score -= min(_I_DOM_CAP,   i_dom_total)
    score -= min(_D_NOISE_CAP, d_noise_total)
    score -= min(_ERR_CAP,     err_total)
    score -= min(_BIAS_CAP,    bias_total)

    # ── Roll/Pitch P-gain asymmetry ───────────────────────────────────────────
    if roll_stats and pitch_stats:
        rp_mean = roll_stats.get("p_mean_abs", 0.0)
        pp_mean = pitch_stats.get("p_mean_abs", 0.0)
        if rp_mean > 1e-4 and pp_mean > 1e-4:
            asym_ratio = max(rp_mean, pp_mean) / min(rp_mean, pp_mean)
            metrics["rp_p_asymmetry_ratio"] = round(float(asym_ratio), 2)
            if asym_ratio > P_ASYM_WARN:
                score -= 3.0
                issues.append(_issue(
                    "warning", "PID-005",
                    f"Roll/Pitch P-term asymmetry: ratio {asym_ratio:.2f} "
                    f"(roll mean |P| {rp_mean:.4f} vs pitch {pp_mean:.4f}). "
                    "Possible payload imbalance, asymmetric motor wear, or different "
                    "airframe stiffness between roll and pitch axes.",
                    value=round(float(asym_ratio), 2),
                    threshold=P_ASYM_WARN,
                ))
        else:
            metrics["rp_p_asymmetry_ratio"] = None

    score = max(0.0, min(100.0, score))

    from analyzer.scoring.engine import _grade
    grade = _grade(score)

    # ── Build summary ─────────────────────────────────────────────────────────
    parts = []
    for ax, stats in [("Roll", roll_stats), ("Pitch", pitch_stats), ("Yaw", yaw_stats)]:
        if stats:
            i_r = stats.get("i_ratio")
            d_n = stats.get("d_noise_idx")
            i_r_s = f"{i_r:.2f}" if i_r is not None else "—"
            d_n_s = f"{d_n:.2f}" if d_n is not None else "—"
            parts.append(f"{ax}: I-ratio {i_r_s} / D-noise {d_n_s}")
    summary = "  |  ".join(parts) if parts else "PID data analysed"

    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ── Per-axis statistics ───────────────────────────────────────────────────────

def _axis_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute PID quality statistics for one axis.
    Returns empty dict if required columns are missing.
    """
    stats: Dict[str, Any] = {}

    p_col   = _col(df, ["P"])
    i_col   = _col(df, ["I"])
    d_col   = _col(df, ["D"])
    out_col = _col(df, ["Out", "out"])
    err_col = _col(df, ["Err", "err", "E"])
    tar_col = _col(df, ["Tar", "tar", "Des", "Target"])

    # P and I columns are the minimum requirement
    if p_col is None or i_col is None:
        logger.debug("PID message missing P or I column — skipping axis.")
        return stats

    p_arr = df[p_col].values.astype(float)
    i_arr = df[i_col].values.astype(float)

    # Use nanmean/nanstd throughout so that unexpected NaN values in any column
    # (e.g. absent field in a specific firmware version) do not propagate into
    # metrics and crash JSON serialisation with "Out of range float values".
    abs_p = float(np.nanmean(np.abs(p_arr)))
    abs_i = float(np.nanmean(np.abs(i_arr)))

    if d_col is not None:
        d_arr = df[d_col].values.astype(float)
        abs_d = float(np.nanmean(np.abs(d_arr)))
    else:
        d_arr = None
        abs_d = 0.0

    # ── I-term dominance ratio ────────────────────────────────────────────────
    total_pid = abs_p + abs_i + abs_d
    i_ratio = abs_i / (total_pid + 1e-9)

    stats["p_mean_abs"] = round(abs_p, 4)
    stats["i_mean_abs"] = round(abs_i, 4)
    stats["d_mean_abs"] = round(abs_d, 4)
    stats["i_ratio"]    = round(float(i_ratio), 3)

    # ── I-term steady bias (integrator trim indicator) ────────────────────────
    i_mean = float(np.nanmean(i_arr))
    i_std  = float(np.nanstd(i_arr))
    i_bias = abs(i_mean) / (i_std + 1e-9)
    stats["i_bias"] = round(float(i_bias), 3)
    stats["i_mean"] = round(i_mean, 4)
    stats["i_std"]  = round(i_std,  4)

    # ── D-term noise index ────────────────────────────────────────────────────
    if d_arr is not None:
        d_std = float(np.nanstd(d_arr))
        # Use Out magnitude as reference; fall back to P magnitude if Out absent
        if out_col is not None:
            out_arr = df[out_col].values.astype(float)
            out_mag = float(np.nanmean(np.abs(out_arr)))
        else:
            out_mag = abs_p
        d_noise = d_std / (out_mag + 1e-9)
        stats["d_noise_idx"] = round(float(d_noise), 3)
        stats["d_std"]       = round(d_std, 4)
    else:
        stats["d_noise_idx"] = None
        stats["d_std"]       = None

    # ── Normalised rate tracking error ────────────────────────────────────────
    if err_col is not None:
        err_arr  = df[err_col].values.astype(float)
        err_rms  = float(np.sqrt(np.nanmean(err_arr ** 2)))
        stats["err_rms"] = round(err_rms, 3)

        if tar_col is not None:
            tar_arr = df[tar_col].values.astype(float)
            tar_rms = float(np.sqrt(np.nanmean(tar_arr ** 2)))
            err_norm = err_rms / (tar_rms + 1e-9)
            stats["err_norm"] = round(float(err_norm), 3)
        else:
            stats["err_norm"] = None
    else:
        stats["err_rms"]  = None
        stats["err_norm"] = None

    return stats


# ── Per-axis scoring ──────────────────────────────────────────────────────────

def _score_axis(
    stats: Dict[str, Any],
    axis_name: str,
    profile: DroneProfile,
) -> Tuple[List[Dict], float, float, float, float]:
    """
    Score one PID axis.

    Returns
    -------
    (issues, i_dom_deduction, d_noise_deduction, err_deduction, bias_deduction)
    Deductions are BEFORE axis weighting and BEFORE category caps.
    """
    issues: List[Dict] = []
    i_dom_ded  = 0.0
    d_noise_ded = 0.0
    err_ded    = 0.0
    bias_ded   = 0.0

    i_ratio  = stats.get("i_ratio",    0.0)
    d_noise  = stats.get("d_noise_idx")
    err_norm = stats.get("err_norm")
    i_bias   = stats.get("i_bias",     0.0)

    ax = axis_name.capitalize()

    # ── 1. I-term dominance ───────────────────────────────────────────────────
    if i_ratio > I_RATIO_CRIT:
        i_dom_ded += 15.0
        issues.append(_issue(
            "critical", "PID-001",
            f"{ax} I-term dominance critical: ratio {i_ratio:.2f} "
            f"(threshold {I_RATIO_CRIT}). The integrator is providing >{I_RATIO_CRIT*100:.0f}% "
            "of the total control output. Likely causes: P/D gains too low, persistent "
            "mechanical offset (prop wear, payload shift), or structural resonance absorbing "
            "corrective input. Run AutoTune or check airframe condition.",
            value=round(i_ratio, 3),
            threshold=I_RATIO_CRIT,
        ))
    elif i_ratio > I_RATIO_WARN:
        i_dom_ded += 10.0
        issues.append(_issue(
            "warning", "PID-001",
            f"{ax} I-term elevated: ratio {i_ratio:.2f} "
            f"(threshold {I_RATIO_WARN}). I-term is compensating more than expected. "
            "Monitor for prop wear or payload imbalance. Consider reviewing P/D gains.",
            value=round(i_ratio, 3),
            threshold=I_RATIO_WARN,
        ))

    # ── 2. D-term noise ───────────────────────────────────────────────────────
    if d_noise is not None:
        if d_noise > D_NOISE_CRIT:
            d_noise_ded += 12.0
            issues.append(_issue(
                "critical", "PID-002",
                f"{ax} D-term noise critical: index {d_noise:.2f} "
                f"(threshold {D_NOISE_CRIT}). Derivative is amplifying IMU noise excessively. "
                "Enable or tighten the harmonic notch filter (INS_HNTCH_ENABLE / INS_HNTCH_FREQ), "
                "reduce D-gain by 20-30%, or lower INS_GYRO_FILTER cutoff frequency.",
                value=round(d_noise, 3),
                threshold=D_NOISE_CRIT,
            ))
        elif d_noise > D_NOISE_WARN:
            d_noise_ded += 8.0
            issues.append(_issue(
                "warning", "PID-002",
                f"{ax} D-term noise elevated: index {d_noise:.2f} "
                f"(threshold {D_NOISE_WARN}). Consider enabling the harmonic notch filter "
                "(INS_HNTCH_ENABLE) or reviewing INS_GYRO_FILTER cutoff.",
                value=round(d_noise, 3),
                threshold=D_NOISE_WARN,
            ))

    # ── 3. Normalised rate tracking error ─────────────────────────────────────
    if err_norm is not None:
        if err_norm > ERR_NORM_CRIT:
            err_ded += 10.0
            issues.append(_issue(
                "critical", "PID-003",
                f"{ax} rate tracking error critical: normalised RMS {err_norm:.2f} "
                f"(threshold {ERR_NORM_CRIT}). The {axis_name} rate controller cannot "
                "track the commanded rate. Check motor/ESC health, propeller condition, "
                "and verify P-gain is not too low.",
                value=round(err_norm, 3),
                threshold=ERR_NORM_CRIT,
            ))
        elif err_norm > ERR_NORM_WARN:
            err_ded += 5.0
            issues.append(_issue(
                "warning", "PID-003",
                f"{ax} rate tracking error elevated: normalised RMS {err_norm:.2f} "
                f"(threshold {ERR_NORM_WARN}). Review P-gain magnitude or check for "
                f"mechanical play / compliance in the {axis_name} axis.",
                value=round(err_norm, 3),
                threshold=ERR_NORM_WARN,
            ))

    # ── 4. I-term steady bias ─────────────────────────────────────────────────
    if i_bias > I_BIAS_WARN:
        bias_ded += 5.0
        if axis_name == "yaw":
            cause = "compass interference or yaw motor/ESC imbalance"
        else:
            cause = "centre-of-gravity offset, motor/ESC trim, or prop pitch imbalance"
        issues.append(_issue(
            "warning", "PID-004",
            f"{ax} integrator steady bias: ratio {i_bias:.2f} "
            f"(threshold {I_BIAS_WARN}). The integrator consistently offsets in one "
            f"direction — possible {cause}.",
            value=round(i_bias, 3),
            threshold=I_BIAS_WARN,
        ))

    return issues, i_dom_ded, d_noise_ded, err_ded, bias_ded


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_window(
    df: Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
) -> Optional[pd.DataFrame]:
    """Filter a DataFrame to the armed flight window. Returns None if empty."""
    if df is None or df.empty:
        return None
    mask = (df["timestamp"] >= arm_time) & (df["timestamp"] <= end_time)
    filtered = df[mask].reset_index(drop=True)
    return filtered if len(filtered) >= 10 else None


def _col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from candidates, or None."""
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
    # Return 100/A (not 50/C) — absence of PIDR/PIDP in the log is a configuration
    # choice (LOG_BITMASK), not a fault. The module is excluded from scoring entirely
    # (available=False), so the score value is cosmetic only; 100/A avoids alarming
    # the operator unnecessarily. Consistent with vtol_transition and airspeed pattern.
    return {
        "score":     100.0,
        "grade":     "A",
        "available": False,
        "issues":    [],
        "metrics":   {"unavailable_reason": reason},
        "summary":   f"PID tuning analysis unavailable: {reason}",
    }
