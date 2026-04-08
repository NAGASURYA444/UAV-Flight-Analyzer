"""
Telemetry Link Quality Module
==============================
Phase 4.6 — GCS Telemetry Radio Health

Data source: RADIO messages (SiK / RFD900 / other MAVLink telemetry radios).
ArduPilot logs RADIO messages when it receives RADIO_STATUS packets from the
telemetry radio and forwards them to DataFlash.

Checks performed:
  — Mean local RSSI (received signal strength at vehicle radio, 0–255)
  — Mean remote RSSI (RSSI at GCS end, 0–255)
  — Signal-to-noise margin (SNR = RSSI − Noise, higher = better)
  — TX buffer utilisation (remaining space 0–100%; low = data backlog)
  — Receive error rate (RxErrors per minute in armed window)
  — RSSI dropout events (RSSI falls below critical floor for ≥ 2 s)

RADIO message fields:
  RSSI     : Signal strength at vehicle radio (0–255, higher = better)
  RemRSSI  : Signal strength at GCS radio (0–255, higher = better)
  TxBuf    : Remaining TX buffer space (0–100%, 100 = fully empty = best)
  Noise    : Local noise floor (0–255, lower = better for SNR)
  RemNoise : GCS-side noise floor (0–255, lower = better)
  RxErrors : Cumulative receive error count (monotonic, resets to 0 at arm)
  Fixed    : Cumulative error-corrected packet count

Unavailable: RADIO messages absent from log.  This is expected when no SiK /
  MAVLink telemetry radio is connected, or LOG_BITMASK does not include
  telemetry radio forwarding.
  → Returns score=100, grade="A", available=False (expected unavailability).

Scoring deductions (from 100):
  −25   Mean RSSI critically low (< 50)
  −15   Mean RSSI low (50–79)
  −20   Mean SNR critically low (< 5 dB)
  −10   Mean SNR moderate (5–9 dB)
  −15   High packet error rate (> 20 errors/min)
  −8    Moderate packet error rate (5–20 errors/min)
  −10   TX buffer congested (mean remaining space < 20%)
  −5    TX buffer moderately congested (20–49%)
  −10   RSSI dropout events detected (RSSI < 30 for ≥ 2 s)
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

_RSSI_CRITICAL   = 50    # Mean RSSI below this → critical
_RSSI_WARN       = 80    # Mean RSSI below this → warning
_SNR_CRITICAL    = 5     # Mean SNR (dB) below this → critical
_SNR_WARN        = 10    # Mean SNR (dB) below this → warning
_ERR_RATE_CRIT   = 20.0  # RxErrors/min → critical
_ERR_RATE_WARN   = 5.0   # RxErrors/min → warning
_TXBUF_CRIT      = 20    # Mean TxBuf % remaining below → critical congestion
_TXBUF_WARN      = 50    # Mean TxBuf % remaining below → moderate congestion
_RSSI_DROPOUT_FL = 30    # RSSI below this = near-zero signal (dropout floor)
_RSSI_DROPOUT_DUR = 2.0  # Minimum duration (s) to count as a dropout episode


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Telemetry Link Quality analysis.

    Returns standardised result dict.
    """
    radio_df = result.get("RADIO")

    # ── Availability check ────────────────────────────────────────────────────
    if radio_df is None or len(radio_df) == 0:
        return _unavailable("RADIO messages not present in log. "
                            "Telemetry radio not connected or LOG_BITMASK "
                            "does not include radio status forwarding.")

    # Scope to armed flight window
    end_time = disarm_time if disarm_time is not None else float("inf")
    armed_df = (
        radio_df[
            (radio_df["timestamp"] >= arm_time) &
            (radio_df["timestamp"] <= end_time)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(armed_df) < 10:
        return _unavailable("Insufficient RADIO samples in armed flight window "
                            f"({len(armed_df)} found, need >= 10).")

    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    metrics["sample_count"] = len(armed_df)

    # ── Step 1: RSSI analysis ─────────────────────────────────────────────────
    rssi_metrics, rssi_issues = _analyse_rssi(armed_df)
    metrics.update(rssi_metrics)
    issues.extend(rssi_issues)

    # ── Step 2: SNR analysis ──────────────────────────────────────────────────
    snr_metrics, snr_issues = _analyse_snr(armed_df)
    metrics.update(snr_metrics)
    issues.extend(snr_issues)

    # ── Step 3: TX buffer analysis ────────────────────────────────────────────
    txbuf_metrics, txbuf_issues = _analyse_txbuf(armed_df)
    metrics.update(txbuf_metrics)
    issues.extend(txbuf_issues)

    # ── Step 4: Packet error rate ─────────────────────────────────────────────
    flight_duration_min = (armed_df["timestamp"].iloc[-1] -
                           armed_df["timestamp"].iloc[0]) / 60.0
    err_metrics, err_issues = _analyse_errors(armed_df, flight_duration_min)
    metrics.update(err_metrics)
    issues.extend(err_issues)

    # ── Step 5: RSSI dropout events ───────────────────────────────────────────
    dropout_metrics, dropout_issues = _analyse_rssi_dropouts(armed_df)
    metrics.update(dropout_metrics)
    issues.extend(dropout_issues)

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _compute_score(metrics, issues)
    grade = _grade(score)

    # Summary line
    rssi_mean = metrics.get("rssi_mean")
    snr_mean  = metrics.get("snr_mean")
    err_rate  = metrics.get("rxerror_rate_per_min", 0.0)
    dropouts  = metrics.get("rssi_dropout_count", 0)

    parts = []
    if rssi_mean is not None:
        parts.append(f"RSSI {rssi_mean:.0f}/255")
    if snr_mean is not None:
        parts.append(f"SNR {snr_mean:.1f} dB")
    if err_rate is not None:
        parts.append(f"Errors {err_rate:.1f}/min")
    parts.append(f"Dropouts: {dropouts}")

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
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_rssi(
    df: pd.DataFrame,
) -> Tuple[Dict, List[Dict]]:
    """Analyse local and remote RSSI signal levels."""
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if "RSSI" not in df.columns:
        metrics["rssi_available"] = False
        return metrics, issues

    rssi_vals = df["RSSI"].dropna().values
    if len(rssi_vals) < 5:
        metrics["rssi_available"] = False
        return metrics, issues

    rssi_mean = float(np.mean(rssi_vals))
    rssi_min  = float(np.min(rssi_vals))
    rssi_max  = float(np.max(rssi_vals))

    metrics["rssi_available"]  = True
    metrics["rssi_mean"]       = round(rssi_mean, 1)
    metrics["rssi_min"]        = round(rssi_min, 1)
    metrics["rssi_max"]        = round(rssi_max, 1)

    # Remote RSSI (GCS end)
    if "RemRSSI" in df.columns:
        rem_vals = df["RemRSSI"].dropna().values
        if len(rem_vals) >= 5:
            metrics["rem_rssi_mean"] = round(float(np.mean(rem_vals)), 1)
            metrics["rem_rssi_min"]  = round(float(np.min(rem_vals)), 1)

    if rssi_mean < _RSSI_CRITICAL:
        issues.append(_issue(
            "critical", "TLM-001",
            f"Telemetry RSSI critically low: mean {rssi_mean:.0f}/255 "
            f"(min {rssi_min:.0f}). Radio link is at risk of complete loss — "
            "check antenna orientation, reduce range, or replace radio hardware.",
            value=round(rssi_mean, 1),
            threshold=float(_RSSI_CRITICAL),
        ))
    elif rssi_mean < _RSSI_WARN:
        issues.append(_issue(
            "warning", "TLM-001",
            f"Telemetry RSSI low: mean {rssi_mean:.0f}/255 "
            f"(min {rssi_min:.0f}). Marginal signal — check antenna orientation "
            "and clearance from power cables.",
            value=round(rssi_mean, 1),
            threshold=float(_RSSI_WARN),
        ))

    return metrics, issues


def _analyse_snr(
    df: pd.DataFrame,
) -> Tuple[Dict, List[Dict]]:
    """Compute signal-to-noise ratio = RSSI − Noise."""
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if "RSSI" not in df.columns or "Noise" not in df.columns:
        metrics["snr_available"] = False
        return metrics, issues

    # Use joint valid mask so RSSI and Noise values come from the same rows.
    # Separate dropna() calls can produce different lengths if NaN positions differ,
    # causing misaligned subtraction.
    valid_mask = df["RSSI"].notna() & df["Noise"].notna()
    rssi_vals  = df.loc[valid_mask, "RSSI"].values
    noise_vals = df.loc[valid_mask, "Noise"].values

    if len(rssi_vals) < 5:
        metrics["snr_available"] = False
        return metrics, issues

    snr_vals   = rssi_vals - noise_vals
    snr_mean   = float(np.mean(snr_vals))
    noise_mean = float(np.mean(noise_vals))
    rssi_mean  = float(np.mean(rssi_vals))

    metrics["snr_available"] = True
    metrics["snr_mean"]      = round(snr_mean, 1)
    metrics["noise_mean"]    = round(noise_mean, 1)

    # Remote SNR — same joint-mask approach
    if "RemRSSI" in df.columns and "RemNoise" in df.columns:
        rem_mask  = df["RemRSSI"].notna() & df["RemNoise"].notna()
        rem_rssi  = df.loc[rem_mask, "RemRSSI"].values
        rem_noise = df.loc[rem_mask, "RemNoise"].values
        if len(rem_rssi) >= 5:
            rem_snr = float(np.mean(rem_rssi - rem_noise))
            metrics["rem_snr_mean"] = round(rem_snr, 1)

    if snr_mean < _SNR_CRITICAL:
        issues.append(_issue(
            "critical", "TLM-002",
            f"Telemetry SNR critically low: {snr_mean:.1f} dB "
            f"(RSSI {rssi_mean:.0f} − Noise {noise_mean:.0f}). "
            "High interference — relocate antenna away from ESCs/motors "
            "and use a shielded coaxial extension.",
            value=round(snr_mean, 1),
            threshold=float(_SNR_CRITICAL),
        ))
    elif snr_mean < _SNR_WARN:
        issues.append(_issue(
            "warning", "TLM-002",
            f"Telemetry SNR low: {snr_mean:.1f} dB. Elevated noise floor "
            f"(Noise = {noise_mean:.0f}). Move telemetry antenna away from "
            "power distribution board and ESC signal wiring.",
            value=round(snr_mean, 1),
            threshold=float(_SNR_WARN),
        ))

    return metrics, issues


def _analyse_txbuf(
    df: pd.DataFrame,
) -> Tuple[Dict, List[Dict]]:
    """
    Analyse TX buffer remaining space.
    TxBuf: 0 = buffer full (congested), 100 = buffer empty (no backlog).
    """
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if "TxBuf" not in df.columns:
        metrics["txbuf_available"] = False
        return metrics, issues

    buf_vals = df["TxBuf"].dropna().values
    if len(buf_vals) < 5:
        metrics["txbuf_available"] = False
        return metrics, issues

    buf_mean = float(np.mean(buf_vals))
    buf_min  = float(np.min(buf_vals))
    pct_congested = float(np.mean(buf_vals < _TXBUF_WARN)) * 100.0

    metrics["txbuf_available"]    = True
    metrics["txbuf_mean_pct"]     = round(buf_mean, 1)
    metrics["txbuf_min_pct"]      = round(buf_min, 1)
    metrics["txbuf_congested_pct"] = round(pct_congested, 1)

    if buf_mean < _TXBUF_CRIT:
        issues.append(_issue(
            "critical", "TLM-003",
            f"Telemetry TX buffer critically full: mean space remaining "
            f"{buf_mean:.0f}% (min {buf_min:.0f}%). GCS is not consuming "
            "telemetry fast enough — reduce telemetry stream rates "
            "(SRx_* parameters) or check GCS connectivity.",
            value=round(buf_mean, 1),
            threshold=float(_TXBUF_CRIT),
        ))
    elif buf_mean < _TXBUF_WARN:
        issues.append(_issue(
            "warning", "TLM-003",
            f"Telemetry TX buffer moderately congested: mean space remaining "
            f"{buf_mean:.0f}%. Consider reducing SRx_* telemetry stream rates "
            "to prevent data backlog.",
            value=round(buf_mean, 1),
            threshold=float(_TXBUF_WARN),
        ))

    return metrics, issues


def _analyse_errors(
    df: pd.DataFrame,
    flight_duration_min: float,
) -> Tuple[Dict, List[Dict]]:
    """Compute packet receive error rate from cumulative RxErrors counter."""
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if "RxErrors" not in df.columns:
        metrics["rxerror_available"] = False
        return metrics, issues

    err_vals = df["RxErrors"].dropna().values
    if len(err_vals) < 2:
        metrics["rxerror_available"] = False
        return metrics, issues

    # Cumulative counter: total errors = last − first (handles reset at arm)
    rxerrors_total = max(0, int(err_vals[-1]) - int(err_vals[0]))

    # Fixed packets (error-corrected): also cumulative
    fixed_total = 0
    if "Fixed" in df.columns:
        fixed_vals = df["Fixed"].dropna().values
        if len(fixed_vals) >= 2:
            fixed_total = max(0, int(fixed_vals[-1]) - int(fixed_vals[0]))

    # Error rate per minute (avoid div/zero)
    if flight_duration_min > 0.01:
        err_rate = rxerrors_total / flight_duration_min
    else:
        err_rate = 0.0

    metrics["rxerror_available"]      = True
    metrics["rxerrors_total"]         = rxerrors_total
    metrics["fixed_total"]            = fixed_total
    metrics["rxerror_rate_per_min"]   = round(err_rate, 2)

    if err_rate > _ERR_RATE_CRIT:
        issues.append(_issue(
            "critical", "TLM-004",
            f"High telemetry packet error rate: {err_rate:.1f} errors/min "
            f"({rxerrors_total} total). Significant RF interference or poor "
            "link margin — check antenna condition, reduce operating range, "
            "and verify radio frequency is not congested.",
            value=round(err_rate, 2),
            threshold=float(_ERR_RATE_CRIT),
        ))
    elif err_rate > _ERR_RATE_WARN:
        issues.append(_issue(
            "warning", "TLM-004",
            f"Elevated telemetry packet error rate: {err_rate:.1f} errors/min "
            f"({rxerrors_total} total). Check antenna orientation and ensure "
            "radio operating frequency is clear of interference.",
            value=round(err_rate, 2),
            threshold=float(_ERR_RATE_WARN),
        ))

    return metrics, issues


def _analyse_rssi_dropouts(
    df: pd.DataFrame,
) -> Tuple[Dict, List[Dict]]:
    """
    Detect episodes where RSSI falls below the near-zero floor (_RSSI_DROPOUT_FL)
    for at least _RSSI_DROPOUT_DUR seconds — indicating a complete link loss event.
    """
    metrics: Dict[str, Any] = {}
    issues: List[Dict] = []

    if "RSSI" not in df.columns or "timestamp" not in df.columns:
        metrics["rssi_dropout_count"] = 0
        metrics["rssi_dropouts"]      = []
        return metrics, issues

    rssi_vals  = df["RSSI"].values
    timestamps = df["timestamp"].values

    dropout_events = []
    in_dropout     = False
    dropout_start  = 0.0
    dropout_min    = 255

    for i, (ts, val) in enumerate(zip(timestamps, rssi_vals)):
        if val < _RSSI_DROPOUT_FL and not in_dropout:
            in_dropout    = True
            dropout_start = float(ts)
            dropout_min   = int(val)
        elif val < _RSSI_DROPOUT_FL and in_dropout:
            dropout_min = min(dropout_min, int(val))
        elif val >= _RSSI_DROPOUT_FL and in_dropout:
            in_dropout = False
            duration   = float(ts) - dropout_start
            if duration >= _RSSI_DROPOUT_DUR:
                dropout_events.append({
                    "timestamp_s": round(dropout_start, 2),
                    "duration_s":  round(duration, 2),
                    "min_rssi":    dropout_min,
                })
            dropout_min = 255

    # Handle dropout that extends to end of window
    if in_dropout:
        duration = float(timestamps[-1]) - dropout_start
        if duration >= _RSSI_DROPOUT_DUR:
            dropout_events.append({
                "timestamp_s": round(dropout_start, 2),
                "duration_s":  round(duration, 2),
                "min_rssi":    dropout_min,
            })

    metrics["rssi_dropout_count"] = len(dropout_events)
    metrics["rssi_dropouts"]      = dropout_events

    for evt in dropout_events:
        sev = "critical" if evt["duration_s"] > 10.0 else "warning"
        issues.append(_issue(
            sev, "TLM-005",
            f"Telemetry RSSI dropout: {evt['duration_s']:.1f} s of near-zero "
            f"signal (RSSI < {_RSSI_DROPOUT_FL}) starting at "
            f"T+{evt['timestamp_s']:.1f} s. Check for physical obstruction, "
            "antenna damage, or RF shielding issues at that flight phase.",
            value=round(evt["duration_s"], 2),
            threshold=float(_RSSI_DROPOUT_DUR),
            timestamp=evt["timestamp_s"],
        ))

    return metrics, issues


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(metrics: Dict, issues: List[Dict]) -> float:
    score = 100.0

    # RSSI level
    rssi_mean = metrics.get("rssi_mean")
    if rssi_mean is not None:
        if rssi_mean < _RSSI_CRITICAL:
            score -= 25.0
        elif rssi_mean < _RSSI_WARN:
            score -= 15.0

    # SNR level
    snr_mean = metrics.get("snr_mean")
    if snr_mean is not None:
        if snr_mean < _SNR_CRITICAL:
            score -= 20.0
        elif snr_mean < _SNR_WARN:
            score -= 10.0

    # TX buffer
    txbuf_mean = metrics.get("txbuf_mean_pct")
    if txbuf_mean is not None:
        if txbuf_mean < _TXBUF_CRIT:
            score -= 10.0
        elif txbuf_mean < _TXBUF_WARN:
            score -= 5.0

    # Packet error rate
    err_rate = metrics.get("rxerror_rate_per_min")
    if err_rate is not None:
        if err_rate > _ERR_RATE_CRIT:
            score -= 15.0
        elif err_rate > _ERR_RATE_WARN:
            score -= 8.0

    # RSSI dropouts
    dropout_count = metrics.get("rssi_dropout_count", 0)
    if dropout_count > 0:
        score -= min(10.0, dropout_count * 5.0)

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
