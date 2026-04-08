"""
Fleet Trend Analysis  (Phase 6)
================================
Computes per-metric trends across a drone's flight history.
Uses simple linear regression (numpy polyfit) — no extra dependencies.

Key outputs
-----------
compute_trends(history)   →  dict of metric → TrendResult
fleet_alerts(history)     →  list of human-readable alert strings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Trend result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrendResult:
    metric:      str
    label:       str
    values:      List[Optional[float]]   # newest → oldest order (as stored)
    slope:       float                   # change per flight (linear regression)
    direction:   str                     # "RISING" | "FALLING" | "STABLE"
    arrow:       str                     # "↑↑" | "↑" | "→" | "↓" | "↓↓"
    alert:       bool                    # True if this trend needs attention
    alert_msg:   str                     # Human-readable alert message
    unit:        str   = ""
    latest:      Optional[float] = None  # Most recent value


# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions: (db_column, display_label, unit, direction_bad, warn_threshold)
# direction_bad = "up"  → rising is bad (e.g. internal resistance)
# direction_bad = "down"→ falling is bad (e.g. cell voltage, score)
# ─────────────────────────────────────────────────────────────────────────────

_METRICS = [
    # (column,                          label,                    unit,    bad,    alert_slope_per_flight)
    ("overall_score",                   "Overall Score",          "",      "down", -2.0),
    ("bat_internal_resistance_mohm",    "Battery IR",             "mohm", "up",    1.0),
    ("bat_end_cell_v",                  "End Cell Voltage",       "V",     "down", -0.02),
    ("bat_capacity_remaining_pct",      "Battery Remaining",      "%",     "down", -3.0),
    ("vibe_x",                          "Vibration X",            "m/s2",  "up",    0.5),
    ("vibe_y",                          "Vibration Y",            "m/s2",  "up",    0.5),
    ("vibe_z",                          "Vibration Z",            "m/s2",  "up",    0.5),
    ("motor_imbalance_pct",             "Motor Imbalance",        "%",     "up",    1.0),
    ("efficiency_wh_per_km",            "Efficiency",             "Wh/km","up",    2.0),
    ("score_battery",                   "Battery Score",          "",      "down", -3.0),
    ("score_motors",                    "Motors Score",           "",      "down", -3.0),
    ("score_vibration",                 "Vibration Score",        "",      "down", -3.0),
]

# Minimum number of flights needed to compute a meaningful trend
_MIN_FLIGHTS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Core trend computation
# ─────────────────────────────────────────────────────────────────────────────

def _linear_slope(values: List[float]) -> float:
    """Fit a line y = a*x + b and return slope a."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    try:
        a, _ = np.polyfit(x, y, 1)
        return float(a)
    except Exception:
        return 0.0


def _arrow(slope: float, direction_bad: str) -> str:
    abs_s = abs(slope)
    if abs_s < 0.01:
        return "→"
    if direction_bad == "up":
        if slope > 0:
            return "↑↑" if abs_s > 2.0 else "↑"
        else:
            return "↓"
    else:  # direction_bad == "down"
        if slope < 0:
            return "↓↓" if abs_s > 2.0 else "↓"
        else:
            return "↑"


def compute_trends(
    history: List[Dict],
) -> Dict[str, TrendResult]:
    """
    Compute trends for each key metric across a drone's flight history.

    Parameters
    ----------
    history : list of flight dicts from db.get_drone_history()
              Expected order: newest first.

    Returns
    -------
    dict mapping metric column name → TrendResult
    """
    if len(history) < _MIN_FLIGHTS:
        return {}

    # Reverse so oldest→newest for regression (chronological order)
    chron = list(reversed(history))

    results: Dict[str, TrendResult] = {}

    for col, label, unit, bad, alert_slope in _METRICS:
        raw = [r.get(col) for r in chron]
        valid = [(i, v) for i, v in enumerate(raw) if v is not None]

        if len(valid) < _MIN_FLIGHTS:
            continue

        xs = [i for i, _ in valid]
        ys = [v for _, v in valid]

        slope = _linear_slope(ys) if len(xs) >= 2 else 0.0

        # Direction
        abs_s = abs(slope)
        if abs_s < 0.01:
            direction = "STABLE"
        elif slope > 0:
            direction = "RISING"
        else:
            direction = "FALLING"

        # Is this slope in the "bad" direction and large enough to alert?
        if bad == "up":
            alert = slope >= abs(alert_slope)
            alert_msg = (
                f"{label} is rising (+{slope:.2f}{unit}/flight). "
                f"Current: {ys[-1]:.2f}{unit}"
            ) if alert else ""
        else:
            alert = slope <= -abs(alert_slope)
            alert_msg = (
                f"{label} is falling ({slope:.2f}{unit}/flight). "
                f"Current: {ys[-1]:.2f}{unit}"
            ) if alert else ""

        results[col] = TrendResult(
            metric    = col,
            label     = label,
            values    = [r.get(col) for r in history],   # newest first
            slope     = slope,
            direction = direction,
            arrow     = _arrow(slope, bad),
            alert     = alert,
            alert_msg = alert_msg,
            unit      = unit,
            latest    = ys[-1] if ys else None,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fleet-level alert summary
# ─────────────────────────────────────────────────────────────────────────────

def fleet_alerts(history: List[Dict]) -> List[str]:
    """
    Return a list of human-readable alert strings for a drone.
    Empty list = no alerts.
    """
    trends = compute_trends(history)
    return [t.alert_msg for t in trends.values() if t.alert and t.alert_msg]


def score_trend_arrow(history: List[Dict]) -> str:
    """
    Return a simple arrow showing whether overall score is trending
    up, down, or stable over the last few flights.
    """
    trends = compute_trends(history)
    t = trends.get("overall_score")
    if t is None:
        return ""
    return t.arrow


def summary_for_cli(history: List[Dict]) -> Tuple[str, List[str]]:
    """
    Returns (trend_arrow, [alert_strings]) for the CLI fleet table.
    """
    arrow  = score_trend_arrow(history)
    alerts = fleet_alerts(history)
    return arrow, alerts
