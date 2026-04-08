"""
VTOL Transition Analysis Module
================================
Phase 4.3 — VTOL ↔ Fixed-Wing Transition Quality (QuadPlane only)

Returns available=False with a neutral score (100) for quadcopter and
fixed_wing types — no deduction applied to those profiles.

Transitions detected
--------------------
Q-mode → FW-mode : VTOL-to-FW  (lift motors ramp down, pusher ramps up)
FW-mode → Q-mode : FW-to-VTOL  (lift motors ramp up, aircraft decelerates)

Per-transition checks
---------------------
1.  Altitude hold   — BARO altitude change in 15 s window after mode switch
2.  Attitude spike  — max roll / pitch deviation in 10 s window
3.  Lift motor ramp — time for lift motors to reach 80 % of hover throttle
                      (FW→VTOL) or drop below 20 % (VTOL→FW)

Scoring (mean of per-transition scores, floored at 0)
------------------------------------------------------
Per transition (deductions from 100 per transition):
-30   Severe altitude loss > 15 m
-15   Altitude loss warning > 5 m
-20   Severe attitude spike > 25 deg (roll or pitch)
-10   Attitude spike warning > 15 deg
-25   Very slow lift motor ramp > 6 s
-15   Slow lift motor ramp > 3 s

If no transitions detected in a VTOL log: score 100, info note.
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
ALT_LOSS_WARN_M      = 5.0    # m  altitude drop in window → warning
ALT_LOSS_CRIT_M      = 15.0   # m  altitude drop → critical
# FW↔VTOL transitions involve large attitude changes during decel/accel.
# Thresholds are more lenient than pure-hover manoeuvres.
PITCH_SPIKE_WARN_DEG = 20.0   # deg attitude spike → warning
PITCH_SPIKE_CRIT_DEG = 35.0   # deg attitude spike → critical
ROLL_SPIKE_WARN_DEG  = 20.0
ROLL_SPIKE_CRIT_DEG  = 35.0
RAMP_WARN_S          = 3.0    # s  lift motor ramp time → warning
RAMP_CRIT_S          = 6.0    # s  lift motor ramp time → critical

ANALYSIS_WINDOW_S    = 15.0   # s  altitude window after transition
ATTITUDE_WINDOW_S    = 10.0   # s  attitude window after transition
ATTITUDE_BASELINE_S  = 3.0    # s  baseline window before transition
HOVER_RAMP_THRESH    = 0.8    # fraction of hover_throttle_pct to declare "ramped up"
IDLE_THRESH_PCT      = 20.0   # % motor throttle below which we call "ramped down"
ACTIVE_THRESH_PCT    = 30.0   # % motor throttle above which we call "active"
RAMP_DETECT_THRESH   = 50.0   # % — transition start detected when mean drops below this

# Landing-type modes: planned descent, so altitude analysis is skipped
_LANDING_MODES = {"QRTL", "QLAND", "LOITER_ALT_QLAND"}

# Q-modes (VTOL hover capable)
_Q_MODES = {
    "QSTABILIZE", "QHOVER", "QLOITER", "QLAND", "QRTL",
    "QAUTOTUNE",  "QACRO",  "LOITER_ALT_QLAND",
}

# Fixed-wing cruise / transition modes
_FW_MODES = {
    "MANUAL", "CIRCLE", "STABILIZE", "TRAINING", "ACRO",
    "FLY_BY_WIRE_A", "FLY_BY_WIRE_B", "CRUISE", "AUTOTUNE",
    "AUTO", "RTL", "LOITER", "TAKEOFF", "AVOID_ADSB",
    "GUIDED", "INITIALISING", "THERMAL",
}


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    VTOL Transition Quality analysis.

    Only meaningful for vtol type. Returns available=False (score 100) for
    multirotor and fixed_wing.
    """
    if profile.type != "vtol":
        return {
            "score":     100.0,
            "grade":     "A",
            "available": False,
            "issues":    [],
            "metrics":   {"unavailable_reason": f"VTOL transition analysis not applicable for {profile.type} type."},
            "summary":   "VTOL transition analysis not applicable for this drone type.",
        }

    issues: List[Dict] = []
    metrics: Dict[str, Any] = {}

    mode_df = result.get("MODE")
    baro_df = result.get("BARO")
    att_df  = result.get("ATT")
    rcou_df = result.get("RCOU")

    end_time = disarm_time if disarm_time is not None else float("inf")

    # Lift motor channels from profile
    hover_thr_pct   = float(getattr(profile.motors, "hover_throttle_pct", 50))
    lift_channels   = [f"C{ch}" for ch in profile.motors.channels]

    # ── Detect transitions ────────────────────────────────────────────────
    transitions = _detect_transitions(
        mode_df, arm_time, end_time,
        rcou_df=rcou_df, lift_channels=lift_channels,
    )

    if not transitions:
        metrics["transition_count"]      = 0
        metrics["transitions"]           = []
        metrics["fw_to_vtol_count"]      = 0
        metrics["vtol_to_fw_count"]      = 0
        metrics["worst_alt_loss_m"]      = 0.0
        metrics["worst_pitch_spike_deg"] = 0.0
        metrics["worst_roll_spike_deg"]  = 0.0
        metrics["slowest_ramp_s"]        = 0.0
        metrics["mean_transition_score"] = 100.0
        issues.append(_issue(
            "info", "VTR-001",
            "No VTOL transitions detected in this flight. "
            "Flight was entirely in one mode category (hover or cruise).",
        ))
        return _build_result(100.0, issues, metrics,
                             "No transitions detected — single-mode flight")

    # ── Analyse each transition ───────────────────────────────────────────
    analysed = []
    for tr in transitions:
        tr_record = _analyse_transition(
            tr, baro_df, att_df, rcou_df,
            lift_channels, hover_thr_pct,
        )
        analysed.append(tr_record)
        issues.extend(tr_record.pop("issues"))   # hoist issues to module level

    # ── Aggregate metrics ─────────────────────────────────────────────────
    fw_to_vtol = sum(1 for t in analysed if t["type"] == "fw_to_vtol")
    vtol_to_fw = sum(1 for t in analysed if t["type"] == "vtol_to_fw")
    all_scores = [t["score"] for t in analysed]
    mean_score = float(np.mean(all_scores)) if all_scores else 100.0

    worst_alt   = max((t.get("alt_drop_m", 0.0) or 0.0) for t in analysed)
    worst_pitch = max((t.get("pitch_spike_deg", 0.0) or 0.0) for t in analysed)
    worst_roll  = max((t.get("roll_spike_deg", 0.0) or 0.0) for t in analysed)
    worst_ramp  = max((t.get("ramp_time_s", 0.0) or 0.0) for t in analysed)

    metrics.update({
        "transition_count":      len(analysed),
        "transitions":           analysed,
        "fw_to_vtol_count":      fw_to_vtol,
        "vtol_to_fw_count":      vtol_to_fw,
        "worst_alt_loss_m":      round(worst_alt, 2),
        "worst_pitch_spike_deg": round(worst_pitch, 2),
        "worst_roll_spike_deg":  round(worst_roll, 2),
        "slowest_ramp_s":        round(worst_ramp, 2),
        "mean_transition_score": round(mean_score, 1),
    })

    issues.insert(0, _issue(
        "info", "VTR-001",
        f"{len(analysed)} VTOL transition{'s' if len(analysed) != 1 else ''} detected "
        f"({vtol_to_fw} to cruise, {fw_to_vtol} to hover). "
        f"Mean quality score: {mean_score:.0f}/100.",
    ))

    # Hard lower bound: if any single transition was very poor, cap mean
    if any(t["score"] < 40 for t in analysed):
        mean_score = max(0.0, mean_score - 5.0)

    summary_parts = [f"{len(analysed)} transition{'s' if len(analysed) != 1 else ''}"]
    if worst_alt > 0.5:
        summary_parts.append(f"max alt loss {worst_alt:.1f} m")
    if worst_ramp > 0:
        summary_parts.append(f"max ramp {worst_ramp:.1f} s")
    summary_parts.append(f"mean score {mean_score:.0f}/100")

    return _build_result(mean_score, issues, metrics, "  |  ".join(summary_parts))


# ─── Transition detection ──────────────────────────────────────────────────────

def _detect_transitions(
    mode_df: Optional[pd.DataFrame],
    arm_time: float,
    end_time: float,
    rcou_df: Optional[pd.DataFrame] = None,
    lift_channels: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Return list of transition events that cross Q↔FW boundary.

    Two detection methods are combined:
    1. Mode-change based  — catches explicit Q↔FW mode switches (e.g. RTL→QRTL)
    2. RCOU-based         — catches VTOL↔FW transitions inside AUTO mode where no
                            MODE message is logged (typical QuadPlane takeoff)
    """
    transitions: List[Dict] = []

    # ── 1. Mode-change based detection ───────────────────────────────────
    if mode_df is not None and not mode_df.empty and "mode_name" in mode_df.columns:
        flight_modes = mode_df[
            (mode_df["timestamp"] >= arm_time) &
            (mode_df["timestamp"] <= end_time)
        ].copy().reset_index(drop=True)

        modes = flight_modes["mode_name"].values
        times = flight_modes["timestamp"].values

        for i in range(1, len(modes)):
            prev_mode = str(modes[i - 1]).upper()
            curr_mode = str(modes[i]).upper()
            ts = float(times[i])

            if prev_mode in _FW_MODES and curr_mode in _Q_MODES:
                transitions.append({
                    "type":        "fw_to_vtol",
                    "timestamp_s": round(ts, 3),
                    "from_mode":   prev_mode,
                    "to_mode":     curr_mode,
                })
            elif prev_mode in _Q_MODES and curr_mode in _FW_MODES:
                transitions.append({
                    "type":        "vtol_to_fw",
                    "timestamp_s": round(ts, 3),
                    "from_mode":   prev_mode,
                    "to_mode":     curr_mode,
                })

    # ── 2. RCOU-based detection (within-FW-mode transitions) ─────────────
    # QuadPlane AUTO missions start with a VTOL takeoff, then transition to
    # fixed-wing cruise without logging a MODE change. Detect this by watching
    # lift motor outputs go from active to idle within a FW mode.
    if rcou_df is not None and lift_channels:
        rcou_trans = _detect_rcou_transitions(
            rcou_df, arm_time, end_time, lift_channels, transitions,
            mode_df=mode_df,
        )
        transitions.extend(rcou_trans)
        transitions.sort(key=lambda t: t["timestamp_s"])

    return transitions


def _detect_rcou_transitions(
    rcou_df: pd.DataFrame,
    arm_time: float,
    end_time: float,
    lift_channels: List[str],
    existing: List[Dict],
    mode_df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Detect VTOL→FW transitions by watching lift motors go active→idle
    within a fixed-wing mode (no corresponding MODE message).

    Only flags a transition if the aircraft is in a FW mode at the detected
    timestamp — this prevents the motor shutdown during landing (QRTL) from
    being misclassified as a cruise transition.
    """
    available = [c for c in lift_channels if c in rcou_df.columns]
    if not available:
        return []

    flight_rcou = rcou_df[
        (rcou_df["timestamp"] >= arm_time) &
        (rcou_df["timestamp"] <= end_time)
    ].copy().reset_index(drop=True)

    if len(flight_rcou) < 10:
        return []

    ts_arr   = flight_rcou["timestamp"].values.astype(float)
    pwm      = np.column_stack([flight_rcou[c].values.astype(float) for c in available])
    mean_pct = np.clip(np.mean((pwm - 1000.0) / 10.0, axis=1), 0.0, 100.0)

    existing_ts = [t["timestamp_s"] for t in existing]
    transitions: List[Dict] = []

    # Pre-build a mode lookup: for any timestamp, what mode are we in?
    # (used to reject transitions that happen in Q-modes like QRTL landing)
    mode_ts: Optional[np.ndarray] = None
    mode_names: Optional[np.ndarray] = None
    if mode_df is not None and not mode_df.empty and "mode_name" in mode_df.columns:
        _m = mode_df[
            (mode_df["timestamp"] >= arm_time) &
            (mode_df["timestamp"] <= end_time)
        ].sort_values("timestamp")
        if len(_m) > 0:
            mode_ts    = _m["timestamp"].values.astype(float)
            mode_names = _m["mode_name"].values.astype(str)

    def _mode_at(t: float) -> str:
        """Return the active mode name at timestamp t."""
        if mode_ts is None or len(mode_ts) == 0:
            return "UNKNOWN"
        idx = int(np.searchsorted(mode_ts, t, side="right")) - 1
        idx = max(0, min(idx, len(mode_names) - 1))
        return str(mode_names[idx]).upper()

    # State machine: find active→idle crossings
    in_active      = mean_pct[0] >= ACTIVE_THRESH_PCT
    ramp_start_idx = 0

    for i in range(1, len(mean_pct)):
        if not in_active and mean_pct[i] >= ACTIVE_THRESH_PCT:
            in_active      = True
            ramp_start_idx = i

        elif in_active and mean_pct[i] < IDLE_THRESH_PCT:
            # Motors went idle — find when ramp-down began (first sample < RAMP_DETECT_THRESH)
            trans_idx = i
            for j in range(i - 1, ramp_start_idx, -1):
                if mean_pct[j] > RAMP_DETECT_THRESH:
                    trans_idx = j + 1
                    break

            trans_ts = float(ts_arr[trans_idx])

            # Only flag if we are in a FW mode (not QRTL / QLAND landing)
            current_mode = _mode_at(trans_ts)
            in_fw_mode = current_mode in _FW_MODES

            # Skip if already captured by a mode change within ±30 s
            already_captured = any(abs(trans_ts - et) < 30.0 for et in existing_ts)

            if in_fw_mode and not already_captured:
                transitions.append({
                    "type":        "vtol_to_fw",
                    "timestamp_s": round(trans_ts, 3),
                    "from_mode":   "AUTO",   # within AUTO — no mode change logged
                    "to_mode":     "AUTO",
                })

            in_active = False

    return transitions


# ─── Per-transition analysis ───────────────────────────────────────────────────

def _analyse_transition(
    tr: Dict,
    baro_df: Optional[pd.DataFrame],
    att_df:  Optional[pd.DataFrame],
    rcou_df: Optional[pd.DataFrame],
    lift_channels: List[str],
    hover_thr_pct: float,
) -> Dict:
    """
    Analyse a single transition event.

    Returns a record dict including a 'score' key and an 'issues' list
    (issues are hoisted by the caller to module level).
    """
    ts        = tr["timestamp_s"]
    tr_type   = tr["type"]
    issues: List[Dict] = []
    score     = 100.0

    record: Dict[str, Any] = {
        "type":            tr_type,
        "timestamp_s":     ts,
        "from_mode":       tr["from_mode"],
        "to_mode":         tr["to_mode"],
        "alt_drop_m":      None,
        "pitch_spike_deg": None,
        "roll_spike_deg":  None,
        "ramp_time_s":     None,
        "score":           100.0,
        "issues":          [],
    }

    # ── 1. Altitude hold check ────────────────────────────────────────────
    # Skip for landing-type transitions (QRTL, QLAND) — descent is intentional
    _skip_alt = tr["to_mode"] in _LANDING_MODES

    if not _skip_alt and baro_df is not None and not baro_df.empty and "Alt" in baro_df.columns:
        alt_col = "Alt"
        window = baro_df[
            (baro_df["timestamp"] >= ts) &
            (baro_df["timestamp"] <= ts + ANALYSIS_WINDOW_S)
        ]["Alt"].values.astype(float)

        # Baseline: altitude at the moment of transition
        at_ts = baro_df[baro_df["timestamp"] >= ts]
        if not at_ts.empty and len(window) >= 3:
            base_alt = float(at_ts.iloc[0]["Alt"])
            alt_drop = float(base_alt - np.min(window))   # positive = drone dropped
            record["alt_drop_m"] = round(alt_drop, 2)

            if alt_drop > ALT_LOSS_CRIT_M:
                score -= 30.0
                issues.append(_issue(
                    "critical", "VTR-003",
                    f"Severe altitude loss during {tr_type.replace('_', ' ')} "
                    f"at T+{ts:.1f} s: {alt_drop:.1f} m drop "
                    f"(threshold {ALT_LOSS_CRIT_M} m). "
                    "Check transition parameters and airspeed settings.",
                    value=round(alt_drop, 2),
                    threshold=ALT_LOSS_CRIT_M,
                    timestamp=ts,
                ))
            elif alt_drop > ALT_LOSS_WARN_M:
                score -= 15.0
                issues.append(_issue(
                    "warning", "VTR-002",
                    f"Altitude loss during {tr_type.replace('_', ' ')} "
                    f"at T+{ts:.1f} s: {alt_drop:.1f} m drop "
                    f"(threshold {ALT_LOSS_WARN_M} m). "
                    "Review ARSPD_FBW_MIN and transition airspeed settings.",
                    value=round(alt_drop, 2),
                    threshold=ALT_LOSS_WARN_M,
                    timestamp=ts,
                ))

    # ── 2. Attitude spike check ───────────────────────────────────────────
    if att_df is not None and not att_df.empty:
        roll_col  = _find_col(att_df, ["Roll",  "roll"])
        pitch_col = _find_col(att_df, ["Pitch", "pitch"])
        des_roll  = _find_col(att_df, ["DesRoll",  "DeRoll"])
        des_pitch = _find_col(att_df, ["DesPitch", "DePitch"])

        if roll_col and pitch_col:
            # Baseline: mean attitude 3 s before transition
            pre = att_df[
                (att_df["timestamp"] >= ts - ATTITUDE_BASELINE_S) &
                (att_df["timestamp"] <  ts)
            ]
            post = att_df[
                (att_df["timestamp"] >= ts) &
                (att_df["timestamp"] <= ts + ATTITUDE_WINDOW_S)
            ]

            if len(pre) >= 3 and len(post) >= 3:
                base_roll  = float(pre[roll_col].mean())
                base_pitch = float(pre[pitch_col].mean())

                roll_err  = np.abs(post[roll_col].values.astype(float)  - base_roll)
                pitch_err = np.abs(post[pitch_col].values.astype(float) - base_pitch)

                max_roll  = float(np.max(roll_err))
                max_pitch = float(np.max(pitch_err))

                record["roll_spike_deg"]  = round(max_roll, 2)
                record["pitch_spike_deg"] = round(max_pitch, 2)

                worst_spike = max(max_roll, max_pitch)
                spike_axis  = "roll" if max_roll >= max_pitch else "pitch"

                if worst_spike > PITCH_SPIKE_CRIT_DEG:
                    score -= 20.0
                    issues.append(_issue(
                        "critical", "VTR-005",
                        f"Severe attitude spike during transition at T+{ts:.1f} s: "
                        f"{worst_spike:.1f} deg {spike_axis} "
                        f"(threshold {PITCH_SPIKE_CRIT_DEG} deg). "
                        "Review VTOL transition gains (Q_TRANSITION_MS).",
                        value=round(worst_spike, 2),
                        threshold=PITCH_SPIKE_CRIT_DEG,
                        timestamp=ts,
                    ))
                elif worst_spike > PITCH_SPIKE_WARN_DEG:
                    score -= 10.0
                    issues.append(_issue(
                        "warning", "VTR-004",
                        f"Attitude spike during transition at T+{ts:.1f} s: "
                        f"{worst_spike:.1f} deg {spike_axis} "
                        f"(threshold {PITCH_SPIKE_WARN_DEG} deg). "
                        "Consider tuning VTOL transition parameters.",
                        value=round(worst_spike, 2),
                        threshold=PITCH_SPIKE_WARN_DEG,
                        timestamp=ts,
                    ))

    # ── 3. Lift motor ramp check ──────────────────────────────────────────
    if rcou_df is not None and not rcou_df.empty and lift_channels:
        available_lift = [c for c in lift_channels if c in rcou_df.columns]

        if available_lift:
            ramp_window = rcou_df[
                (rcou_df["timestamp"] >= ts) &
                (rcou_df["timestamp"] <= ts + 30.0)   # 30 s max ramp window
            ].copy().reset_index(drop=True)

            if len(ramp_window) >= 5:
                ramp_ts = ramp_window["timestamp"].values
                # Convert PWM to percent: (pwm - 1000) / 10
                motor_data = np.column_stack([
                    (ramp_window[c].values.astype(float) - 1000.0) / 10.0
                    for c in available_lift
                ])
                mean_throttle = np.mean(motor_data, axis=1)

                ramp_time_s = None

                if tr_type == "fw_to_vtol":
                    # Lift motors should ramp UP to hover throttle
                    target = hover_thr_pct * HOVER_RAMP_THRESH
                    reached = np.where(mean_throttle >= target)[0]
                    if len(reached) > 0:
                        ramp_time_s = float(ramp_ts[reached[0]] - ts)
                    # If never reached: ramp_time_s stays None (no ramp issue flagged)

                elif tr_type == "vtol_to_fw":
                    # Lift motors should ramp DOWN to idle
                    reached = np.where(mean_throttle <= IDLE_THRESH_PCT)[0]
                    if len(reached) > 0:
                        ramp_time_s = float(ramp_ts[reached[0]] - ts)

                record["ramp_time_s"] = round(ramp_time_s, 2) if ramp_time_s is not None else None

                if ramp_time_s is not None:
                    ramp_dir = "spool-up" if tr_type == "fw_to_vtol" else "spool-down"
                    if ramp_time_s > RAMP_CRIT_S:
                        score -= 25.0
                        issues.append(_issue(
                            "critical", "VTR-007",
                            f"Very slow lift motor {ramp_dir} during transition "
                            f"at T+{ts:.1f} s: {ramp_time_s:.1f} s "
                            f"(threshold {RAMP_CRIT_S} s). "
                            "Possible ESC/motor fault or excessive Q_TRAN_FAIL_MS setting.",
                            value=round(ramp_time_s, 2),
                            threshold=RAMP_CRIT_S,
                            timestamp=ts,
                        ))
                    elif ramp_time_s > RAMP_WARN_S:
                        score -= 15.0
                        issues.append(_issue(
                            "warning", "VTR-006",
                            f"Slow lift motor {ramp_dir} during transition "
                            f"at T+{ts:.1f} s: {ramp_time_s:.1f} s "
                            f"(threshold {RAMP_WARN_S} s). "
                            "Check ESC calibration and motor responsiveness.",
                            value=round(ramp_time_s, 2),
                            threshold=RAMP_WARN_S,
                            timestamp=ts,
                        ))

    record["score"]  = round(max(0.0, score), 1)
    record["issues"] = issues
    return record


# ─── Result builder ───────────────────────────────────────────────────────────

def _build_result(
    score: float,
    issues: List[Dict],
    metrics: Dict[str, Any],
    summary: str,
) -> Dict[str, Any]:
    score = max(0.0, min(100.0, score))
    from analyzer.scoring.engine import _grade
    grade = _grade(score)
    return {
        "score":     round(score, 1),
        "grade":     grade,
        "available": True,
        "issues":    issues,
        "metrics":   metrics,
        "summary":   summary,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
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
