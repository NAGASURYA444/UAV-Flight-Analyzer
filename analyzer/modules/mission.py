"""
Mission Execution Quality Module
==================================
Phase 4.4 — Waypoint completion and cross-track accuracy analysis

Checks performed
----------------
1.  Waypoint completion rate   — how many CMD nav-waypoints were reached
2.  Cross-track error (XTE)    — GPS deviation from the straight-line path
                                  between consecutive waypoints

Availability
------------
Module returns available=False (score 100, no issues) when:
  - No CMD data in the log
  - No AUTO mode flown (mission was not executed)
  - Fewer than 2 nav-waypoints found (no path to measure XTE against)
  - GPS data unavailable

Issue codes
-----------
  MSN-001  Info — mission partially flown (< 50% of plan) — operational, no score deduction
  MSN-002  Info — mission partially flown (50-80% of plan) — operational, no score deduction
  MSN-003  Mean cross-track error > 20 m (critical)
  MSN-004  Mean cross-track error 10-20 m (warning)
  MSN-005  Max cross-track error > 50 m (warning)

Scoring (from 100)
------------------
  Score is based ONLY on navigation accuracy (cross-track error).
  Waypoint completion is informational — operator may cut mission short
  intentionally (weather, battery conservation, recall) so it must NOT
  penalise the flight health score.

  -20   mean XTE > 20 m
  -10   mean XTE 10-20 m
   -5   max XTE > 50 m
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analyzer.config.drone_profile import DroneProfile
from analyzer.parser.bin_parser import ParseResult

logger = logging.getLogger(__name__)

# ── NAV command IDs that represent waypoint destinations ─────────────────────
# ArduPilot MAVLink command IDs (MAV_CMD):
#   16 = NAV_WAYPOINT
#   17 = NAV_LOITER_UNLIM
#   18 = NAV_LOITER_TURNS
#   19 = NAV_LOITER_TIME
#   20 = NAV_RETURN_TO_LAUNCH
#   21 = NAV_LAND
#   22 = NAV_TAKEOFF
NAV_COMMAND_IDS = {16, 17, 18, 19, 20, 21, 22}

# ArduPilot mode names that indicate autonomous mission execution
AUTO_MODE_NAMES = {"AUTO", "GUIDED"}

# Cross-track thresholds
XTE_WARN_M   = 10.0   # m — mean cross-track warning
XTE_CRIT_M   = 20.0   # m — mean cross-track critical
XTE_MAX_M    = 50.0   # m — peak cross-track warning

# Completion thresholds
COMPLETION_WARN_PCT = 80.0   # % waypoints reached — warning below this
COMPLETION_CRIT_PCT = 50.0   # % waypoints reached — critical below this


def analyse(
    result: ParseResult,
    profile: DroneProfile,
    arm_time: float = 0.0,
    disarm_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Mission Execution Quality — Phase 4.4."""

    issues:  List[Dict] = []
    metrics: Dict[str, Any] = {}

    def _unavailable(reason: str) -> Dict[str, Any]:
        return {
            "score": 100.0,
            "grade": "A",
            "available": False,
            "issues": [],
            "metrics": {"unavailable_reason": reason},
            "summary": f"Mission analysis: {reason}",
        }

    # ── Data retrieval ────────────────────────────────────────────────────────
    cmd_df  = result.get("CMD")
    mode_df = result.get("MODE")
    gps_df  = result.get("GPS")

    if cmd_df is None or cmd_df.empty:
        return _unavailable("No CMD (mission command) data in this log.")

    if gps_df is None or gps_df.empty:
        return _unavailable("No GPS data in this log.")

    # ── Check AUTO mode was flown ─────────────────────────────────────────────
    auto_flown = False
    if mode_df is not None and not mode_df.empty and "mode_name" in mode_df.columns:
        auto_flown = mode_df["mode_name"].isin(AUTO_MODE_NAMES).any()
    if not auto_flown:
        return _unavailable("No AUTO/GUIDED mode flown — mission not executed.")

    # ── Extract nav waypoints from CMD ────────────────────────────────────────
    # CMD columns: TimeUS, CTot, CNum, CId, Prm1..Prm4, Lat, Lng, Alt
    #
    # ArduPilot logs CMD messages both during pre-flight mission upload AND
    # when each item is executed in flight.  We deduplicate by CNum (keep the
    # first occurrence per sequence number) to get the planned mission items.
    nav_cols_needed = {"CId", "Lat", "Lng", "CNum"}
    if not nav_cols_needed.issubset(cmd_df.columns):
        return _unavailable("CMD message missing required columns (CId, Lat, Lng, CNum).")

    # Filter to nav commands with valid coordinates, then deduplicate by CNum
    all_nav = cmd_df[cmd_df["CId"].isin(NAV_COMMAND_IDS)].copy()
    all_nav = all_nav[(all_nav["Lat"].notna()) & (all_nav["Lng"].notna())]
    all_nav = all_nav[(all_nav["Lat"].abs() > 0.001) & (all_nav["Lng"].abs() > 0.001)]

    if "timestamp" in all_nav.columns:
        all_nav = all_nav.sort_values("timestamp")
    # One row per unique CNum — this is the planned mission
    nav_df = all_nav.drop_duplicates(subset="CNum", keep="first").reset_index(drop=True)

    # Some ArduPilot firmware versions log CMD Lat/Lng as integer degrees × 1e7
    # (e.g., 123456789 instead of 12.3456789).  Detect and normalise.
    if len(nav_df) > 0 and nav_df["Lat"].abs().max() > 90:
        nav_df = nav_df.copy()
        nav_df["Lat"] = nav_df["Lat"] / 1e7
        nav_df["Lng"] = nav_df["Lng"] / 1e7

    if len(nav_df) < 2:
        return _unavailable("Fewer than 2 valid nav-waypoints in mission — cannot compute path.")

    # ── Waypoint completion count ─────────────────────────────────────────────
    # ArduPilot logs CMD messages BOTH during pre-flight mission upload (all
    # entries within milliseconds of each other) AND when each item is
    # executed in flight (spaced seconds to minutes apart).
    #
    # Strategy: find in-flight CMD entries by looking for the first large time
    # gap (>= 10 s) between consecutive CMD entries in the armed window.
    # Only CMD entries AFTER that first large gap are counted as executed.
    # This reliably separates upload bursts from real in-flight executions.
    total_wpts = len(nav_df)
    planned_cnums = set(nav_df["CNum"].unique())
    # Default to unknown (0 confirmed) — only set to a real value when we have evidence
    executed_cnums = planned_cnums   # used for leg filtering; stays full set when unverifiable
    executed_wpts = None             # None = unverifiable (distinct from 0 = confirmed none reached)

    if "CNum" in cmd_df.columns:
        armed_cmd = cmd_df.copy()
        if "timestamp" in armed_cmd.columns:
            armed_cmd = armed_cmd[armed_cmd["timestamp"] >= arm_time]
            if disarm_time is not None:
                armed_cmd = armed_cmd[armed_cmd["timestamp"] <= disarm_time]

        if not armed_cmd.empty:
            ts_sorted = armed_cmd.sort_values("timestamp")
            ts_vals = ts_sorted["timestamp"].values
            if len(ts_vals) >= 2:
                gaps = np.diff(ts_vals)
                # An upload burst = multiple CMD entries within < 1 s of each other.
                # This happens when the GCS uploads the entire mission in bulk before arming.
                # In that case, the first large gap (>= 10 s) separates the burst from the
                # first true in-flight execution.
                # If there is NO burst pattern (all entries well-spaced), all entries are
                # individual in-flight executions — count them all.
                has_burst = bool(np.any(gaps < 1.0))

                if has_burst:
                    large_gap_mask = gaps >= 10.0
                    if large_gap_mask.any():
                        split_pos = int(np.argmax(large_gap_mask)) + 1
                        inflight_cmd = ts_sorted.iloc[split_pos:]
                    else:
                        # All CMDs are within a tight burst — no in-flight execution found
                        inflight_cmd = ts_sorted.iloc[0:0]  # empty
                else:
                    # No burst — all CMDs are individual in-flight executions
                    inflight_cmd = ts_sorted

                nav_executed = inflight_cmd[inflight_cmd["CId"].isin(NAV_COMMAND_IDS)]
                executed_cnums = set(nav_executed["CNum"].unique()) & planned_cnums
                executed_wpts = len(executed_cnums)
            else:
                # Only 1 CMD row in the armed window — not enough to detect burst vs execution.
                # Check if that single row is itself a NAV command before counting it.
                single_row = ts_sorted.iloc[[0]]
                nav_single = single_row[single_row["CId"].isin(NAV_COMMAND_IDS)]
                single_cnums = set(nav_single["CNum"].unique()) & planned_cnums
                executed_wpts = len(single_cnums)  # 0 if non-NAV, 1 only if it's a real NAV cmd
        else:
            executed_wpts = 0

    # completion_pct: None when unverifiable, else evidence-based percentage
    if executed_wpts is None:
        completion_pct = None
        mission_complete = None
    else:
        completion_pct = (executed_wpts / total_wpts * 100.0) if total_wpts > 0 else 100.0
        mission_complete = (executed_wpts >= total_wpts)

    metrics["total_waypoints"]    = total_wpts
    metrics["executed_waypoints"] = executed_wpts if executed_wpts is not None else "unknown"
    metrics["completion_pct"]     = round(completion_pct, 1) if completion_pct is not None else None
    metrics["mission_complete"]   = mission_complete

    # ── Cross-track error analysis ────────────────────────────────────────────
    # XTE is only meaningful when the autopilot is actively following the mission.
    # Restrict GPS to AUTO/GUIDED mode segments only — manual/FBWA segments
    # are excluded because the pilot is deliberately flying a different path.
    gps_armed = gps_df.copy()
    if "timestamp" in gps_armed.columns:
        gps_armed = gps_armed[gps_armed["timestamp"] >= arm_time]
        if disarm_time is not None:
            gps_armed = gps_armed[gps_armed["timestamp"] <= disarm_time]

    # Filter to AUTO/GUIDED mode windows only
    if mode_df is not None and not mode_df.empty and "mode_name" in mode_df.columns:
        gps_armed = _filter_auto_only(gps_armed, mode_df)

    # Need Lat/Lng columns in GPS
    gps_lat_col = next((c for c in ("Lat", "lat", "GPS_Lat") if c in gps_armed.columns), None)
    gps_lng_col = next((c for c in ("Lng", "lng", "Lon", "lon", "GPS_Lng") if c in gps_armed.columns), None)

    xte_values: List[float] = []

    if gps_lat_col and gps_lng_col and len(nav_df) >= 2:
        # Build exec_nav: only the waypoints the aircraft actually flew, in CNum order.
        # This is critical for missions with DO_JUMP or non-sequential execution —
        # the first N rows of nav_df are NOT necessarily the flown waypoints.
        # When executed_cnums == planned_cnums (default / unknown), use all nav_df rows.
        if executed_cnums and executed_cnums != planned_cnums and len(executed_cnums) >= 2:
            exec_nav = nav_df[nav_df["CNum"].isin(executed_cnums)].sort_values("CNum").reset_index(drop=True)
            # Also append the next unexecuted waypoint (the target when RTL fired)
            last_exec_cnum = int(exec_nav["CNum"].max())
            next_rows = nav_df[nav_df["CNum"] > last_exec_cnum]
            if not next_rows.empty:
                exec_nav = pd.concat([exec_nav, next_rows.iloc[[0]]]).reset_index(drop=True)
        else:
            exec_nav = nav_df.sort_values("CNum").reset_index(drop=True)

        # ── Prepend a synthetic WP0 (aircraft position at AUTO start) ─────────
        # Without this, GPS samples from the transit phase (aircraft approaching
        # WP1 from home before it has started tracking the WP1→WP2 leg) have
        # t ≈ 0 on the WP1→WP2 leg and a large perpendicular distance, which
        # inflates the mean XTE significantly.  Adding the home/arm position as
        # WP0 creates a valid approach leg (WP0→WP1) so transit samples are
        # attributed there with low XTE, not to the first survey leg.
        if mode_df is not None and not mode_df.empty and "mode_name" in mode_df.columns:
            mode_sorted_tmp = mode_df.sort_values("timestamp").reset_index(drop=True)
            auto_mask_tmp = mode_sorted_tmp["mode_name"].isin(AUTO_MODE_NAMES)
            if auto_mask_tmp.any() and "timestamp" in gps_df.columns:
                first_auto_ts = float(mode_sorted_tmp[auto_mask_tmp].iloc[0]["timestamp"])
                # GPS sample closest to (but not after) AUTO start
                pre_auto_gps = gps_df[
                    (gps_df["timestamp"] <= first_auto_ts) &
                    (gps_df[gps_lat_col].notna()) & (gps_df[gps_lng_col].notna()) &
                    (gps_df[gps_lat_col].abs() > 0.001) & (gps_df[gps_lng_col].abs() > 0.001)
                ]
                if not pre_auto_gps.empty:
                    gps_row0 = pre_auto_gps.iloc[-1]
                    wp0_lat  = float(gps_row0[gps_lat_col])
                    wp0_lng  = float(gps_row0[gps_lng_col])
                    first_cnum = int(exec_nav["CNum"].min()) if "CNum" in exec_nav.columns else 1
                    # Only prepend if WP0 is meaningfully far from WP1 (> 5 m)
                    wp1_lat = exec_nav.iloc[0]["Lat"]
                    wp1_lng = exec_nav.iloc[0]["Lng"]
                    if _haversine(wp0_lat, wp0_lng, wp1_lat, wp1_lng) >= 5.0:
                        wp0_row = pd.DataFrame([{
                            "CNum": first_cnum - 1,
                            "CId":  16,
                            "Lat":  wp0_lat,
                            "Lng":  wp0_lng,
                        }])
                        exec_nav = pd.concat(
                            [wp0_row, exec_nav], ignore_index=True
                        ).reset_index(drop=True)

        wpt_lats = exec_nav["Lat"].values
        wpt_lngs = exec_nav["Lng"].values

        gps_lats = gps_armed[gps_lat_col].values
        gps_lngs = gps_armed[gps_lng_col].values

        # Build list of valid legs (skip legs shorter than 5 m)
        legs = []
        leg_labels = []
        for i in range(len(wpt_lats) - 1):
            a_lat, a_lng = wpt_lats[i],     wpt_lngs[i]
            b_lat, b_lng = wpt_lats[i + 1], wpt_lngs[i + 1]
            # Use positional index (WP0 = synthetic home, WP1..WPN = nav waypoints in order)
            # so labels are consistent with total_waypoints count shown in the report.
            # CNum (CMD sequence number) is NOT used because it counts all command types
            # (DO_CHANGE_SPEED, DO_SET_ROI, etc.), making labels like WP192 appear even
            # when the mission only has 158 nav waypoints.
            if _haversine(a_lat, a_lng, b_lat, b_lng) >= 5.0:
                legs.append((a_lat, a_lng, b_lat, b_lng))
                leg_labels.append(f"WP{i}->WP{i + 1}")

        # Parallel lists to record timestamp and best-leg index per XTE sample
        xte_timestamps: List[Optional[float]] = []
        xte_best_legs:  List[Optional[int]]   = []
        gps_ts_vals = gps_armed["timestamp"].values if "timestamp" in gps_armed.columns else None

        # For each GPS sample, compute the MINIMUM XTE across all legs.
        # This correctly handles lawnmower survey patterns where a GPS point
        # is equidistant from multiple parallel legs.
        for j in range(len(gps_lats)):
            p_lat = gps_lats[j]
            p_lng = gps_lngs[j]
            min_xte    = None
            best_leg_k = None
            for k, (a_lat, a_lng, b_lat, b_lng) in enumerate(legs):
                xte = _cross_track_error(a_lat, a_lng, b_lat, b_lng, p_lat, p_lng)
                if xte is not None:
                    if min_xte is None or xte < min_xte:
                        min_xte    = xte
                        best_leg_k = k
            if min_xte is not None:
                xte_values.append(min_xte)
                xte_timestamps.append(float(gps_ts_vals[j]) if gps_ts_vals is not None else None)
                xte_best_legs.append(best_leg_k)

    if xte_values:
        mean_xte    = float(np.mean(xte_values))
        max_xte     = float(np.max(xte_values))
        argmax_idx  = int(np.argmax(xte_values))
        max_xte_ts  = xte_timestamps[argmax_idx] if xte_timestamps else None
        max_xte_leg_idx = xte_best_legs[argmax_idx] if xte_best_legs else None
        max_xte_leg_label = (
            leg_labels[max_xte_leg_idx]
            if (max_xte_leg_idx is not None and max_xte_leg_idx < len(leg_labels))
            else "—"
        )
        metrics["mean_cross_track_m"]  = round(mean_xte, 1)
        metrics["max_cross_track_m"]   = round(max_xte, 1)
        metrics["xte_sample_count"]    = len(xte_values)
        metrics["max_xte_timestamp_s"] = round(max_xte_ts, 1) if max_xte_ts is not None else None
        metrics["max_xte_leg"]         = max_xte_leg_label
    else:
        mean_xte = None
        max_xte  = None
        metrics["mean_cross_track_m"]  = None
        metrics["max_cross_track_m"]   = None
        metrics["max_xte_timestamp_s"] = None
        metrics["max_xte_leg"]         = None

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Waypoint completion is an OPERATIONAL metric (operator may cut the mission
    # short intentionally — weather, battery conservation, recall, etc.).
    # It is logged as INFO only and does NOT affect the flight health score.
    # Score is based purely on navigation accuracy (cross-track error).
    score = 100.0
    deductions: List[str] = []

    # Completion — info only, no score deduction
    if mission_complete is None:
        # CNum column absent — cannot verify completion
        issues.append({
            "severity": "info",
            "code": "MSN-000",
            "message": (
                f"Mission completion unverifiable: CMD log lacks sequence numbers. "
                f"Mission had {total_wpts} planned waypoints."
            ),
        })
    elif not mission_complete:
        if completion_pct < COMPLETION_CRIT_PCT:
            issues.append({
                "severity": "info",
                "code": "MSN-001",
                "message": (
                    f"Mission partially flown: {executed_wpts}/{total_wpts} waypoints "
                    f"({completion_pct:.0f}% of plan). Operational decision or early recall."
                ),
                "value": completion_pct,
            })
        else:
            issues.append({
                "severity": "info",
                "code": "MSN-002",
                "message": (
                    f"Mission partially flown: {executed_wpts}/{total_wpts} waypoints "
                    f"({completion_pct:.0f}% of plan)."
                ),
                "value": completion_pct,
            })

    # Cross-track error penalties
    if mean_xte is not None:
        if mean_xte > XTE_CRIT_M:
            score -= 20
            deductions.append(f"high mean XTE ({mean_xte:.1f} m)")
            issues.append({
                "severity": "critical",
                "code": "MSN-003",
                "message": (
                    f"Mean cross-track error {mean_xte:.1f} m — significantly off route. "
                    f"Check GPS accuracy, wind correction, and navigation parameters."
                ),
                "value": round(mean_xte, 1),
                "threshold": XTE_CRIT_M,
            })
        elif mean_xte > XTE_WARN_M:
            score -= 10
            deductions.append(f"elevated mean XTE ({mean_xte:.1f} m)")
            # Add context for fixed-wing: 10-20 m is generally acceptable at cruise speeds
            fw_context = ""
            if profile.type in ("fixed_wing", "vtol"):
                fw_context = (
                    f" Note: 10-20 m is generally acceptable for fixed-wing cruise surveys "
                    f"at typical speeds — consider tuning only if survey accuracy is critical."
                )
            issues.append({
                "severity": "warning",
                "code": "MSN-004",
                "message": (
                    f"Mean cross-track error {mean_xte:.1f} m — check NAVL1_PERIOD or "
                    f"LOITER_RAD for tighter path tracking.{fw_context}"
                ),
                "value": round(mean_xte, 1),
                "threshold": XTE_WARN_M,
            })

        if max_xte is not None and max_xte > XTE_MAX_M:
            max_ts_s  = metrics.get("max_xte_timestamp_s")
            max_leg   = metrics.get("max_xte_leg", "—")
            loc_str   = ""
            if max_ts_s is not None:
                loc_str += f" at T+{max_ts_s:.1f} s"
            if max_leg and max_leg != "—":
                loc_str += f" on {max_leg}"

            # Detect if the spike is an artifact (RTL transition OR mission start transit).
            # Both cases produce large XTE readings that are not navigation failures.
            rtl_artifact   = False
            start_artifact = False
            rtl_context    = ""
            start_context  = ""
            if max_ts_s is not None and mode_df is not None and not mode_df.empty and "timestamp" in mode_df.columns:
                mode_sorted = mode_df.sort_values("timestamp").reset_index(drop=True)
                auto_mask = mode_sorted["mode_name"].isin(AUTO_MODE_NAMES)
                if auto_mask.any():
                    # --- RTL artifact: spike within 10 s of AUTO mode ending ---
                    last_auto_idx = int(auto_mask[auto_mask].index[-1])
                    if last_auto_idx + 1 < len(mode_sorted):
                        auto_end_s = float(mode_sorted.iloc[last_auto_idx + 1]["timestamp"])
                        if abs(max_ts_s - auto_end_s) <= 10.0:
                            rtl_artifact = True
                            rtl_context = (
                                f" This spike occurred within {abs(auto_end_s - max_ts_s):.1f} s of "
                                f"AUTO mode ending (T+{auto_end_s:.1f} s) — likely an RTL-transition "
                                f"artifact as the aircraft departed the planned route to return home, "
                                f"not a navigation failure during the mission. No score penalty applied."
                            )

                    # --- Start artifact: spike within 30 s of AUTO mode starting ---
                    first_auto_s = float(mode_sorted[auto_mask].iloc[0]["timestamp"])
                    if not rtl_artifact and (max_ts_s - first_auto_s) <= 30.0:
                        start_artifact = True
                        start_context = (
                            f" This spike occurred {max_ts_s - first_auto_s:.1f} s after AUTO mode "
                            f"started (T+{first_auto_s:.1f} s) — likely a start-of-mission positioning "
                            f"artifact while the aircraft was transiting to the first waypoint, "
                            f"not a navigation failure. No score penalty applied."
                        )

            # Only penalise if this is a genuine in-mission deviation
            if not rtl_artifact and not start_artifact:
                score -= 5
                deductions.append(f"max XTE spike ({max_xte:.1f} m)")

            artifact_context = rtl_context or start_context
            is_artifact = bool(rtl_artifact or start_artifact)
            issues.append({
                "severity": "info" if is_artifact else "warning",
                "code": "MSN-005",
                "message": (
                    f"Peak cross-track error {max_xte:.1f} m{loc_str} — large deviation detected. "
                    f"Check for wind gusts, GPS glitches, or navigation mode transitions.{artifact_context}"
                ),
                "value": round(max_xte, 1),
                "threshold": XTE_MAX_M,
                "timestamp_s": max_ts_s,
            })

    score = max(0.0, min(100.0, score))
    grade = _grade(score)

    # ── Summary ───────────────────────────────────────────────────────────────
    if mission_complete is None:
        compl_str = f"{total_wpts} WPTs planned (completion unverifiable)"
    elif mission_complete:
        compl_str = f"{total_wpts}/{total_wpts} WPTs complete"
    else:
        compl_str = f"{executed_wpts}/{total_wpts} WPTs ({completion_pct:.0f}%)"

    xte_str = ""
    if mean_xte is not None:
        xte_str = f"  |  mean XTE {mean_xte:.1f} m"

    summary = f"{compl_str}{xte_str}"
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
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

_EARTH_R_M = 6_371_000.0  # mean Earth radius in metres


def _filter_auto_only(gps_df, mode_df) -> "pd.DataFrame":
    """Return only GPS rows that fall within AUTO or GUIDED mode segments."""
    if "timestamp" not in gps_df.columns or "timestamp" not in mode_df.columns:
        return gps_df

    import numpy as np
    mode_sorted = mode_df.sort_values("timestamp").reset_index(drop=True)
    mode_ts    = mode_sorted["timestamp"].values
    mode_names = mode_sorted["mode_name"].values if "mode_name" in mode_sorted.columns else None
    if mode_names is None:
        return gps_df

    gps_ts  = gps_df["timestamp"].values
    indices = np.searchsorted(mode_ts, gps_ts, side="right") - 1
    indices = np.clip(indices, 0, len(mode_names) - 1)
    active  = mode_names[indices]
    mask    = np.isin(active, list(AUTO_MODE_NAMES))
    return gps_df[mask].reset_index(drop=True)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(math.sqrt(min(a, 1.0)))


def _cross_track_error(
    a_lat: float, a_lng: float,
    b_lat: float, b_lng: float,
    p_lat: float, p_lng: float,
) -> Optional[float]:
    """
    Return the perpendicular cross-track distance (metres) from point P to
    the great-circle segment A→B.  Returns None if the nearest point on the
    segment is outside A–B (i.e. the perpendicular foot lies beyond the endpoints).
    """
    try:
        # Convert to flat-earth local coords (metres) centred on A
        # Good approximation for distances < 50 km
        ref_lat = math.radians(a_lat)

        def _to_xy(lat: float, lng: float):
            x = math.radians(lng - a_lng) * _EARTH_R_M * math.cos(ref_lat)
            y = math.radians(lat - a_lat) * _EARTH_R_M
            return x, y

        ax, ay = 0.0, 0.0
        bx, by = _to_xy(b_lat, b_lng)
        px, py = _to_xy(p_lat, p_lng)

        # Vector AB
        abx = bx - ax
        aby = by - ay
        ab_len = math.hypot(abx, aby)
        if ab_len < 1e-3:
            return None

        # Project P onto AB
        t = ((px - ax) * abx + (py - ay) * aby) / (ab_len * ab_len)

        # Only count the sample if P is between A and B (t in [0, 1])
        if t < 0.0 or t > 1.0:
            return None

        # Closest point on segment
        cx = ax + t * abx
        cy = ay + t * aby

        return math.hypot(px - cx, py - cy)
    except Exception:
        return None


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"
