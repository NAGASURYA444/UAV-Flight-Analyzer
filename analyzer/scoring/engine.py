"""
Scoring Engine
==============
Aggregates individual module scores into a single Flight Health Score,
determines a verdict, and builds a prioritised issue log.

Score tiers
-----------
90 – 100  : A  EXCELLENT  — No action needed
75 –  89  : B  GOOD       — Minor observations
60 –  74  : C  FAIR       — Attention recommended
40 –  59  : D  POOR       — Maintenance required before next flight
 0 –  39  : F  CRITICAL   — Grounded until inspected

The weighted sum is computed from available modules only (unavailable
modules are excluded and weights are renormalised so the total stays 100).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from analyzer.config.drone_profile import DroneProfile

logger = logging.getLogger(__name__)


VERDICTS: Dict[str, Dict[str, str]] = {
    "A": {
        "label": "EXCELLENT",
        "action": "No maintenance action required. Drone is in optimal condition.",
        "color": "green",
    },
    "B": {
        "label": "GOOD",
        "action": "Minor observations noted. Monitor on next flight.",
        "color": "bright_green",
    },
    "C": {
        "label": "FAIR",
        "action": "Attention recommended. Address warnings before prolonged operations.",
        "color": "yellow",
    },
    "D": {
        "label": "POOR",
        "action": "Maintenance required before next flight. Review all warnings and critical issues.",
        "color": "orange3",
    },
    "F": {
        "label": "CRITICAL",
        "action": "DRONE GROUNDED. Do not fly until all critical issues are resolved and inspected.",
        "color": "red",
    },
}

SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


def compute(
    module_results: Dict[str, Dict[str, Any]],
    profile: DroneProfile,
) -> Dict[str, Any]:
    """
    Compute the overall flight health score.

    Parameters
    ----------
    module_results : dict of {module_name: result_dict}
        Each result_dict must have keys: score, grade, available, issues, metrics, summary
    profile        : DroneProfile — provides scoring weights

    Returns
    -------
    dict with keys:
        overall_score    : float  (0-100)
        grade            : str    (A/B/C/D/F)
        verdict          : str    (EXCELLENT / GOOD / FAIR / POOR / CRITICAL)
        action           : str    (recommended action)
        module_scores    : dict   {module_name: {score, grade, available}}
        all_issues       : list   — merged, deduplicated, sorted by severity
        issue_summary    : dict   {critical: N, warning: N, info: N}
        summary_lines    : list   — one-line summaries per module
        recommendations  : list   — top actionable recommendations
    """
    weights = dict(profile.scoring_weights)

    # ── Build module score table ──────────────────────────────────────────────
    module_scores: Dict[str, Dict] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for module_name, result in module_results.items():
        available = result.get("available", True)
        score = result.get("score", 50.0)
        grade = result.get("grade", _grade(score))

        module_scores[module_name] = {
            "score": round(score, 1),
            "grade": grade,
            "available": available,
            "summary": result.get("summary", ""),
        }

        if not available:
            logger.debug("Module '%s' unavailable — excluded from weighted score.", module_name)
            continue

        weight = weights.get(module_name, 0.0)
        weighted_sum += score * weight
        total_weight += weight

    # Renormalise if some modules were unavailable
    if total_weight > 0:
        overall_score = weighted_sum / total_weight
    else:
        overall_score = 50.0   # fallback if no modules available

    overall_score = max(0.0, min(100.0, overall_score))

    # Hard penalty: any module with grade F applies an additional flat -5 per module
    # on top of its weighted contribution.  This is intentional — a catastrophic
    # failure (e.g. dead motor, critical battery discharge) should pull the overall
    # score down more than its weight alone would suggest, so operators notice it.
    # NOTE: this means the overall score is NOT a simple weighted average when any
    # module grades F; the per-module scores alone cannot predict the final total.
    f_modules = [n for n, s in module_scores.items() if s["grade"] == "F" and s["available"]]
    if f_modules:
        penalty = len(f_modules) * 5.0
        overall_score = max(0.0, overall_score - penalty)

    grade = _grade(overall_score)
    verdict_info = VERDICTS[grade]

    # ── Merge all issues ──────────────────────────────────────────────────────
    all_issues: List[Dict] = []
    for module_name, result in module_results.items():
        for issue in result.get("issues", []):
            enriched = dict(issue)
            enriched["module"] = module_name
            all_issues.append(enriched)

    # Sort: critical first, then warning, then info; within each group by module
    all_issues.sort(key=lambda x: (
        SEVERITY_RANK.get(x.get("severity", "info"), 2),
        x.get("module", ""),
    ))

    # Issue counts
    issue_summary = {
        "critical": sum(1 for i in all_issues if i.get("severity") == "critical"),
        "warning": sum(1 for i in all_issues if i.get("severity") == "warning"),
        "info": sum(1 for i in all_issues if i.get("severity") == "info"),
        "total": len(all_issues),
    }

    # ── Build recommendations ─────────────────────────────────────────────────
    recommendations = _build_recommendations(all_issues, module_scores)

    # ── Summary lines ─────────────────────────────────────────────────────────
    summary_lines = [
        f"{name.upper():20s}  [{info['grade']}] {info['score']:5.1f}/100  "
        + ("(data unavailable)" if not info["available"] else info.get("summary", ""))
        for name, info in module_scores.items()
    ]

    return {
        "overall_score": round(overall_score, 1),
        "grade": grade,
        "verdict": verdict_info["label"],
        "action": verdict_info["action"],
        "color": verdict_info["color"],
        "module_scores": module_scores,
        "all_issues": all_issues,
        "issue_summary": issue_summary,
        "summary_lines": summary_lines,
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations builder
# ─────────────────────────────────────────────────────────────────────────────

_RECO_MAP: Dict[str, str] = {
    # Battery
    "BAT-001": "Always fully charge the battery before flight. Partially charged packs reduce endurance and increase cell stress.",
    "BAT-002": "Adjust RTL or battery failsafe voltage threshold to land with higher reserve (min 3.6 V/cell).",
    "BAT-003": "Battery shows elevated internal resistance — check cell balance and consider retirement if trend persists.",
    "BAT-004": "Inspect battery connector and balance lead for loose contacts. Clean with contact cleaner.",
    "BAT-005": "Reduce payload or flight aggressiveness. Consider a higher capacity battery for this mission profile.",
    "BAT-007": "Tighten battery failsafe threshold or reduce mission length to preserve cell life.",
    "BAT-008": "Check battery compartment ventilation. Avoid flying in high ambient temperatures with heavy loads.",
    "BAT-009": "Profile cell_count does not match the measured battery voltage. Update the drone profile yaml with the correct cell_count so voltage thresholds and capacity checks are accurate.",
    # Motors
    "MOT-003": "Inspect and balance propellers on the affected motor. Check motor bearings for play or rough rotation.",
    "MOT-004": "Full mechanical inspection: check all props, motor mounts, and frame arms for damage or looseness.",
    "MOT-005": "Verify actual payload weight against profile setting. If correct, inspect prop pitch/diameter selection.",
    "MOT-006": "Drone is near the edge of its power envelope. Reduce payload, fly in calmer conditions, or upgrade motors.",
    "MOT-007": "Review PID tuning — roll/pitch rate P-gains may be too high. Use AutoTune or reduce gains by 10-15%.",
    "MOT-008": "Engine/motor reached full throttle saturation. For multirotors: do not fly until thrust-to-weight ratio is improved. For fixed-wing: verify this is not caused by headwind or climb — if observed consistently in calm conditions, consider a higher-KV motor or larger propeller.",
    "MOT-009": "CRITICAL: Dead motor detected. Do not fly. Inspect motor windings, ESC, and wiring harness immediately.",
    # Overview
    "OVR-001": "GPS data missing — ensure GPS module is connected and has a clear view of the sky before flight.",
    "OVR-002": "Investigate cause of mode changes. Check RC signal quality, failsafe settings, and EKF health.",
    "OVR-003": "Review endurance deficit root causes in the report.",
    "OVR-004": "Flight endurance exceeded the set expectation — no action required.",
    # Vibration
    "VIB-001": "High mean vibration detected — inspect motor mounts, propeller balance, and frame arm tightness.",
    "VIB-002": "Peak vibration spike critical — immediately inspect props for damage, motor bearings, and landing gear dampers.",
    "VIB-003": "Accelerometer clipping detected — vibration is saturating the IMU. Improve vibration isolation (dampers/foam) before next flight.",
    "VIB-004": "Vibration increasing through flight — check for prop loosening, motor bearing wear, or resonance build-up.",
    "VIB-005": "PID-band oscillation (2-5 Hz) in IMU data — review PID tuning. Reduce roll/pitch P-gains by 10-15% or run AutoTune.",
    "VIB-006": "Structural resonance detected — inspect frame arms, motor mounts, and battery straps for looseness.",
    # Sensors
    "GPS-001": "Poor GPS fix quality — check antenna placement, clear sky view, and GPS cable connections.",
    "GPS-002": "Low satellite count — move to open area, check antenna orientation, and wait for full constellation lock.",
    "GPS-003": "High HDOP — GPS accuracy degraded. Avoid precision missions until HDOP < 1.5.",
    "GPS-004": "GPS position jumps detected — potential multipath interference. Check antenna shielding and avoid flying near reflective surfaces.",
    "EKF-001": "EKF innovation ratio high — sensor data inconsistency. Check compass calibration, GPS interference, and vibration levels.",
    "EKF-002": "EKF filter faults during flight — do not fly until EKF health is confirmed. Review compass, GPS, and baro calibration.",
    "EKF-003": "EKF height innovation anomaly — check barometer for blockage/condensation and GPS altitude consistency.",
    "IMU-001": "IMU temperature critical — risk of sensor drift. Ensure adequate flight controller ventilation.",
    "IMU-002": "Dual IMU readings diverging — one IMU may be faulty. Check vibration isolation and FC mounting.",
    "MAG-001": "Magnetometer field variation high — compass interference detected. Check for nearby power cables, motors, or ESC wiring.",
    "BARO-001": "Barometer altitude instability — check for air pressure leaks around the FC enclosure and ensure baro foam is intact.",
    # Sensor sub-codes (SEN-XXX used by sensors module)
    "SEN-010": "GPS fix quality degraded — check antenna placement, ensure clear sky view, and verify GPS cable connections.",
    "SEN-011": "Low satellite count — move to an open area and wait for full constellation lock before arming.",
    "SEN-012": "GPS satellite count critically low during flight — risk of navigation errors. Check GPS antenna orientation.",
    "SEN-013": "High HDOP — GPS horizontal accuracy degraded. Avoid precision missions until HDOP improves.",
    "SEN-014": "GPS position jumps detected — possible multipath interference. Check antenna shielding.",
    "SEN-020": "EKF sensor disagreement — check compass calibration, GPS signal quality, and vibration levels.",
    "SEN-021": "EKF filter fault — do not fly until EKF health is confirmed. Review compass, GPS, and baro calibration.",
    "SEN-022": "EKF height innovation elevated — check barometer for blockage/condensation and GPS altitude consistency.",
    "SEN-030": "IMU temperature critical — ensure adequate FC ventilation. Avoid heavy loads in high ambient temperatures.",
    "SEN-031": "Large IMU temperature swing — verify IMU heater is functioning for consistent sensor calibration.",
    "SEN-032": "Dual IMU disagreement — check vibration isolation and verify both IMUs are properly mounted and calibrated.",
    "SEN-040": "Mount the compass at least 10 cm away from all high-current cables and ESCs. Route motor wires away from the GPS/compass mast.",
    "SEN-041": "Magnetometer interference spikes — check compass separation from motors and power distribution board.",
    "SEN-050": "Barometer noise elevated — check baro foam seal is intact and baro port is shielded from prop wash.",
    "SEN-051": "Barometer pressure fluctuations — check for airflow interference on the baro port or foam degradation.",
    # Control performance
    "CTL-001": "Roll tracking error elevated — review roll P-gain (ATC_RAT_RLL_P). Run AutoTune or reduce P by 10-15% if oscillating.",
    "CTL-002": "Pitch tracking error elevated — review pitch P-gain (ATC_RAT_PIT_P). Run AutoTune or reduce P by 10-15% if oscillating.",
    "CTL-003": "Yaw tracking error elevated — check yaw P-gain (ATC_RAT_YAW_P), compass calibration, and motor/ESC symmetry.",
    "CTL-004": "Excessive time with large attitude errors — check for wind disturbance, payload imbalance, or insufficient PID gains.",
    "CTL-005": "PID oscillation detected — reduce roll/pitch P-gains by 10-15% or run ArduPilot AutoTune. Check propeller balance and motor mount tightness.",
    # Power efficiency
    "EFF-001": "GPS unavailable — flight distance and Wh/km efficiency cannot be computed. Verify GPS connectivity.",
    "EFF-002": "Flight efficiency below baseline — investigate headwind, payload weight, altitude, or propulsion system wear. Compare against a reference flight.",
    "EFF-003": "Average power near or above rated maximum — verify current sensor calibration. Consider higher capacity battery or reduced payload.",
    # RC Link
    "RCL-001": "Autonomous flight confirmed — no RC receiver in log. Ensure GCS failsafe and battery failsafe are correctly configured for safe RTL/Land.",
    "RCL-002": "RC receiver connected but no stick input detected. Confirm this is expected for this mission type.",
    "RCL-003": "RC signal dropout detected — check transmitter battery, antenna orientation, and distance from receiver. Consider a diversity receiver or antenna tracker.",
    "RCL-005": "RC channel near control limits — check control surface travel limits, stick expo settings, and whether the drone is flying at the edge of its control envelope.",
    # Failsafe events
    "FSF-001": "Battery failsafe triggered in flight — adjust battery failsafe voltage threshold or reduce mission length. Ensure battery is fully charged before each flight.",
    "FSF-002": "RC/Radio failsafe triggered — check RC transmitter range, antenna orientation, and failsafe action (RTL/Land). Test failsafe behavior on the ground before next flight.",
    "FSF-003": "GPS failsafe triggered — check GPS antenna placement, cable connections, and ensure clear sky view. Verify GPS failsafe action is set to Loiter or Land.",
    "FSF-004": "EKF failsafe triggered — review compass calibration, GPS signal quality, and vibration levels. Do not fly until EKF health is confirmed stable.",
    "FSF-005": "Geofence breach detected — review mission boundaries and geofence settings. Ensure geofence polygon matches the planned operational area.",
    "FSF-006": "Thrust loss detected during flight — inspect all motors, ESCs, and wiring immediately. Do not fly until full motor and ESC inspection is complete.",
    "FSF-007": "DRONE GROUNDED: Crash check triggered mid-flight — ArduPilot detected tumbling or uncontrolled flight and disarmed. Perform full structural inspection (frame arms, motor mounts, landing gear), replace any bent or cracked components, re-check motor/ESC health, and verify EKF and vibration levels are within limits before next flight.",
    # Landing Quality
    "LND-001": "Hard landing detected — inspect airframe, landing gear, and motor mounts for structural damage before next flight.",
    "LND-002": "Rough landing — check landing gear and frame for stress. Review descent speed parameter (LAND_SPEED for ArduCopter).",
    "LND-003": "High horizontal speed at touchdown — review LAND_SPEED_HIGH parameter and ensure approach path terminates directly over the landing zone.",
    "LND-004": "Landing bounce — reduce final descent speed (LAND_SPEED) or check for ground effect interference close to the surface.",
    "LND-005": "Unstable approach — review PID tuning for landing phase, reduce approach speed, and check wind conditions at the landing site.",
    # Wind
    "WND-001": "Wind conditions logged — use for mission planning, risk assessment, and flight envelope validation.",
    "WND-002": "Strong wind detected — verify all flights in these conditions are within aircraft wind rating. Check motor and ESC temperatures after landing.",
    "WND-003": "Gusty conditions — consider replanning missions for calmer conditions. Check prop locks, mounting screws, and vibration isolation after flight.",
    # FC Power & Error Health
    "FCH-001": "VCC rail critically low — inspect power module output, wiring, and capacitors. Replace power module if voltage does not stabilise.",
    "FCH-002": "VCC rail sag detected — check power module output voltage and filter capacitor health. Verify no loose connections on the 5V power rail.",
    "FCH-003": "VCC rail noisy — check for ESC switching noise coupling to the FC 5V rail. Add capacitors or use a dedicated low-noise BEC for the FC.",
    "FCH-004": "Servo rail low — check BEC output voltage and servo power connections. Verify no servo is drawing excessive current.",
    "FCH-005": "Servo rail critically low — actuator reliability at risk. Replace BEC or isolate the overloaded servo immediately.",
    "FCH-006": "FC firmware error event — review the specific subsystem calibration and hardware connections listed in the issue.",
    "FCH-007": "FC error cleared mid-flight — paired with a prior error trigger. Investigate root cause even if it self-resolved.",
    "FCH-008": "Multiple FC firmware errors — perform full FC health check: compass calibration, GPS antenna, vibration isolation, and power supply quality.",
    # VTOL Transition
    "VTR-001": "VTOL transition count logged — use for transition quality trending across flights.",
    "VTR-002": "Altitude loss during transition — review ARSPD_FBW_MIN and Q_TRANSITION_MS. Increase transition airspeed threshold.",
    "VTR-003": "Severe altitude loss — ground the aircraft. Review VTOL transition parameters (Q_TRANSITION_MS, ARSPD_FBW_MIN) and verify airspeed sensor calibration.",
    "VTR-004": "Attitude spike during transition — tune VTOL transition gains. Check Q_A_RAT_PIT_P and Q_A_RAT_RLL_P for transition-phase instability.",
    "VTR-005": "Severe attitude spike — ground until reviewed. Check VTOL transition PID gains and verify airspeed sensor is calibrated.",
    "VTR-006": "Slow lift motor ramp — check ESC calibration and motor responsiveness. Verify Q_TRAN_FAIL_MS is set appropriately.",
    "VTR-007": "Very slow lift motor ramp — possible ESC/motor fault. Inspect motor winding resistance, ESC firmware, and signal wiring before next flight.",
    # Mission Execution
    "MSN-001": "Mission partially flown (< 50% of plan) — informational only. Review flight records to confirm whether early return was intentional or caused by a failsafe/emergency.",
    "MSN-002": "Mission partially flown — informational only. Confirm whether early return was planned or caused by an unplanned event.",
    "MSN-003": "Mean cross-track error critically high (> 20 m) — check NAVL1_PERIOD (fixed-wing) or WPNAV_RADIUS (copter) for tighter path tracking. Verify GPS accuracy and compass calibration.",
    "MSN-004": "Elevated mean cross-track error (10-20 m) — review NAVL1_PERIOD or LOITER_RAD for improved accuracy. Check wind compensation parameters (TECS for fixed-wing).",
    "MSN-005": "Peak cross-track deviation > 50 m detected — investigate GPS glitches, wind gusts, or navigation mode transitions at that timestamp. Consider adding waypoint radius margins.",
    # Airspeed Health
    "ASP-001": "Airspeed monitoring active (GPS groundspeed proxy). Install a dedicated airspeed sensor (ARSP) for more accurate stall and overspeed protection.",
    "ASP-002": "Stall risk detected — significant cruise time below minimum airspeed proxy. Check ARSPD_FBW_MIN, TECS_SKIM_SPD, and approach speed settings. Verify airframe stall speed.",
    "ASP-003": "Overspeed event detected — review ARSPD_FBW_MAX and TECS settings. Check for dive scenarios or loss of altitude hold. Verify airframe VNE (never-exceed speed).",
    "ASP-004": "High airspeed variability — review TECS_THR_DAMP and TECS_PTCH2THR for smoother throttle response. Check for wind turbulence in the operational area.",
    # PID Tuning Quality
    "PID-001": "I-term dominance detected — the integral term is compensating for a persistent error the P/D terms cannot correct. Check for: (1) prop wear or imbalance on the affected axis, (2) payload shift between flights, (3) P/D gains set too low. Run ArduPilot AutoTune or increase P-gain by 10-15% on the flagged axis.",
    "PID-002": "D-term noise elevated — the derivative term is amplifying IMU vibration noise. Enable the harmonic notch filter (set INS_HNTCH_ENABLE=1 and configure INS_HNTCH_FREQ to the motor RPM fundamental). Alternatively reduce INS_GYRO_FILTER cutoff or lower D-gain by 15-20%.",
    "PID-003": "Rate tracking error elevated — the rate PID loop is not keeping up with commanded rates. Check motor and ESC health on the affected axis. Verify P-gain is not too low. If mechanical condition is good, increase P-gain by 10% and re-evaluate.",
    "PID-004": "Integrator steady bias detected — the integrator is consistently fighting an offset in one direction. For roll/pitch: check centre-of-gravity position, motor/ESC calibration, and prop pitch consistency. For yaw: check compass calibration and motor/ESC balance. Set appropriate TRIM values if offset is intentional.",
    "PID-005": "Roll/Pitch P-gain asymmetry — roll and pitch axes show significantly different P-term magnitudes. Check payload mounting symmetry, verify propeller pitch is identical on all motors, and inspect motor mounts for stiffness differences between axes.",
    # Power Rail Health
    "PWR-001": "CRITICAL: FC brownout detected — the flight controller's 5V power module could not sustain voltage during flight. Do not fly until resolved. Inspect power module output voltage (should be 5.0–5.3 V), main battery connector torque, and filter capacitor health. Replace power module if output sags below 4.8 V under load.",
    "PWR-002": "Voltage sag under peak current demand - battery or wiring cannot sustain voltage under high load. Perform an internal resistance (IR) test on the battery. If IR is elevated (>25 mohm for 6S), retire the battery. Check main power connector torque and verify wire gauge is adequate for the peak current draw.",
    "PWR-003": "Sustained voltage sag during high-current flight phases — battery is consistently underperforming under operational load. Check battery state of health, verify it is within cycle life, and confirm current draw is within the battery's rated continuous current. Consider a higher-rated battery if this is a normal mission profile.",
    "PWR-004": "Power bus voltage noise elevated — possible cell imbalance, arcing connector, or ESC switching noise. Inspect XT60/XT90 connector pins for oxidation or loose crimp, check battery cell balance via a cell checker, and ensure power distribution board ground connections are solid. Adding a low-ESR capacitor across the main bus may reduce ESC switching noise.",
    # Telemetry Link Quality
    "TLM-001": "Telemetry RSSI low — signal strength at the vehicle radio is marginal. Check antenna orientation (vertical, unobstructed), verify antenna connections are fully seated, and ensure the antenna is mounted away from carbon fibre structure and high-current cables. If range is near the radio's limit, consider a higher-gain antenna or an RFD900 long-range radio upgrade.",
    "TLM-002": "Telemetry SNR low — the noise floor is high relative to signal strength. Relocate the telemetry antenna away from ESCs, motors, and the power distribution board. Use a shielded coaxial extension to move the antenna to the highest point of the airframe. Check for nearby interference sources (WiFi routers, other radios) at the operating site.",
    "TLM-003": "Telemetry TX buffer congested — the radio cannot transmit data as fast as the flight controller is generating it. Reduce MAVLink telemetry stream rates: lower SRx_EXTRA1, SRx_EXTRA2, SRx_POSITION, SRx_EXT_STAT to 2–4 Hz for most ground station needs. Alternatively, upgrade to a higher-bandwidth radio (RFD900x, microhard).",
    "TLM-004": "Elevated telemetry packet error rate — RF link quality is degraded. Check for interference on the operating frequency, verify antenna condition (no broken elements), and confirm the radio's NET ID and encryption key match on both ends. Consider switching to an alternative frequency band if the current one is congested.",
    "TLM-005": "Telemetry RSSI dropout detected — vehicle radio lost signal completely for a sustained period. Review the flight path and identify whether the dropout correlates with a specific heading, altitude, or range. Check for antenna pattern nulls (upgrade from omni to directional if operating at long range), and verify antenna mounting is not blocked by the airframe.",

    # Altitude Control Quality
    "CTN-001": "Altitude tracking error elevated — the flight controller is not holding commanded altitude accurately. For multirotors: review PSC_ACCZ_P/I gains and ensure the barometer is well-shielded from prop wash. For fixed-wing: check TECS_HGT_OMEGA and TECS_HGT_P gains. Verify the baro is not exposed to direct sunlight or pressure differentials from the airframe.",
    "CTN-002": "Climb-rate tracking error elevated — the vertical velocity controller is lagging behind commanded climb/descent rates. For multirotors: adjust PSC_VELZ_P and ensure PILOT_SPEED_UP/DN match the available thrust margin. For fixed-wing: review TECS_VERT_ACC and TECS_TIME_CONST. Check for throttle saturation limiting climb rate response.",
    "CTN-003": "Peak altitude deviation detected — a single large altitude excursion occurred during the flight. Review the timestamp and correlate with mode changes, GPS glitches, or wind gusts. If caused by a commanded step-change, consider adding altitude slew-rate limiting. If unexplained, check barometer and EKF health at that timestamp.",
    "CTN-004": "Altitude hold instability — altitude is oscillating during hover or loiter phases. For multirotors: reduce PSC_ACCZ_P or increase PSC_ACCZ_I to damp oscillation. Check vibration levels (high vibration corrupts baro readings). Ensure the barometer foam seal is intact. For fixed-wing: review TECS gains for altitude hold mode.",

    # ESC Telemetry Health
    "ESC-001": "ESC or motor temperature excessive — one or more ESCs or motor windings reached dangerous temperatures in flight. Inspect propeller size and pitch (over-propped motors run hot), check airflow over ESCs, verify motor KV rating matches battery voltage, and allow adequate cool-down time between flights. Consider upgrading to higher-rated ESCs or motors if this occurs regularly at normal throttle levels.",
    "ESC-002": "Motor current imbalance detected — significant difference in current draw between motors indicates uneven loading. Inspect propellers for damage, warping, or incorrect pitch mix. Check motor bearings for roughness or drag. Verify all motors are the same KV rating and that ESC calibration is uniform. Clean and balance all propellers before next flight.",
    "ESC-003": "Motor RPM imbalance detected — motors are spinning at significantly different speeds under identical commands. This typically indicates a damaged or incorrectly pitched propeller, a partially seized motor bearing, or a failing ESC. Ground the drone and inspect all motors and propellers before the next flight.",
    "ESC-004": "ESC reported internal errors during flight — one or more ESCs logged error counts via BLHeli32/AM32 telemetry. This can indicate desync events (motor stuttering), over-temperature shutdowns, or firmware faults. Review ESC configuration (demag timing, motor timing), check motor/propeller balance, and update ESC firmware. Do not fly until the root cause is identified.",
}


def _build_recommendations(
    all_issues: List[Dict],
    module_scores: Dict[str, Dict],
) -> List[Dict]:
    """Build a prioritised, deduplicated list of recommendations."""
    seen_codes: set = set()
    recommendations: List[Dict] = []

    for issue in all_issues:
        code = issue.get("code", "")
        if code in seen_codes:
            continue
        seen_codes.add(code)

        text = _RECO_MAP.get(code)
        if not text:
            # Generic fallback
            text = issue.get("message", "Investigate and resolve this issue.")

        recommendations.append({
            "priority": "HIGH" if issue.get("severity") == "critical" else
                        "MEDIUM" if issue.get("severity") == "warning" else "LOW",
            "code": code,
            "module": issue.get("module", ""),
            "recommendation": text,
        })

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
