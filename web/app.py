"""
UAV Flight Analyzer — Web Frontend (Phase 7)
=============================================
FastAPI backend serving the two-tab web dashboard.

Run with:
    python run_web.py
or:
    uvicorn web.app:app --host 0.0.0.0 --port 5000 --reload
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="UAV Flight Analyzer", version="7.0")

STATIC_DIR = Path(__file__).parent / "static"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────────────────────────────────────
# Profile API
# ─────────────────────────────────────────────────────────────────────────────

PROFILE_MAP = {
    "quadcopter": ROOT / "profiles" / "quadcopter.yaml",
    "quadplane":  ROOT / "profiles" / "quadplane.yaml",
    "fixed_wing": ROOT / "profiles" / "fixed_wing.yaml",
}


@app.get("/api/profiles/{vehicle_type}")
async def get_profile(vehicle_type: str):
    """Return editable profile fields for a vehicle type."""
    from analyzer.config.drone_profile import DroneProfile

    path = PROFILE_MAP.get(vehicle_type)
    if not path or not path.exists():
        raise HTTPException(status_code=400, detail=f"Unknown vehicle type: {vehicle_type}")

    p = DroneProfile.load(str(path))
    return {
        "drone_name":             p.name,
        "type":                   p.type,
        # Operational
        "expected_endurance_min": p.expected_endurance_min,
        "expected_range_km":      p.expected_range_km,
        "max_payload_kg":         p.max_payload_kg,
        "max_speed_ms":           p.max_speed_ms,
        "max_altitude_m":         p.max_altitude_m,
        # Battery
        "battery": {
            "cell_count":               p.battery.cell_count,
            "capacity_mah":             p.battery.capacity_mah,
            "nominal_voltage":          p.battery.nominal_voltage,
            "full_voltage":             p.battery.full_voltage,
            "min_cell_voltage":         p.battery.min_cell_voltage,
            "critical_cell_voltage":    p.battery.critical_cell_voltage,
            "max_continuous_current_a": p.battery.max_continuous_current_a,
            "cycles_lifespan":          p.battery.cycles_lifespan,
        },
        # Motors
        "motors": {
            "count":                      p.motors.count,
            "channels":                   p.motors.channels,
            "hover_throttle_pct":         p.motors.hover_throttle_pct,
            "high_throttle_warn_pct":     p.motors.high_throttle_warn_pct,
            "high_throttle_critical_pct": p.motors.high_throttle_critical_pct,
            "max_pwm":                    p.motors.max_pwm,
            "min_pwm":                    p.motors.min_pwm,
        },
        # Vibration
        "vibration": {
            "warn_threshold":     p.vibration.warn_threshold,
            "critical_threshold": p.vibration.critical_threshold,
            "clip_warn_per_min":  p.vibration.clip_warn_per_min,
        },
        # GPS
        "gps": {
            "min_satellites":      p.gps.min_satellites,
            "max_hdop":            p.gps.max_hdop,
            "min_fix_type":        p.gps.min_fix_type,
            "max_position_jump_m": p.gps.max_position_jump_m,
        },
        # Maintenance
        "maintenance": {
            "motor_hours":              p.maintenance.motor_hours,
            "prop_cycles":              p.maintenance.prop_cycles,
            "battery_cycles":           p.maintenance.battery_cycles,
            "esc_hours":                p.maintenance.esc_hours,
            "frame_inspection_hours":   p.maintenance.frame_inspection_hours,
        },
    }


@app.get("/api/profiles")
async def list_profiles():
    """Return all available vehicle types."""
    return list(PROFILE_MAP.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Analyze API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    log_file: UploadFile = File(...),
    drone_id: str = Form(...),
    vehicle_type: str = Form(...),
    profile_overrides: str = Form(default="{}"),
):
    """Upload a .BIN file and run full analysis. Returns report JSON."""
    if not log_file.filename.lower().endswith(".bin"):
        raise HTTPException(status_code=400, detail="Only .BIN files are supported.")

    drone_id = drone_id.strip()
    if not drone_id:
        raise HTTPException(status_code=400, detail="Drone ID is required.")

    if vehicle_type not in PROFILE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown vehicle type: {vehicle_type}")

    try:
        overrides = json.loads(profile_overrides)
    except json.JSONDecodeError:
        overrides = {}

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        shutil.copyfileobj(log_file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        report = _run_analysis(
            log_path=tmp_path,
            vehicle_type=vehicle_type,
            drone_id=drone_id,
            overrides=overrides,
            original_filename=log_file.filename,
        )
        return JSONResponse(content=report)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass  # Windows: pymavlink may still hold the file handle; OS cleans it up


def _apply_overrides(profile, overrides: Dict) -> None:
    """Patch a DroneProfile in-place with user-supplied field overrides."""
    if not overrides:
        return

    def _f(v):
        """Safe float conversion — returns None if v is None or NaN."""
        if v is None:
            return None
        try:
            r = float(v)
            import math
            return None if math.isnan(r) else r
        except (TypeError, ValueError):
            return None

    def _i(v):
        """Safe int conversion."""
        r = _f(v)
        return None if r is None else int(r)

    # Operational
    for key in ("expected_endurance_min", "expected_range_km", "max_payload_kg",
                "max_speed_ms", "max_altitude_m"):
        v = _f(overrides.get(key))
        if v is not None:
            setattr(profile, key, v)

    # Battery
    bat = overrides.get("battery") or {}
    for key in ("capacity_mah", "nominal_voltage", "full_voltage", "min_cell_voltage",
                "critical_cell_voltage", "max_continuous_current_a"):
        v = _f(bat.get(key))
        if v is not None:
            setattr(profile.battery, key, v)
    v = _i(bat.get("cell_count"))
    if v is not None:
        profile.battery.cell_count = v
    v = _i(bat.get("cycles_lifespan"))
    if v is not None:
        profile.battery.cycles_lifespan = v

    # Motors
    mot = overrides.get("motors") or {}
    v = _i(mot.get("count"))
    if v is not None:
        profile.motors.count = v
    if mot.get("channels") is not None:
        profile.motors.channels = [int(c) for c in mot["channels"]]
    for key in ("hover_throttle_pct", "high_throttle_warn_pct", "high_throttle_critical_pct"):
        v = _f(mot.get(key))
        if v is not None:
            setattr(profile.motors, key, v)
    for key in ("max_pwm", "min_pwm"):
        v = _i(mot.get(key))
        if v is not None:
            setattr(profile.motors, key, v)

    # Vibration
    vib = overrides.get("vibration") or {}
    for key in ("warn_threshold", "critical_threshold"):
        v = _f(vib.get(key))
        if v is not None:
            setattr(profile.vibration, key, v)
    v = _i(vib.get("clip_warn_per_min"))
    if v is not None:
        profile.vibration.clip_warn_per_min = v

    # GPS
    gps = overrides.get("gps") or {}
    for key in ("max_hdop", "max_position_jump_m"):
        v = _f(gps.get(key))
        if v is not None:
            setattr(profile.gps, key, v)
    for key in ("min_satellites", "min_fix_type"):
        v = _i(gps.get(key))
        if v is not None:
            setattr(profile.gps, key, v)

    # Maintenance
    maint = overrides.get("maintenance") or {}
    for key in ("motor_hours", "esc_hours", "frame_inspection_hours"):
        v = _f(maint.get(key))
        if v is not None:
            setattr(profile.maintenance, key, v)
    for key in ("prop_cycles", "battery_cycles"):
        v = _i(maint.get(key))
        if v is not None:
            setattr(profile.maintenance, key, v)


def _run_analysis(
    log_path: Path,
    vehicle_type: str,
    drone_id: str,
    overrides: Dict,
    original_filename: str,
) -> Dict:
    from analyzer.parser.bin_parser import parse_bin
    from analyzer.config.drone_profile import DroneProfile
    from analyzer.modules import (
        flight_overview, battery, motors, vibration, sensors, control,
        efficiency, rc_link, landing_wind, fc_health, vtol_transition,
        mission, airspeed, pid_tuning, power_rail, telemetry,
        esc_telemetry, altitude_control,
    )
    from analyzer.scoring import engine as scoring_engine
    from analyzer.report import json_report, html_report as html_report_mod
    from fleet import db as fleet_db, trend as fleet_trend

    drone_profile = DroneProfile.load(str(PROFILE_MAP[vehicle_type]))
    _apply_overrides(drone_profile, overrides)

    parse_result = parse_bin(str(log_path))

    overview_result = flight_overview.analyse(parse_result, drone_profile)
    arm_time    = overview_result["metrics"].get("arm_time_s", 0.0)
    disarm_time = overview_result["metrics"].get("disarm_time_s")

    module_results = {
        "flight_overview":  overview_result,
        "battery":          battery.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "motors":           motors.analyse(parse_result, drone_profile, arm_time=arm_time),
        "vibration":        vibration.analyse(parse_result, drone_profile, arm_time=arm_time),
        "sensors":          sensors.analyse(parse_result, drone_profile, arm_time=arm_time),
        "control":          control.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "efficiency":       efficiency.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "rc_link":          rc_link.analyse(parse_result, drone_profile, arm_time=arm_time),
        "landing_wind":     landing_wind.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "fc_health":        fc_health.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "vtol_transition":  vtol_transition.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "mission":          mission.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "airspeed":         airspeed.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "pid_tuning":       pid_tuning.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "power_rail":       power_rail.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "telemetry":        telemetry.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "esc_telemetry":    esc_telemetry.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
        "altitude_control": altitude_control.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time),
    }
    module_results["flight_overview"] = flight_overview.post_analyse(
        module_results["flight_overview"], module_results, drone_profile
    )

    scoring_result = scoring_engine.compute(module_results, drone_profile)

    # Use the uploaded file's mtime as the flight date (same logic as CLI path).
    # log_path is the temp file on disk — its mtime is set when the upload was saved,
    # which is close to "now", but it's the best we can do without parsing log internals.
    # More importantly: this is stable per upload, so re-analysis of the same file
    # produces the same flight_date rather than the analysis timestamp.
    _log_mtime = os.path.getmtime(log_path)
    _log_start_time = datetime.fromtimestamp(_log_mtime, tz=timezone.utc).isoformat()

    parse_meta = {
        "filepath":        original_filename,
        "message_count":   parse_result.message_count,
        "available_types": parse_result.available_types,
        "duration_s":      round(parse_result.duration_s, 2),
        "log_start_time":  _log_start_time,
    }
    profile_info = {
        "id":   drone_profile.id,
        "name": drone_id,
        "type": drone_profile.type,
    }

    report_dict = json_report.generate(
        parse_result_meta=parse_meta,
        module_results=module_results,
        scoring_result=scoring_result,
        profile_info=profile_info,
        output_path=None,
    )
    report_dict["meta"]["drone_name"] = drone_id
    report_dict["meta"]["log_file"]   = original_filename

    # Save to fleet DB
    flight_id = fleet_db.save_flight(report_dict)

    # Attach fleet history + trends
    history = fleet_db.get_drone_history(drone_id, limit=20)
    trends  = fleet_trend.compute_trends(history)
    report_dict["fleet_history"] = history
    report_dict["fleet_trends"]  = {
        k: {
            "label":     t.label,
            "arrow":     t.arrow,
            "slope":     round(t.slope, 4),
            "direction": t.direction,
            "alert":     t.alert,
            "alert_msg": t.alert_msg,
            "unit":      t.unit,
            "latest":    t.latest,
            "values":    [v for v in t.values if v is not None][-10:],
        }
        for k, t in trends.items()
    }
    report_dict["flight_id"] = flight_id

    # Generate and save HTML report
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in drone_id)
    html_filename = f"{safe_name}_{flight_id}_report.html"
    html_path = REPORTS_DIR / html_filename
    html_report_mod.generate(report_dict, output_path=str(html_path))
    report_dict["html_report_url"] = f"/reports/{html_filename}"

    return report_dict


# ─────────────────────────────────────────────────────────────────────────────
# Fleet API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/fleet")
async def get_fleet():
    """Fleet overview — one row per drone, worst score first."""
    from fleet import db as fleet_db, trend as fleet_trend
    rows = fleet_db.get_fleet_summary()
    result = []
    for r in rows:
        name = r["drone_name"]
        history = fleet_db.get_drone_history(name, limit=10)
        arrow, alerts = fleet_trend.summary_for_cli(history)
        result.append({**dict(r), "trend_arrow": arrow, "alerts": alerts})
    return result


@app.get("/api/fleet/drone-ids")
async def get_drone_ids():
    """Return all distinct drone IDs for the Drone ID autocomplete."""
    from fleet import db as fleet_db
    rows = fleet_db.get_fleet_summary()
    return [r["drone_name"] for r in rows]


@app.get("/api/fleet/{drone_id}/history")
async def get_drone_history(drone_id: str):
    """History + trends + alerts for one drone."""
    from fleet import db as fleet_db, trend as fleet_trend
    history = fleet_db.get_drone_history(drone_id, limit=50)
    if not history:
        raise HTTPException(status_code=404, detail=f"No flights found for: {drone_id}")
    trends = fleet_trend.compute_trends(history)
    alerts = fleet_trend.fleet_alerts(history)
    return {
        "drone_id":      drone_id,
        "drone_type":    history[0].get("drone_type", ""),
        "total_flights": len(history),
        "history":       history,
        "trends": {
            k: {
                "label":     t.label,
                "arrow":     t.arrow,
                "slope":     round(t.slope, 4),
                "direction": t.direction,
                "alert":     t.alert,
                "alert_msg": t.alert_msg,
                "unit":      t.unit,
                "latest":    t.latest,
                "values":    [v for v in t.values if v is not None][-10:],
            }
            for k, t in trends.items()
        },
        "alerts": alerts,
    }


@app.get("/api/fleet/{drone_id}/export")
async def export_csv(drone_id: str):
    """Download drone flight history as CSV (opens in Excel)."""
    from fleet import db as fleet_db
    history = fleet_db.get_drone_history(drone_id, limit=500)
    if not history:
        raise HTTPException(status_code=404, detail=f"No flights for: {drone_id}")

    cols = [
        "flight_date", "log_file", "overall_score", "grade", "verdict",
        "duration_s", "crit_count", "warn_count", "info_count",
        "bat_internal_resistance_mohm", "bat_end_cell_v", "bat_start_cell_v",
        "bat_capacity_remaining_pct", "bat_capacity_consumed_mah",
        "bat_avg_current_a", "vibe_x", "vibe_y", "vibe_z",
        "motor_imbalance_pct", "motor_avg_throttle_pct",
        "efficiency_wh_per_km", "energy_wh",
        "distance_km", "max_altitude_agl_m", "airborne_s",
        "score_battery", "score_motors", "score_vibration",
        "score_sensors", "score_control", "score_efficiency",
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["drone_id"] + cols)
    for f in history:
        writer.writerow([drone_id] + [f.get(c, "") for c in cols])

    buf.seek(0)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in drone_id)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_history.csv"'},
    )
