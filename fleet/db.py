"""
Fleet Database  (Phase 6)
=========================
SQLite-backed storage for all analyzed flight records.

Auto-created on first use at  <project_root>/fleet.db
Never requires manual setup — completely transparent to the user.

Key operations
--------------
save_flight(report)          Store one flight record (upsert by drone+log_file)
get_fleet_summary()          Latest flight per drone + trend flag
get_drone_history(name)      All flights for one drone, newest first
get_all_flights()            Every record in the DB
delete_flight(flight_id)     Remove a specific record
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default DB location: project root / fleet.db
_DEFAULT_DB = Path(__file__).parent.parent / "fleet.db"


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_FLIGHTS = """
CREATE TABLE IF NOT EXISTS flights (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    drone_name                   TEXT    NOT NULL,
    drone_type                   TEXT,
    log_file                     TEXT,
    flight_date                  TEXT,
    analyzed_at                  TEXT,

    -- Flight summary
    duration_s                   REAL,
    overall_score                REAL,
    grade                        TEXT,
    verdict                      TEXT,
    crit_count                   INTEGER DEFAULT 0,
    warn_count                   INTEGER DEFAULT 0,
    info_count                   INTEGER DEFAULT 0,

    -- Module scores (NULL = module unavailable in that log)
    score_flight_overview        REAL,
    score_battery                REAL,
    score_motors                 REAL,
    score_vibration              REAL,
    score_sensors                REAL,
    score_control                REAL,
    score_efficiency             REAL,
    score_rc_link                REAL,
    score_landing_wind           REAL,
    score_fc_health              REAL,
    score_vtol_transition        REAL,
    score_mission                REAL,
    score_airspeed               REAL,
    score_pid_tuning             REAL,
    score_power_rail             REAL,
    score_telemetry              REAL,
    score_esc_telemetry          REAL,
    score_altitude_control       REAL,

    -- Key metrics (for trending)
    bat_start_cell_v             REAL,
    bat_end_cell_v               REAL,
    bat_internal_resistance_mohm REAL,
    bat_capacity_consumed_mah    REAL,
    bat_capacity_remaining_pct   REAL,
    bat_avg_current_a            REAL,
    vibe_x                       REAL,
    vibe_y                       REAL,
    vibe_z                       REAL,
    vibe_clips                   INTEGER,
    motor_imbalance_pct          REAL,
    motor_avg_throttle_pct       REAL,
    airborne_s                   REAL,
    distance_km                  REAL,
    max_altitude_agl_m           REAL,
    efficiency_wh_per_km         REAL,
    energy_wh                    REAL,
    roll_rms_deg                 REAL,
    pitch_rms_deg                REAL,
    descent_rate_ms              REAL,
    wind_speed_ms                REAL,

    -- Unique constraint: same drone + same log file = update, not duplicate
    UNIQUE(drone_name, log_file)
)
"""

_CREATE_ISSUES = """
CREATE TABLE IF NOT EXISTS flight_issues (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_id  INTEGER NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
    severity   TEXT,
    code       TEXT,
    module     TEXT,
    message    TEXT
)
"""

_CREATE_IDX_DRONE = "CREATE INDEX IF NOT EXISTS idx_flights_drone ON flights(drone_name)"
_CREATE_IDX_DATE  = "CREATE INDEX IF NOT EXISTS idx_flights_date  ON flights(flight_date)"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _init(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(_CREATE_FLIGHTS)
        conn.execute(_CREATE_ISSUES)
        conn.execute(_CREATE_IDX_DRONE)
        conn.execute(_CREATE_IDX_DATE)


def _m(modules: Dict, mod: str, key: str) -> Optional[float]:
    """Safely extract a metric value from module_results."""
    return (modules.get(mod) or {}).get("metrics", {}).get(key)


def _ms(module_scores: Dict, mod: str) -> Optional[float]:
    """Extract module score, return None if unavailable."""
    info = module_scores.get(mod, {})
    if not info.get("available", True):
        return None
    return info.get("score")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_flight(
    report: Dict[str, Any],
    db_path: Path = _DEFAULT_DB,
) -> int:
    """
    Store one flight report in the database.
    If the same (drone_name, log_file) already exists, the record is updated.

    Returns the flight id.
    """
    _init(db_path)

    meta    = report.get("meta", {})
    summary = report.get("flight_summary", {})
    modules = report.get("module_results", {})
    issues  = report.get("all_issues", [])
    mscores = summary.get("module_scores", {})
    isums   = summary.get("issue_summary", {})

    drone_name = meta.get("drone_name", "Unknown")
    log_file   = meta.get("log_file", "")

    # Prefer log start time; fall back to analysis timestamp
    flight_date = meta.get("log_start_time") or meta.get("generated_at", "")
    analyzed_at = meta.get("generated_at", datetime.utcnow().isoformat())

    row = {
        "drone_name":   drone_name,
        "drone_type":   meta.get("drone_type"),
        "log_file":     log_file,
        "flight_date":  flight_date,
        "analyzed_at":  analyzed_at,
        "duration_s":   meta.get("log_duration_s"),
        "overall_score": summary.get("overall_score"),
        "grade":        summary.get("grade"),
        "verdict":      summary.get("verdict"),
        "crit_count":   isums.get("critical", 0),
        "warn_count":   isums.get("warning",  0),
        "info_count":   isums.get("info",     0),

        # Module scores
        "score_flight_overview":  _ms(mscores, "flight_overview"),
        "score_battery":          _ms(mscores, "battery"),
        "score_motors":           _ms(mscores, "motors"),
        "score_vibration":        _ms(mscores, "vibration"),
        "score_sensors":          _ms(mscores, "sensors"),
        "score_control":          _ms(mscores, "control"),
        "score_efficiency":       _ms(mscores, "efficiency"),
        "score_rc_link":          _ms(mscores, "rc_link"),
        "score_landing_wind":     _ms(mscores, "landing_wind"),
        "score_fc_health":        _ms(mscores, "fc_health"),
        "score_vtol_transition":  _ms(mscores, "vtol_transition"),
        "score_mission":          _ms(mscores, "mission"),
        "score_airspeed":         _ms(mscores, "airspeed"),
        "score_pid_tuning":       _ms(mscores, "pid_tuning"),
        "score_power_rail":       _ms(mscores, "power_rail"),
        "score_telemetry":        _ms(mscores, "telemetry"),
        "score_esc_telemetry":    _ms(mscores, "esc_telemetry"),
        "score_altitude_control": _ms(mscores, "altitude_control"),

        # Battery metrics
        "bat_start_cell_v":              _m(modules, "battery", "start_cell_voltage_v"),
        "bat_end_cell_v":                _m(modules, "battery", "end_cell_voltage_v"),
        "bat_internal_resistance_mohm":  _m(modules, "battery", "internal_resistance_mohm"),
        "bat_capacity_consumed_mah":     _m(modules, "battery", "capacity_consumed_mah"),
        "bat_capacity_remaining_pct":    _m(modules, "battery", "capacity_remaining_pct"),
        "bat_avg_current_a":             _m(modules, "battery", "avg_current_a"),

        # Vibration metrics
        "vibe_x":     _m(modules, "vibration", "mean_vibe_x"),
        "vibe_y":     _m(modules, "vibration", "mean_vibe_y"),
        "vibe_z":     _m(modules, "vibration", "mean_vibe_z"),
        "vibe_clips": _m(modules, "vibration", "total_clipping_events"),

        # Motor metrics
        "motor_imbalance_pct":    _m(modules, "motors", "avg_motor_imbalance_pct"),
        "motor_avg_throttle_pct": _m(modules, "motors", "avg_throttle_pct"),

        # Flight overview metrics
        "airborne_s":        _m(modules, "flight_overview", "airborne_s"),
        "distance_km":       _m(modules, "flight_overview", "distance_km"),
        "max_altitude_agl_m":_m(modules, "flight_overview", "max_altitude_agl_m"),

        # Efficiency metrics
        "efficiency_wh_per_km": _m(modules, "efficiency", "wh_per_km"),
        "energy_wh":            _m(modules, "efficiency", "energy_wh"),

        # Control metrics
        "roll_rms_deg":   _m(modules, "control", "roll_rms_deg"),
        "pitch_rms_deg":  _m(modules, "control", "pitch_rms_deg"),

        # Landing metrics
        "descent_rate_ms": _m(modules, "landing_wind", "descent_rate_ms"),
        "wind_speed_ms":   _m(modules, "landing_wind", "wind_speed_ms"),
    }

    cols   = ", ".join(row.keys())
    params = ", ".join(f":{k}" for k in row.keys())
    update = ", ".join(
        f"{k}=excluded.{k}" for k in row.keys()
        if k not in ("drone_name", "log_file")
    )

    sql = (
        f"INSERT INTO flights ({cols}) VALUES ({params}) "
        f"ON CONFLICT(drone_name, log_file) DO UPDATE SET {update}"
    )

    with _connect(db_path) as conn:
        cur = conn.execute(sql, row)
        flight_id = cur.lastrowid

        # For upserts (ON CONFLICT DO UPDATE), lastrowid may be 0 — look it up
        if not flight_id:
            r = conn.execute(
                "SELECT id FROM flights WHERE drone_name=? AND log_file=?",
                (drone_name, log_file),
            ).fetchone()
            if r:
                flight_id = r["id"]

        # Re-save issues (delete old ones first for upsert correctness)
        conn.execute("DELETE FROM flight_issues WHERE flight_id=?", (flight_id,))
        if issues:
            conn.executemany(
                "INSERT INTO flight_issues (flight_id,severity,code,module,message) "
                "VALUES (?,?,?,?,?)",
                [
                    (flight_id,
                     i.get("severity"), i.get("code"),
                     i.get("module"),   i.get("message"))
                    for i in issues
                ],
            )

    logger.info("Fleet DB: saved flight id=%d  drone=%s  log=%s", flight_id, drone_name, log_file)
    return flight_id


def get_drone_history(
    drone_name: str,
    limit: int = 20,
    db_path: Path = _DEFAULT_DB,
) -> List[Dict]:
    """Return up to `limit` flights for one drone, newest first."""
    _init(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM flights WHERE drone_name=? ORDER BY flight_date DESC, id DESC LIMIT ?",
            (drone_name, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_fleet_summary(db_path: Path = _DEFAULT_DB) -> List[Dict]:
    """
    Return one row per drone (most recent flight) plus flight count.
    Ordered by overall_score ascending (worst first — needs attention first).
    """
    _init(db_path)
    sql = """
        SELECT f.*,
               cnt.total_flights
        FROM flights f
        INNER JOIN (
            SELECT drone_name, MAX(id) AS max_id, COUNT(*) AS total_flights
            FROM flights
            GROUP BY drone_name
        ) cnt ON f.drone_name = cnt.drone_name AND f.id = cnt.max_id
        ORDER BY COALESCE(f.overall_score, 999) ASC
    """
    with _connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_all_flights(db_path: Path = _DEFAULT_DB) -> List[Dict]:
    """Return every flight record, newest first."""
    _init(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM flights ORDER BY flight_date DESC, id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_flight(flight_id: int, db_path: Path = _DEFAULT_DB) -> None:
    """Remove a flight record and its associated issues."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM flights WHERE id=?", (flight_id,))
    logger.info("Fleet DB: deleted flight id=%d", flight_id)


def get_db_path() -> Path:
    return _DEFAULT_DB
