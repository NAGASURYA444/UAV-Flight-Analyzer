"""
ArduPilot DataFlash .bin log parser.

Reads all tracked message types from a binary log and returns a dict of
pandas DataFrames — one DataFrame per message type.  All DataFrames carry
a normalised 'timestamp' column in seconds from log start.

Supported formats
-----------------
* ArduPilot DataFlash binary (.bin) — primary target
* Textual / .log variants are NOT supported here; use the binary format.

Key message types extracted
----------------------------
BAT/BAT2   — battery voltage, current, capacity
GPS/GPS2   — position, fix quality, satellites, HDOP
IMU/IMU2   — accelerometer, gyro, temperature
BARO/BARO2 — altitude, pressure
RCOU       — motor / servo PWM outputs
RCIN       — RC receiver inputs
ATT        — desired vs actual attitude
VIBE       — vibration levels and clipping
MODE       — flight mode changes
ERR        — subsystem errors and failsafes
EV         — discrete events (arm, disarm, …)
MAG/MAG2   — magnetometer
POWR       — board and servo rail voltages
CMD        — mission commands
MSG        — text messages from the flight controller
PARM       — parameter values at time of logging
RADIO      — telemetry radio signal quality (SiK / RFD900 RADIO_STATUS)
ESC        — per-motor ESC telemetry (BLHeli32/AM32: RPM, current, temp, errors)
CTUN       — altitude/climb-rate controller tuning (DAlt, Alt, DCRt, CRt)
NTUN       — ArduPlane navigation tuning (AltErr, XT, AspdE, TAlt)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Message types we want to extract ─────────────────────────────────────────
TRACKED_MESSAGES: frozenset = frozenset({
    # Power
    "BAT", "BAT2",
    # Position & navigation
    "GPS", "GPS2",
    # Inertial
    "IMU", "IMU2", "IMU3",
    # Altitude
    "BARO", "BARO2",
    # Motor outputs
    "RCOU", "RCIN",
    # Attitude
    "ATT",
    # Vibration
    "VIBE",
    # Events & modes
    "MODE", "ERR", "EV",
    # Magnetometer
    "MAG", "MAG2",
    # System
    "POWR",
    # Mission
    "CMD",
    # Info
    "MSG", "PARM",
    # EKF health
    "EKF4", "NKF1", "NKF4", "XKF4",
    # Telemetry radio (SiK / RFD900 / MAVLink RADIO_STATUS forwarded to DataFlash)
    "RADIO",
    # ESC telemetry (BLHeli32 / AM32 — per-motor RPM, current, temp, errors)
    "ESC",
    # Altitude / climb-rate controller tuning (ArduCopter & ArduPlane)
    "CTUN",
    # Navigation tuning — ArduPlane/QuadPlane altitude error, XTE, airspeed error
    "NTUN",
})

# Subsystem IDs used in ERR messages (ArduCopter)
ERR_SUBSYSTEMS: Dict[int, str] = {
    1:  "Main",
    2:  "Radio",
    3:  "Compass",
    4:  "OptFlow",
    5:  "FailsafeRadio",
    6:  "FailsafeBattery",
    7:  "FailsafeGPS",
    8:  "FailsafeFence",
    9:  "FlightMode",
    10: "GPS",
    11: "Crash Check",
    12: "Flip",
    13: "AutoTune",
    14: "Parachute",
    15: "EKF/DCM Check",
    16: "Failsafe EKF/DCM",
    17: "Barometer",
    18: "CPU",
    19: "Logging",
    20: "Thrust Loss Check",
    21: "Sensor Health",
    22: "Terrain",
    23: "Navigation",
    24: "Failsafe ADSB",
    25: "Winch",
}

# EV (event) code names (ArduCopter)
EV_NAMES: Dict[int, str] = {
    10: "Armed",
    11: "Disarmed",
    15: "Auto Armed",
    16: "Takeoff",
    17: "Land Complete Maybe",
    18: "Land Complete",
    19: "Lost GPS",
    21: "Flip Start",
    22: "Flip End",
    25: "Set Home",
    26: "Emergency Landing",
    27: "Land State Maybe",
    28: "Land State Complete",
    38: "EKF Alt Reset",
    39: "Land Abort",
    40: "Scripting Init Success",
}

# ArduPlane / QuadPlane flight mode names
PLANE_MODES: Dict[int, str] = {
    0:  "MANUAL",
    1:  "CIRCLE",
    2:  "STABILIZE",
    3:  "TRAINING",
    4:  "ACRO",
    5:  "FLY_BY_WIRE_A",
    6:  "FLY_BY_WIRE_B",
    7:  "CRUISE",
    8:  "AUTOTUNE",
    10: "AUTO",
    11: "RTL",
    12: "LOITER",
    13: "TAKEOFF",
    14: "AVOID_ADSB",
    15: "GUIDED",
    16: "INITIALISING",
    17: "QSTABILIZE",
    18: "QHOVER",
    19: "QLOITER",
    20: "QLAND",
    21: "QRTL",
    22: "QAUTOTUNE",
    23: "QACRO",
    24: "THERMAL",
    25: "LOITER_ALT_QLAND",
}

# ArduCopter flight mode names
COPTER_MODES: Dict[int, str] = {
    0:  "STABILIZE",
    1:  "ACRO",
    2:  "ALT_HOLD",
    3:  "AUTO",
    4:  "GUIDED",
    5:  "LOITER",
    6:  "RTL",
    7:  "CIRCLE",
    9:  "LAND",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
    25: "SYSTEMID",
    26: "AUTOROTATE",
    27: "AUTO_RTL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class ParseResult:
    """Container returned by parse_bin().  Holds DataFrames and metadata."""

    def __init__(
        self,
        dataframes: Dict[str, pd.DataFrame],
        filepath: str,
        message_count: int,
        available_types: List[str],
    ) -> None:
        self.dataframes = dataframes
        self.filepath = filepath
        self.message_count = message_count
        self.available_types = available_types

    # Convenience accessors ───────────────────────────────────────────────────

    def get(self, msg_type: str) -> Optional[pd.DataFrame]:
        """Return DataFrame for msg_type, or None if not present."""
        return self.dataframes.get(msg_type)

    def has(self, *msg_types: str) -> bool:
        """Return True if ALL requested message types are available."""
        return all(t in self.dataframes for t in msg_types)

    @property
    def time_range(self) -> tuple[float, float]:
        """Overall (start, end) in seconds."""
        t_min, t_max = float("inf"), float("-inf")
        for df in self.dataframes.values():
            if "timestamp" in df.columns and len(df) > 0:
                t_min = min(t_min, df["timestamp"].iloc[0])
                t_max = max(t_max, df["timestamp"].iloc[-1])
        if t_min == float("inf"):
            return 0.0, 0.0
        return t_min, t_max

    @property
    def duration_s(self) -> float:
        s, e = self.time_range
        return max(0.0, e - s)

    def __repr__(self) -> str:
        return (
            f"ParseResult(file={Path(self.filepath).name!r}, "
            f"messages={self.message_count:,}, types={len(self.available_types)})"
        )


def parse_bin(filepath: str) -> ParseResult:
    """
    Parse an ArduPilot DataFlash .bin log file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .bin file.

    Returns
    -------
    ParseResult
        Contains a dict of DataFrames keyed by message type, plus metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not .bin.
    RuntimeError
        If pymavlink cannot open the file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")
    if path.suffix.lower() != ".bin":
        raise ValueError(f"Expected a .bin file, got: {path.suffix!r}")

    logger.info("Opening log: %s", path.name)

    # Use pymavlink DFReader — the correct tool for DataFlash binary logs
    try:
        from pymavlink import DFReader
        log = DFReader.DFReader_binary(str(path), zero_time_base=True)
    except Exception as exc:
        raise RuntimeError(f"pymavlink failed to open '{path.name}': {exc}") from exc

    raw: Dict[str, List[dict]] = {}
    message_count = 0

    while True:
        try:
            msg = log.recv_match()
        except StopIteration:
            break
        except Exception as exc:
            logger.debug("Skipping unreadable message: %s", exc)
            continue

        if msg is None:
            break

        msg_type = msg.get_type()
        if msg_type in ("NONE", "FMT", "FMTU", "MULT", "UNIT", "PARM"):
            # PARM is handled separately — keep parameter snapshot at start
            if msg_type == "PARM":
                try:
                    raw.setdefault("PARM", []).append(msg.to_dict())
                    message_count += 1
                except Exception:
                    pass
            continue

        if msg_type not in TRACKED_MESSAGES:
            continue

        try:
            record = msg.to_dict()
            record.pop("mavpackettype", None)   # internal pymavlink field
        except Exception as exc:
            logger.debug("Could not serialise %s message: %s", msg_type, exc)
            continue

        raw.setdefault(msg_type, []).append(record)
        message_count += 1

    logger.info("Read %d messages across %d message types.", message_count, len(raw))

    dataframes = _build_dataframes(raw)
    return ParseResult(
        dataframes=dataframes,
        filepath=str(path),
        message_count=message_count,
        available_types=sorted(dataframes.keys()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataframes(raw: Dict[str, List[dict]]) -> Dict[str, pd.DataFrame]:
    """Convert raw record lists to sorted DataFrames with normalised timestamps."""
    dataframes: Dict[str, pd.DataFrame] = {}
    t_offset: Optional[float] = None  # subtract log-start so t=0 is first message

    # First pass: find global time offset from the most populous timed message
    for records in raw.values():
        if not records:
            continue
        first = records[0]
        if "TimeUS" in first:
            t0 = first["TimeUS"] / 1_000_000.0
            if t_offset is None or t0 < t_offset:
                t_offset = t0

    if t_offset is None:
        t_offset = 0.0

    for msg_type, records in raw.items():
        if not records:
            continue

        df = pd.DataFrame(records)

        # Normalise timestamp
        if "TimeUS" in df.columns:
            df["timestamp"] = df["TimeUS"] / 1_000_000.0 - t_offset
            df = df.sort_values("timestamp").reset_index(drop=True)
        elif "TimeMS" in df.columns:
            df["timestamp"] = df["TimeMS"] / 1_000.0 - t_offset
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Normalise BAT/BAT2 column names across ArduPilot firmware versions.
        # Pre-3.6 firmware used VoltA/CurrA/ItotA; 3.6+ uses Volt/Curr/CurrTot.
        if msg_type in ("BAT", "BAT2"):
            rename_map = {}
            if "Volt" not in df.columns and "VoltA" in df.columns:
                rename_map["VoltA"] = "Volt"
            if "Curr" not in df.columns and "CurrA" in df.columns:
                rename_map["CurrA"] = "Curr"
            if "CurrTot" not in df.columns and "ItotA" in df.columns:
                rename_map["ItotA"] = "CurrTot"
            if rename_map:
                df = df.rename(columns=rename_map)
                logger.debug("Normalised %s columns: %s", msg_type, rename_map)

        # Enrich MODE messages with human-readable names.
        # Auto-detect firmware: XKF4 = ArduPlane/VTOL, NKF4 = ArduCopter.
        if msg_type == "MODE" and "Mode" in df.columns:
            is_plane = "XKF4" in raw
            mode_map = PLANE_MODES if is_plane else COPTER_MODES
            df["mode_name"] = df["Mode"].map(mode_map).fillna(
                df["Mode"].astype(str)
            )
            df["firmware"] = "ArduPlane" if is_plane else "ArduCopter"

        # Enrich ERR messages
        if msg_type == "ERR":
            if "Subsys" in df.columns:
                df["subsys_name"] = df["Subsys"].map(ERR_SUBSYSTEMS).fillna(
                    df["Subsys"].astype(str)
                )

        # Enrich EV messages
        if msg_type == "EV" and "Id" in df.columns:
            df["event_name"] = df["Id"].map(EV_NAMES).fillna(
                df["Id"].astype(str)
            )

        dataframes[msg_type] = df
        logger.debug("  %-8s  %d records", msg_type, len(df))

    return dataframes
