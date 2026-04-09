"""
Drone profile loader and validator.
Profiles are YAML files in the profiles/ directory.
Each drone in a fleet gets its own profile tuned from testing data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

def _detect_cells_from_voltage(voltage: float) -> Optional[int]:
    """
    Infer LiPo/LiHV cell count from a pack voltage reading.
    Valid per-cell range: 3.20 V (depleted) to 4.35 V (LiHV full charge).
    Returns None if no standard cell count gives a plausible result.
    """
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]:
        v_cell = voltage / n
        if 3.20 <= v_cell <= 4.35:
            return n
    return None

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    capacity_mah: float = 5000.0
    cell_count: int = 4
    nominal_voltage: float = 14.8
    full_voltage: float = 16.8
    min_cell_voltage: float = 3.5
    critical_cell_voltage: float = 3.3
    max_continuous_current_a: float = 50.0
    cycles_lifespan: int = 300


@dataclass
class MotorConfig:
    count: int = 4
    channels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    max_pwm: int = 2000
    min_pwm: int = 1000
    hover_throttle_pct: float = 50.0
    high_throttle_warn_pct: float = 80.0
    high_throttle_critical_pct: float = 92.0


@dataclass
class VibrationConfig:
    warn_threshold: float = 15.0
    critical_threshold: float = 30.0
    clip_warn_per_min: int = 5


@dataclass
class GPSConfig:
    min_satellites: int = 8
    max_hdop: float = 1.5
    min_fix_type: int = 3
    max_position_jump_m: float = 10.0


@dataclass
class MaintenanceConfig:
    motor_hours: float = 200.0
    prop_cycles: int = 100
    battery_cycles: int = 300
    esc_hours: float = 200.0
    frame_inspection_hours: float = 50.0


@dataclass
class PusherMotorConfig:
    """Configuration for the pusher/cruise motor on VTOL QuadPlane airframes."""
    channel: int = 3                    # RCOU channel (typically C3 on QuadPlane)
    max_pwm: int = 2000
    min_pwm: int = 1000
    high_throttle_warn_pct: float = 80.0
    high_throttle_critical_pct: float = 92.0


@dataclass
class DroneProfile:
    id: str = "default"
    name: str = "Default Drone"
    type: str = "multirotor"          # multirotor | fixed_wing | vtol
    expected_endurance_min: float = 20.0
    expected_range_km: float = 5.0
    max_payload_kg: float = 1.0
    max_speed_ms: float = 15.0
    max_altitude_m: float = 120.0

    battery: BatteryConfig = field(default_factory=BatteryConfig)
    motors: MotorConfig = field(default_factory=MotorConfig)
    vibration: VibrationConfig = field(default_factory=VibrationConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    pusher_motor: Optional[PusherMotorConfig] = None   # VTOL QuadPlane only

    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "battery": 0.30,
        "motors": 0.25,
        "flight_overview": 0.25,
        "sensors": 0.20,
    })

    thresholds: Dict[str, float] = field(default_factory=dict)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def motor_category(self) -> str:
        """Auto-generate drone category from motor count and type."""
        if self.type == "fixed_wing":
            return "Fixed-Wing"
        if self.type == "vtol":
            return "VTOL / QuadPlane"
        count = self.motors.count
        return {
            1:  "Single Motor",
            3:  "Tricopter",
            4:  "Quadcopter",
            6:  "Hexacopter",
            8:  "Octocopter",
            12: "Dodecacopter",
        }.get(count, f"{count}-Motor Multirotor")

    @property
    def min_pack_voltage(self) -> float:
        return self.battery.min_cell_voltage * self.battery.cell_count

    @property
    def critical_pack_voltage(self) -> float:
        return self.battery.critical_cell_voltage * self.battery.cell_count

    @property
    def pwm_range(self) -> int:
        return self.motors.max_pwm - self.motors.min_pwm

    def pwm_to_throttle_pct(self, pwm: float) -> float:
        """Convert raw PWM value to 0-100 throttle percentage."""
        return max(0.0, min(100.0, (pwm - self.motors.min_pwm) / self.pwm_range * 100.0))

    def get_threshold(self, key: str, default: float) -> float:
        """Return a profile-overridden threshold, or the module default if not set."""
        val = self.thresholds.get(key)
        if val is not None:
            return float(val)
        return default

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, yaml_path: str) -> "DroneProfile":
        """Load a drone profile from a YAML file. Falls back to defaults on missing file."""
        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Profile not found at '%s' — using built-in defaults.", yaml_path)
            return cls()
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls._from_dict(data)

    @classmethod
    def from_log(cls, result: Any) -> "DroneProfile":
        """
        Build a DroneProfile for an unknown log by:
          1. Detecting the drone type from the log data.
          2. Loading the matching default YAML (profiles/defaults/<type>_default.yaml).
             This gives ALL fields (operational expectations, vibration thresholds,
             GPS, maintenance, scoring weights) with sensible type-appropriate values.
          3. Auto-patching only the values the log can tell us:
             - motor channels and count  (from RCOU)
             - battery cell count        (from BAT voltage)

        The result is a fully-populated profile — identical in structure to a
        fleet-specific YAML — ready for the frontend to display and let the user
        customise into their own fleet profile.

        Detection logic
        ---------------
        Drone type  : XKF4 / TECS present → ArduPlane firmware.
                      Q-prefixed modes present → VTOL; else fixed_wing.
                      Otherwise → multirotor.
        Motor count : RCOU channels whose minimum PWM is near idle (≤ min_pwm+80).
        Cell count  : First BAT voltage mapped to nearest plausible cell count
                      (3.20–4.35 V/cell range covers LiPo and LiHV).
        """
        dataframes = getattr(result, "dataframes", {})

        # ── Step 1: detect drone type ─────────────────────────────────────────
        has_plane_fw = "XKF4" in dataframes
        if has_plane_fw:
            mode_df = dataframes.get("MODE")
            is_vtol = False
            if mode_df is not None and "mode_name" in mode_df.columns:
                mode_names = set(mode_df["mode_name"].dropna().tolist())
                is_vtol = any(isinstance(m, str) and m.startswith("Q")
                              for m in mode_names)
            drone_type = "vtol" if is_vtol else "fixed_wing"
        else:
            drone_type = "multirotor"

        # ── Step 2: load the matching default YAML ────────────────────────────
        # The default YAML lives next to the fleet profiles in profiles/defaults/.
        # Resolve the path relative to THIS file so it works regardless of the
        # working directory the CLI is launched from.
        defaults_dir = Path(__file__).parent.parent.parent / "profiles" / "defaults"
        default_yaml = defaults_dir / f"{drone_type}_default.yaml"

        if default_yaml.exists():
            with open(default_yaml, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            profile = cls._from_dict(data)
            logger.info("Loaded default profile: %s", default_yaml.name)
        else:
            # Fallback: bare dataclass defaults if default YAML somehow missing
            logger.warning("Default profile not found at '%s' — using built-in defaults.",
                           default_yaml)
            profile = cls()
            profile.type = drone_type

        profile.id   = "auto-detected"
        profile.name = f"{profile.motor_category}"

        # ── Step 3: auto-patch motor channels from RCOU ───────────────────────
        rcou_df = dataframes.get("RCOU")
        if rcou_df is not None:
            idle_thresh = profile.motors.min_pwm + 80
            detected_channels: List[int] = []
            for i in range(1, 17):
                col = f"C{i}"
                if col not in rcou_df.columns:
                    continue
                col_min = float(rcou_df[col].min())
                col_max = float(rcou_df[col].max())
                if (col_max - col_min > 100
                        and 800 <= col_min
                        and col_max <= 2200
                        and col_min <= idle_thresh):
                    detected_channels.append(i)
            if detected_channels:
                profile.motors.channels = detected_channels
                profile.motors.count    = len(detected_channels)
                logger.info("Auto-detected %d motor channel(s): %s",
                            len(detected_channels),
                            [f"C{c}" for c in detected_channels])

        # ── Step 4: auto-patch battery cell count from BAT voltage ────────────
        bat_df = dataframes.get("BAT")
        if bat_df is None:
            bat_df = dataframes.get("BAT2")
        if bat_df is not None and "Volt" in bat_df.columns and len(bat_df) > 0:
            start_v = float(bat_df["Volt"].iloc[0])
            detected_cells = _detect_cells_from_voltage(start_v)
            if detected_cells and detected_cells != profile.battery.cell_count:
                profile.battery.cell_count     = detected_cells
                profile.battery.nominal_voltage = round(detected_cells * 3.7, 1)
                profile.battery.full_voltage    = round(detected_cells * 4.2, 1)
                logger.info("Auto-detected %dS battery from %.2f V pack voltage.",
                            detected_cells, start_v)

        return profile

    @classmethod
    def _from_dict(cls, data: Dict) -> "DroneProfile":
        profile = cls()

        scalar_fields = [
            "id", "name", "type", "expected_endurance_min",
            "expected_range_km", "max_payload_kg", "max_speed_ms", "max_altitude_m",
        ]
        for key in scalar_fields:
            if key in data:
                setattr(profile, key, data[key])

        def _apply(cfg_obj, sub_dict: Optional[Dict]):
            if not sub_dict:
                return
            for k, v in sub_dict.items():
                if hasattr(cfg_obj, k):
                    setattr(cfg_obj, k, v)

        _apply(profile.battery, data.get("battery"))
        _apply(profile.motors, data.get("motors"))
        _apply(profile.vibration, data.get("vibration"))
        _apply(profile.gps, data.get("gps"))
        _apply(profile.maintenance, data.get("maintenance"))

        if "pusher_motor" in data and data["pusher_motor"]:
            profile.pusher_motor = PusherMotorConfig()
            _apply(profile.pusher_motor, data["pusher_motor"])

        if "scoring_weights" in data:
            profile.scoring_weights.update(data["scoring_weights"])

        if "thresholds" in data:
            profile.thresholds.update(data["thresholds"])

        return profile
