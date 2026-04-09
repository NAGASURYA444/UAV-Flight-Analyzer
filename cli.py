"""
UAV Flight Log Analyzer — CLI
==============================
Usage:
    python cli.py analyze <log.bin> [--profile profiles/default.yaml] [--output report.json]
    python cli.py analyze <log.bin> --drone-id UAV-001
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ── Lazy imports (keeps startup fast even if numpy/pandas not installed yet) ──
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool):
    """UAV Flight Log Analyzer — Phase 4.8"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )


@cli.command()
@click.argument("log_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--profile", "-p",
    default=None,
    show_default=True,
    help="Path to a drone YAML profile. Defaults to profiles/default.yaml.",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Write JSON report to this path. Defaults to <log_name>_report.json.",
)
@click.option(
    "--no-report",
    is_flag=True,
    default=False,
    help="Do not write a JSON file; only show terminal output.",
)
@click.option(
    "--html",
    "generate_html",
    is_flag=True,
    default=False,
    help="Also generate a self-contained HTML report alongside the JSON report.",
)
@click.option(
    "--drone-id",
    "drone_id",
    default=None,
    help="Unique drone identifier (e.g. 'Insight-4'). Overrides the profile name in the fleet DB.",
)
def analyze(
    log_file: str,
    profile: Optional[str],
    output: Optional[str],
    no_report: bool,
    generate_html: bool,
    drone_id: Optional[str],
):
    """
    Analyse an ArduPilot DataFlash .bin log file.

    LOG_FILE  Path to the .bin file to analyse.
    """
    _print_header()

    # ── Imports (deferred so --help is instant) ───────────────────────────────
    from analyzer.parser.bin_parser import parse_bin
    from analyzer.config.drone_profile import DroneProfile
    from analyzer.modules import flight_overview, battery, motors, vibration, sensors, control, efficiency, rc_link, landing_wind, fc_health, vtol_transition, mission, airspeed, pid_tuning, power_rail, telemetry, esc_telemetry, altitude_control
    from analyzer.scoring import engine as scoring_engine
    from analyzer.report import json_report, html_report

    # ── Parse first (needed for auto-detection when no profile given) ─────────
    with console.status(f"[bold cyan]Parsing {Path(log_file).name} ...[/]"):
        try:
            parse_result = parse_bin(log_file)
        except Exception as exc:
            console.print(f"[bold red]PARSE ERROR:[/] {exc}")
            sys.exit(1)

    console.print(
        f"  Parsed  : [green]{parse_result.message_count:,}[/] messages  "
        f"| [green]{len(parse_result.available_types)}[/] message types  "
        f"| Duration [green]{parse_result.duration_s / 60:.1f} min[/]\n"
    )

    # ── Profile ───────────────────────────────────────────────────────────────
    if profile is None:
        with console.status("[bold cyan]Auto-detecting drone profile from log...[/]"):
            drone_profile = DroneProfile.from_log(parse_result)
        console.print(
            f"  Profile : [cyan]{drone_profile.motor_category}[/]  "
            f"([dim]auto-detected — use --profile for fleet-specific thresholds[/])"
        )
    else:
        with console.status("[bold cyan]Loading drone profile...[/]"):
            drone_profile = DroneProfile.load(profile)
        console.print(f"  Profile : [cyan]{drone_profile.motor_category}[/]  ([dim]{drone_profile.id}[/])")

    console.print(f"  Type    : [cyan]{drone_profile.type}[/]")
    console.print(f"  Expect  : [cyan]{drone_profile.expected_endurance_min:.0f} min[/] endurance")
    if drone_id:
        console.print(f"  Drone ID: [bold cyan]{drone_id}[/]")
    console.print()

    # ── Analyse ───────────────────────────────────────────────────────────────
    with console.status("[bold cyan]Running analysis modules...[/]"):
        # Flight overview (must run first — provides arm_time)
        overview_result = flight_overview.analyse(parse_result, drone_profile)
        arm_time = overview_result["metrics"].get("arm_time_s", 0.0)

        # Motor analysis
        motors_result = motors.analyse(parse_result, drone_profile, arm_time=arm_time)

        # Vibration analysis
        vibration_result = vibration.analyse(parse_result, drone_profile, arm_time=arm_time)

        # Sensor health analysis
        sensors_result = sensors.analyse(parse_result, drone_profile, arm_time=arm_time)

        # Extract disarm time before battery/control/efficiency analyses
        disarm_time = overview_result["metrics"].get("disarm_time_s")

        # Battery analysis (needs disarm_time to exclude post-shutdown data)
        battery_result = battery.analyse(parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time)

        # Control performance analysis
        control_result = control.analyse(parse_result, drone_profile,
                                         arm_time=arm_time, disarm_time=disarm_time)

        # Power efficiency analysis (Phase 3)
        efficiency_result = efficiency.analyse(parse_result, drone_profile,
                                               arm_time=arm_time, disarm_time=disarm_time)

        # RC Link & Failsafe Events (Phase 4.1)
        rc_link_result = rc_link.analyse(parse_result, drone_profile, arm_time=arm_time)

        # Landing Quality & Wind Analysis (Phase 4.2)
        landing_wind_result = landing_wind.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # FC Power & Error Health (Phase 4.3)
        fc_health_result = fc_health.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # VTOL Transition Analysis (Phase 4.3 — vtol type only)
        vtol_transition_result = vtol_transition.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # Mission Execution Quality (Phase 4.4)
        mission_result = mission.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # Airspeed Health (Phase 4.4 — fixed_wing and vtol only)
        airspeed_result = airspeed.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # PID Tuning Quality (Phase 4.5 — all types; unavailable if PIDR/PIDP absent)
        pid_tuning_result = pid_tuning.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # Power Rail Health (Phase 4.5 — all types; analyses main battery bus)
        power_rail_result = power_rail.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # Telemetry Link Quality (Phase 4.6 — all types; unavailable if RADIO absent)
        telemetry_result = telemetry.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # ESC Telemetry Health (Phase 4.7 — all types; unavailable if ESC telemetry absent)
        esc_telemetry_result = esc_telemetry.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

        # Altitude Control Quality (Phase 4.8 — all types; uses CTUN messages)
        altitude_control_result = altitude_control.analyse(
            parse_result, drone_profile, arm_time=arm_time, disarm_time=disarm_time
        )

    module_results = {
        "flight_overview":   overview_result,
        "battery":           battery_result,
        "motors":            motors_result,
        "vibration":         vibration_result,
        "sensors":           sensors_result,
        "control":           control_result,
        "efficiency":        efficiency_result,
        "rc_link":           rc_link_result,
        "landing_wind":      landing_wind_result,
        "fc_health":         fc_health_result,
        "vtol_transition":   vtol_transition_result,
        "mission":           mission_result,
        "airspeed":          airspeed_result,
        "pid_tuning":        pid_tuning_result,
        "power_rail":        power_rail_result,
        "telemetry":         telemetry_result,
        "esc_telemetry":     esc_telemetry_result,
        "altitude_control":  altitude_control_result,
    }

    # Phase 4: Enrich endurance root causes with cross-module data
    module_results["flight_overview"] = flight_overview.post_analyse(
        module_results["flight_overview"], module_results, drone_profile
    )

    # ── Score ─────────────────────────────────────────────────────────────────
    scoring_result = scoring_engine.compute(module_results, drone_profile)

    # ── Display ───────────────────────────────────────────────────────────────
    _print_flight_overview(overview_result)
    _print_module_result("BATTERY ANALYSIS", battery_result, "BAT")
    _print_motors_result(motors_result)
    _print_vibration_result(vibration_result)
    _print_sensors_result(sensors_result)
    _print_control_result(control_result)
    _print_efficiency_result(efficiency_result)
    _print_rc_link_result(rc_link_result)
    _print_landing_wind_result(landing_wind_result)
    _print_fc_health_result(fc_health_result)
    _print_vtol_transition_result(vtol_transition_result)
    _print_mission_result(mission_result)
    _print_airspeed_result(airspeed_result)
    _print_pid_tuning_result(pid_tuning_result)
    _print_power_rail_result(power_rail_result)
    _print_telemetry_result(telemetry_result)
    _print_esc_telemetry_result(esc_telemetry_result)
    _print_altitude_control_result(altitude_control_result)
    _print_all_issues(scoring_result["all_issues"])
    _print_recommendations(scoring_result["recommendations"])
    _print_verdict(scoring_result)

    # ── Write report ──────────────────────────────────────────────────────────
    if not no_report:
        if output is None:
            stem = Path(log_file).stem
            output = str(Path(log_file).parent / f"{stem}_report.json")

        # Use the log file's modification time as the flight date.
        # This is the most reliable proxy — the .bin file is written during flight,
        # so its mtime is close to the actual flight date even when analysed later.
        from datetime import timezone as _tz
        _log_mtime = os.path.getmtime(log_file)
        _log_start_time = datetime.fromtimestamp(_log_mtime, tz=_tz.utc).isoformat()

        parse_meta = {
            "filepath": parse_result.filepath,
            "message_count": parse_result.message_count,
            "available_types": parse_result.available_types,
            "duration_s": round(parse_result.duration_s, 2),
            "log_start_time": _log_start_time,
        }
        profile_info = {
            "id": drone_profile.id,
            "name": drone_profile.name,
            "type": drone_profile.type,
        }

        report_dict = json_report.generate(
            parse_result_meta=parse_meta,
            module_results=module_results,
            scoring_result=scoring_result,
            profile_info=profile_info,
            output_path=output,
        )

        # Override drone name with --drone-id if supplied
        if drone_id:
            report_dict["meta"]["drone_name"] = drone_id
            # Re-write the JSON file with the updated drone_name
            with open(output, "w", encoding="utf-8") as _f:
                json.dump(report_dict, _f, indent=2, default=str)

        console.print(f"\n  JSON report : [green]{output}[/]")

        # ── Phase 6: auto-save to fleet DB ───────────────────────────────────
        try:
            from fleet import db as fleet_db, trend as fleet_trend
            flight_id = fleet_db.save_flight(report_dict)
            drone_name = report_dict.get("meta", {}).get("drone_name", "Unknown")
            history = fleet_db.get_drone_history(drone_name, limit=20)
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
            console.print(
                f"  Fleet DB    : [cyan]saved (flight #{flight_id}, "
                f"{len(history)} flights for {drone_name})[/]"
            )
        except Exception as _fleet_err:
            console.print(f"  Fleet DB    : [yellow]skipped ({_fleet_err})[/]")

        if generate_html:
            html_path = str(Path(output).with_suffix(".html"))
            html_report.generate(report_dict, output_path=html_path)
            console.print(f"  HTML report : [green]{html_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_header():
    console.print()
    console.print(Panel.fit(
        "[bold white]UAV FLIGHT LOG ANALYZER[/]  [dim]Phase 4.8[/]",
        border_style="cyan",
        padding=(0, 4),
    ))
    console.print()


def _print_flight_overview(result: dict):
    m = result.get("metrics", {})
    issues = result.get("issues", [])
    score = result.get("score", 0)
    grade = result.get("grade", "?")

    console.rule(f"[bold]FLIGHT OVERVIEW  [{_grade_color(grade)}]{grade}[/] {score:.0f}/100[/]")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=22)
    grid.add_column()

    # Duration
    airborne_s = m.get("airborne_s", 0)
    armed_s = m.get("armed_duration_s", 0)
    grid.add_row("Airborne time", f"[bold]{_fmt_dur(airborne_s)}[/]  (armed: {_fmt_dur(armed_s)})")

    # Endurance comparison
    endo = m.get("endurance", {})
    actual_min = endo.get("actual_min", 0)
    expected_min = endo.get("expected_min", 0)
    delta_pct = endo.get("delta_pct", 0)
    delta_str = f"[green]+{delta_pct:.1f}%[/]" if delta_pct >= 0 else f"[red]{delta_pct:.1f}%[/]"
    grid.add_row("Endurance", f"{actual_min:.1f} min  (expected: {expected_min:.0f} min)  {delta_str}")

    # Endurance root causes
    causes = endo.get("root_causes", [])
    if causes:
        grid.add_row("  Root cause(s)", "[yellow]" + "; ".join(causes) + "[/]")

    if m.get("max_altitude_msl_m") is not None or m.get("max_altitude_agl_m") is not None:
        msl = m.get("max_altitude_msl_m")
        agl = m.get("max_altitude_agl_m")
        parts = []
        if msl is not None:
            parts.append(f"{msl:.0f} m MSL")
        if agl is not None:
            parts.append(f"[bold]{agl:.0f} m AGL[/]")
        grid.add_row("Max altitude", "  /  ".join(parts))
    if m.get("max_speed_ms") is not None:
        grid.add_row("Max speed", f"{m['max_speed_ms']:.1f} m/s")
    if m.get("distance_km") is not None:
        grid.add_row("Distance", f"{m['distance_km']:.3f} km")

    phases = m.get("phase_sequence", "—")
    grid.add_row("Phase sequence", f"[cyan]{phases}[/]")

    mode_count = m.get("mode_change_count", 0)
    grid.add_row("Mode changes", str(mode_count))

    console.print(grid)
    console.print()


def _print_module_result(title: str, result: dict, prefix: str):
    score = result.get("score", 0)
    grade = result.get("grade", "?")
    available = result.get("available", True)
    summary = result.get("summary", "")
    issues = result.get("issues", [])

    color = _grade_color(grade)
    console.rule(f"[bold]{title}  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim yellow]Data not available in this log.[/]")
        console.print()
        return

    if summary:
        console.print(f"  {summary}")

    if issues:
        for issue in issues:
            _print_issue_line(issue)

    console.print()


def _print_motors_result(result: dict):
    score = result.get("score", 0)
    grade = result.get("grade", "?")
    available = result.get("available", True)
    issues = result.get("issues", [])
    m = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]MOTOR ANALYSIS  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim yellow]{m.get('unavailable_reason', 'Motor data not available in this log.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    # Overall throttle
    avg_thr = m.get("avg_throttle_pct")
    p95_thr = m.get("p95_throttle_pct")
    if avg_thr is not None:
        grid.add_row("Avg / P95 throttle",
                     f"[bold]{avg_thr:.1f}%[/]  /  {p95_thr:.1f}%")

    # Hover comparison
    median_thr = m.get("median_throttle_pct")
    expected_thr = m.get("expected_hover_throttle_pct")
    delta = m.get("hover_throttle_delta_pct")
    if median_thr is not None and expected_thr is not None:
        delta_color = "yellow" if abs(delta) > 10 else "green"
        grid.add_row("Median / Expected hover",
                     f"{median_thr:.1f}%  /  {expected_thr:.0f}%  "
                     f"[{delta_color}](diff {delta:+.1f}%)[/]")

    # Symmetry
    sym = m.get("symmetry_score")
    avg_imb = m.get("avg_motor_imbalance_pct")
    max_imb = m.get("max_motor_imbalance_pct")
    if sym is not None:
        sym_color = "red" if sym < 60 else ("yellow" if sym < 80 else "green")
        grid.add_row("Symmetry score",
                     f"[{sym_color}]{sym:.0f}/100[/]  "
                     f"(avg imbalance: {avg_imb:.1f}%  max: {max_imb:.1f}%)")

    # Oscillation
    osc = m.get("oscillation_index")
    if osc is not None:
        osc_color = "red" if osc > 6.0 else ("yellow" if osc > 3.0 else "green")
        grid.add_row("Oscillation index", f"[{osc_color}]{osc:.2f}[/]  (>3.0 = PID concern, >6.0 = critical)")

    # High throttle time
    frac_high = m.get("frac_time_high_throttle", 0.0)
    fh_color = "red" if frac_high > 0.25 else ("yellow" if frac_high > 0.10 else "green")
    grid.add_row("High-throttle time", f"[{fh_color}]{frac_high*100:.1f}%[/]")

    console.print(grid)

    # Per-motor table
    per_motor = m.get("per_motor", [])
    if per_motor:
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="dim",
                    padding=(0, 1))
        tbl.add_column("Motor", width=7)
        tbl.add_column("Avg %", width=7)
        tbl.add_column("Min %", width=7)
        tbl.add_column("Max %", width=7)
        tbl.add_column("Std %", width=7)
        tbl.add_column("Saturation", width=12)

        sat_map = {s["channel"]: s for s in m.get("motor_saturation", [])}
        dead = m.get("dead_motors", [])

        for pm in per_motor:
            ch = pm["channel"]
            sat = sat_map.get(ch, {})
            sat_count = sat.get("saturation_count", 0)
            sat_str = f"[red]{sat_count} events[/]" if sat_count > 0 else "[dim]none[/]"
            ch_style = "bold red" if ch in dead else ""
            tbl.add_row(
                f"[{ch_style}]{ch}[/]" if ch_style else ch,
                f"{pm['avg_throttle_pct']:.1f}",
                f"{pm['min_throttle_pct']:.1f}",
                f"{pm['max_throttle_pct']:.1f}",
                f"{pm['std_throttle_pct']:.1f}",
                sat_str,
            )
        console.print(tbl)

    if issues:
        for issue in issues:
            _print_issue_line(issue)
    console.print()


def _print_vibration_result(result: dict):
    score = result.get("score", 0)
    grade = result.get("grade", "?")
    available = result.get("available", True)
    issues = result.get("issues", [])
    m = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]VIBRATION ANALYSIS  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print("  [dim yellow]No VIBE or IMU data available in this log.[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=24)
    grid.add_column()

    # Mean vibe per axis (flat keys from vibration module)
    mx = m.get("mean_vibe_x")
    my = m.get("mean_vibe_y")
    mz = m.get("mean_vibe_z")
    if mx is not None:
        grid.add_row("Mean vibe X/Y/Z", f"{mx:.1f} / {my:.1f} / {mz:.1f} m/s²")

    px = m.get("peak_vibe_x")
    py = m.get("peak_vibe_y")
    pz = m.get("peak_vibe_z")
    if px is not None:
        grid.add_row("Peak vibe X/Y/Z", f"{px:.1f} / {py:.1f} / {pz:.1f} m/s²")

    total_clips = m.get("total_clipping_events")
    if total_clips is not None:
        clip_color = "red" if total_clips > 10 else ("yellow" if total_clips > 0 else "green")
        clip_rate = m.get("clipping_per_min", 0.0)
        grid.add_row("Clip events (total)", f"[{clip_color}]{total_clips}[/]  ({clip_rate:.1f}/min)")

    if m.get("fft_available"):
        dom_freq = m.get("dominant_freq_hz")
        dom_band = m.get("dominant_band", "—")
        if dom_freq is not None:
            grid.add_row("Dominant frequency", f"{dom_freq:.1f} Hz  [{dom_band}]")

    trend_dir = m.get("vibration_trend", "stable")
    t_color = "red" if trend_dir == "increasing_critical" else (
              "yellow" if trend_dir == "increasing" else "green")
    grid.add_row("Vibration trend", f"[{t_color}]{trend_dir}[/]")

    console.print(grid)
    if issues:
        for issue in issues:
            _print_issue_line(issue)
    console.print()


def _print_sensors_result(result: dict):
    score = result.get("score", 0)
    grade = result.get("grade", "?")
    available = result.get("available", True)
    issues = result.get("issues", [])
    m = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]SENSOR HEALTH  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print("  [dim yellow]No sensor data available in this log.[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=24)
    grid.add_column()

    # GPS
    gps = m.get("gps", {})
    if gps:
        fix_pct = gps.get("fix_pct_good", 0)           # key from sensors module
        avg_sats = gps.get("avg_satellites", 0)
        avg_hdop = gps.get("avg_hdop")
        min_sats = gps.get("min_satellites")
        fix_color = "green" if fix_pct >= 95 else ("yellow" if fix_pct >= 80 else "red")
        grid.add_row("GPS fix quality",
                     f"[{fix_color}]{fix_pct:.0f}% good[/]  "
                     f"avg {avg_sats:.0f} sats  (min: {min_sats})"
                     + (f"  HDOP {avg_hdop:.2f}" if avg_hdop is not None else ""))

    # EKF
    ekf = m.get("ekf", {})
    if ekf:
        inno = ekf.get("avg_velocity_innovation")      # key from sensors module
        fault_pct = ekf.get("ekf_fault_pct", 0)       # key from sensors module
        healthy_pct = ekf.get("ekf_healthy_pct")
        if inno is not None:
            inno_color = "red" if inno > 1.0 else ("yellow" if inno > 0.5 else "green")
            h_str = f"  healthy: {healthy_pct:.0f}%" if healthy_pct is not None else ""
            grid.add_row("EKF velocity innovation", f"[{inno_color}]{inno:.3f}[/]  (fault: {fault_pct:.1f}%{h_str})")

    # IMU temperature
    imu = m.get("imu", {})
    if imu:
        temp = imu.get("imu_temp_max_c")               # key from sensors module
        if temp is not None:
            t_color = "red" if temp > 85 else ("yellow" if temp > 70 else "green")
            grid.add_row("IMU temperature (max)", f"[{t_color}]{temp:.1f} °C[/]")

    # Magnetometer
    mag = m.get("mag", {})
    if mag:
        cv = mag.get("mag_field_cv")                   # key from sensors module
        spikes = mag.get("mag_interference_spikes", 0) # key from sensors module
        if cv is not None:
            mag_color = "red" if cv > 0.40 else ("yellow" if cv > 0.20 else "green")
            grid.add_row("Mag field variation (CV)", f"[{mag_color}]{cv:.3f}[/]  ({spikes} interference spikes)")

    console.print(grid)
    if issues:
        for issue in issues:
            _print_issue_line(issue)
    console.print()


def _print_control_result(result: dict):
    score = result.get("score", 0)
    grade = result.get("grade", "?")
    available = result.get("available", True)
    issues = result.get("issues", [])
    m = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]CONTROL PERFORMANCE  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        reason = m.get("unavailable_reason", "ATT data not available in this log.")
        console.print(f"  [dim yellow]{reason}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    roll_rms = m.get("roll_rms_deg")
    pitch_rms = m.get("pitch_rms_deg")
    if roll_rms is not None and pitch_rms is not None:
        r_color = "red" if roll_rms > 3.5 else ("yellow" if roll_rms > 1.5 else "green")
        p_color = "red" if pitch_rms > 3.5 else ("yellow" if pitch_rms > 1.5 else "green")
        grid.add_row("Roll / Pitch RMS error",
                     f"[{r_color}]{roll_rms:.2f}°[/] / [{p_color}]{pitch_rms:.2f}°[/]  "
                     f"(max: {m.get('roll_max_err_deg', 0):.1f}° / {m.get('pitch_max_err_deg', 0):.1f}°)")

    yaw_rms = m.get("yaw_rms_deg")
    if yaw_rms is not None:
        y_color = "red" if yaw_rms > 6.0 else ("yellow" if yaw_rms > 3.0 else "green")
        straight_note = " [dim](straight legs only)[/]" if m.get("yaw_straight_only") else ""
        excluded = m.get("yaw_turn_excluded", 0)
        excl_note = f"  [dim]{excluded} turn samples excluded[/]" if excluded > 0 else ""
        grid.add_row("Yaw RMS error",
                     f"[{y_color}]{yaw_rms:.2f}°[/]  "
                     f"(max incl. turns: {m.get('yaw_max_err_deg', 0):.1f}°)"
                     f"{straight_note}{excl_note}")
    elif "has_yaw" in m:
        # Fixed-wing and VTOL do not track yaw error — directional control is via bank angle
        grid.add_row("Yaw RMS error",
                     "[dim]N/A — not scored for fixed-wing / VTOL "
                     "(direction controlled by banking, not a yaw actuator)[/]")

    large_pct = m.get("large_error_pct")
    if large_pct is not None:
        lp_color = "red" if large_pct > 10 else ("yellow" if large_pct > 5 else "green")
        grid.add_row("Large-error time", f"[{lp_color}]{large_pct:.1f}%[/]")

    roll_osc = m.get("roll_osc_rms_deg")
    pitch_osc = m.get("pitch_osc_rms_deg")
    if roll_osc is not None:
        max_osc = max(roll_osc, pitch_osc or 0.0)
        osc_color = "red" if max_osc > 4.0 else ("yellow" if max_osc > 2.0 else "green")
        grid.add_row("Oscillation (window RMS)", f"[{osc_color}]{max_osc:.2f}°[/]")

    console.print(grid)
    if issues:
        for issue in issues:
            _print_issue_line(issue)
    console.print()


def _print_efficiency_result(result: dict):
    score = result.get("score", 0)
    grade = result.get("grade", "?")
    available = result.get("available", True)
    issues = result.get("issues", [])
    m = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]POWER EFFICIENCY  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        reason = m.get("unavailable_reason", "BAT data not available in this log.")
        console.print(f"  [dim yellow]{reason}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    energy_wh = m.get("energy_wh")
    mah = m.get("mah_consumed")
    if energy_wh is not None:
        grid.add_row("Energy consumed",
                     f"[bold]{energy_wh:.1f} Wh[/]  ({mah:.0f} mAh)" if mah else f"[bold]{energy_wh:.1f} Wh[/]")

    dist = m.get("distance_km")
    if dist is not None:
        grid.add_row("Distance flown", f"{dist:.3f} km")

    wh_km = m.get("wh_per_km")
    exp_wh_km = m.get("expected_wh_per_km")
    if wh_km is not None:
        ratio = wh_km / exp_wh_km if exp_wh_km and exp_wh_km > 0 else 1.0
        eff_color = "red" if ratio > 1.5 else ("yellow" if ratio > 1.2 else "green")
        exp_str = f"  (expected: {exp_wh_km:.1f})" if exp_wh_km else ""
        grid.add_row("Efficiency (Wh/km)",
                     f"[{eff_color}]{wh_km:.1f} Wh/km[/]{exp_str}")

    mah_km = m.get("mah_per_km")
    if mah_km is not None:
        grid.add_row("Efficiency (mAh/km)", f"{mah_km:.0f} mAh/km")

    avg_w = m.get("avg_power_w")
    max_w = m.get("max_rated_power_w")
    if avg_w is not None:
        pw_color = "red" if (max_w and avg_w > max_w) else (
                   "yellow" if (max_w and avg_w > max_w * 0.9) else "green")
        max_str = f"  (rated max: {max_w:.0f} W)" if max_w else ""
        grid.add_row("Average power", f"[{pw_color}]{avg_w:.0f} W[/]{max_str}")

    wh_min = m.get("wh_per_min")
    if wh_min:
        grid.add_row("Energy rate", f"{wh_min:.3f} Wh/min")

    # VTOL phase split
    if m.get("phase_split_available"):
        hover_pct  = m.get("hover_time_pct", 0)
        cruise_pct = m.get("cruise_time_pct", 0)
        hover_wh   = m.get("hover_energy_wh", 0)
        cruise_wh  = m.get("cruise_energy_wh", 0)
        hover_w    = m.get("hover_avg_power_w", 0)
        cruise_w   = m.get("cruise_avg_power_w", 0)
        grid.add_row("Hover / Cruise time",
                     f"{hover_pct:.0f}% / {cruise_pct:.0f}%")
        grid.add_row("Hover / Cruise energy",
                     f"{hover_wh:.1f} Wh / {cruise_wh:.1f} Wh")
        grid.add_row("Hover / Cruise avg power",
                     f"{hover_w:.0f} W / {cruise_w:.0f} W")

    console.print(grid)
    if issues:
        for issue in issues:
            _print_issue_line(issue)
    console.print()


def _print_rc_link_result(result: dict):
    score  = result.get("score", 0)
    grade  = result.get("grade", "?")
    issues = result.get("issues", [])
    m      = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]RC LINK & FAILSAFES  [{color}]{grade}[/] {score:.0f}/100[/]")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    # RC mode
    rc_mode = m.get("rc_mode", "autonomous")
    mode_labels = {
        "autonomous":          "[dim]Autonomous (no RC)[/]",
        "rc_connected_unused": "[dim]RC connected — not used[/]",
        "rc_active":           "[green]RC actively used[/]",
    }
    grid.add_row("RC mode", mode_labels.get(rc_mode, rc_mode))

    # RC dropouts (only shown if RC was active)
    if rc_mode == "rc_active":
        dropout_count = m.get("rc_dropout_count", 0)
        dropout_s     = m.get("rc_dropout_total_s", 0.0)
        d_color = "red" if dropout_count > 3 else ("yellow" if dropout_count > 0 else "green")
        grid.add_row(
            "RC dropouts",
            f"[{d_color}]{dropout_count}[/]"
            + (f"  (total {dropout_s:.1f} s lost)" if dropout_count > 0 else "  [dim](none)[/]"),
        )

        sat_channels = m.get("rc_saturated_channels", [])
        if sat_channels:
            sat_str = ", ".join(s["channel"] for s in sat_channels)
            grid.add_row("Saturated channels", f"[yellow]{sat_str}[/]")

    # Failsafe events
    fsf_count  = m.get("failsafe_event_count", 0)
    fsf_events = m.get("failsafe_events", [])
    fsf_color  = "red" if fsf_count > 0 else "green"
    grid.add_row(
        "Failsafe events",
        f"[{fsf_color}]{fsf_count}[/]"
        + ("  [dim](none triggered)[/]" if fsf_count == 0 else ""),
    )

    # List each failsafe event with timestamp
    for evt in fsf_events:
        grid.add_row(
            f"  {evt['subsystem']}",
            f"[red]triggered at T+{evt['timestamp_s']:.1f} s[/]",
        )

    console.print(grid)

    if issues:
        for issue in issues:
            sev = issue.get("severity", "info")
            # Suppress plain INFO lines that are already shown in the grid
            if sev == "info" and issue.get("code") in ("RCL-001", "RCL-002"):
                continue
            _print_issue_line(issue)

    console.print()


def _print_landing_wind_result(result: dict):
    score  = result.get("score", 0)
    grade  = result.get("grade", "?")
    issues = result.get("issues", [])
    m      = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]LANDING QUALITY & WIND  [{color}]{grade}[/] {score:.0f}/100[/]")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    # ── Landing section ───────────────────────────────────────────────────────
    if m.get("landing_detected"):
        td = m.get("touchdown_s")
        grid.add_row("Landing detected", f"T+{td:.1f} s" if td is not None else "Yes")

        dr = m.get("descent_rate_ms")
        if dr is not None:
            dr_color = "red" if dr >= 3.0 else ("yellow" if dr >= 1.5 else "green")
            label = "Hard landing" if dr >= 3.0 else ("Rough" if dr >= 1.5 else "Normal")
            grid.add_row("Descent rate at TD",
                         f"[{dr_color}]{dr:.2f} m/s[/]  [dim]({label})[/]")

        hs = m.get("horizontal_speed_ms")
        if hs is not None:
            hs_color = "yellow" if hs > 2.0 else "green"
            grid.add_row("Horizontal speed at TD", f"[{hs_color}]{hs:.2f} m/s[/]")

        bounce = m.get("bounce_detected", False)
        b_color = "yellow" if bounce else "green"
        grid.add_row("Bounce detected",
                     f"[{b_color}]{'Yes' if bounce else 'No'}[/]")

        roll_rms  = m.get("approach_roll_rms_deg")
        pitch_rms = m.get("approach_pitch_rms_deg")
        if roll_rms is not None and pitch_rms is not None:
            ap_color = "yellow" if max(roll_rms, pitch_rms) > 5.0 else "green"
            grid.add_row("Approach stability",
                         f"[{ap_color}]Roll {roll_rms:.1f}° / Pitch {pitch_rms:.1f}° RMS[/]")
    else:
        grid.add_row("Landing detected", "[dim]Not detected in log window[/]")

    # ── Wind section ──────────────────────────────────────────────────────────
    if m.get("wind_available"):
        ws     = m.get("wind_speed_ms", 0.0)
        ws_std = m.get("wind_speed_std_ms")
        wdir   = m.get("wind_direction_deg")
        method = m.get("wind_method", "")

        ws_color = "red" if ws > 10 else ("yellow" if ws > 7 else "green")
        dir_str  = f" from {wdir:.0f}°" if wdir is not None else ""
        std_str  = f"  (±{ws_std:.1f} m/s)" if ws_std is not None else ""
        grid.add_row("Wind estimate",
                     f"[{ws_color}]{ws:.1f} m/s{dir_str}[/]{std_str}"
                     f"  [dim][{method}][/]")
    else:
        grid.add_row("Wind estimate", "[dim]N/A — insufficient data[/]")

    console.print(grid)

    if issues:
        for issue in issues:
            # Suppress WND-001 (always-info) from the issue list — already shown in grid
            if issue.get("code") == "WND-001":
                continue
            _print_issue_line(issue)

    console.print()


def _print_fc_health_result(result: dict):
    score  = result.get("score", 0)
    grade  = result.get("grade", "?")
    issues = result.get("issues", [])
    m      = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]FC POWER & ERROR HEALTH  [{color}]{grade}[/] {score:.0f}/100[/]")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    powr_avail = m.get("powr_available", False)
    if powr_avail:
        vcc_min  = m.get("vcc_min_v")
        vcc_mean = m.get("vcc_mean_v")
        vcc_std  = m.get("vcc_std_v", 0.0)
        if vcc_min is not None:
            vcc_color = "red" if vcc_min < 4.3 else ("yellow" if vcc_min < 4.6 else "green")
            grid.add_row("VCC rail (min / mean)",
                         f"[{vcc_color}]{vcc_min:.3f} V[/]  /  {vcc_mean:.3f} V"
                         + (f"  (std {vcc_std:.3f} V)" if vcc_std else ""))

        servo_avail = m.get("servo_rail_available", False)
        if servo_avail:
            srv_min  = m.get("servo_min_v")
            srv_mean = m.get("servo_mean_v")
            if srv_min is not None:
                srv_color = "red" if srv_min < 4.3 else ("yellow" if srv_min < 4.7 else "green")
                grid.add_row("Servo rail (min / mean)",
                             f"[{srv_color}]{srv_min:.3f} V[/]  /  {srv_mean:.3f} V")
    else:
        grid.add_row("Power rail", "[dim]POWR data not available in this log[/]")

    err_count  = m.get("fc_error_count", 0)
    err_color  = "red" if err_count > 0 else "green"
    grid.add_row("Firmware error events",
                 f"[{err_color}]{err_count}[/]"
                 + ("  [dim](none)[/]" if err_count == 0 else ""))

    err_events = m.get("fc_error_events", [])
    for evt in err_events:
        if evt.get("status") == "triggered":
            grid.add_row(
                f"  {evt['subsystem']}",
                f"[red]ECode {evt['ecode']} at T+{evt['timestamp_s']:.1f} s[/]",
            )

    console.print(grid)

    if issues:
        for issue in issues:
            if issue.get("code") not in ("FCH-007",):   # suppress cleared-info from list
                _print_issue_line(issue)

    console.print()


def _print_vtol_transition_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]VTOL TRANSITION QUALITY  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'Not applicable for this drone type.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    t_count = m.get("transition_count", 0)
    grid.add_row("Transitions detected",
                 str(t_count) if t_count > 0 else "[dim]None[/]")

    if t_count > 0:
        fw2vtol = m.get("fw_to_vtol_count", 0)
        vtol2fw = m.get("vtol_to_fw_count", 0)
        grid.add_row("  FW-to-VTOL / VTOL-to-FW", f"{fw2vtol}  /  {vtol2fw}")

        mean_score = m.get("mean_transition_score", 100.0)
        ms_color   = "red" if mean_score < 60 else ("yellow" if mean_score < 80 else "green")
        grid.add_row("Mean transition score",
                     f"[{ms_color}]{mean_score:.0f}/100[/]")

        worst_alt = m.get("worst_alt_loss_m", 0.0)
        if worst_alt > 0.5:
            alt_color = "red" if worst_alt > 15 else ("yellow" if worst_alt > 5 else "green")
            grid.add_row("Worst alt loss", f"[{alt_color}]{worst_alt:.1f} m[/]")

        worst_pitch = m.get("worst_pitch_spike_deg", 0.0)
        worst_roll  = m.get("worst_roll_spike_deg",  0.0)
        if max(worst_pitch, worst_roll) > 1.0:
            att_val   = max(worst_pitch, worst_roll)
            att_color = "red" if att_val > 25 else ("yellow" if att_val > 15 else "green")
            grid.add_row("Worst attitude spike",
                         f"[{att_color}]{att_val:.1f} deg[/]  "
                         f"(pitch {worst_pitch:.1f} / roll {worst_roll:.1f})")

        worst_ramp = m.get("slowest_ramp_s", 0.0)
        if worst_ramp > 0.0:
            ramp_color = "red" if worst_ramp > 6 else ("yellow" if worst_ramp > 3 else "green")
            grid.add_row("Slowest motor ramp", f"[{ramp_color}]{worst_ramp:.1f} s[/]")

        # Per-transition table
        transitions = m.get("transitions", [])
        if transitions:
            tbl = Table(box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1))
            tbl.add_column("Time (s)", width=10)
            tbl.add_column("Type", width=14)
            tbl.add_column("Modes", width=22)
            tbl.add_column("Alt loss", width=10)
            tbl.add_column("Att spike", width=10)
            tbl.add_column("Ramp (s)", width=9)
            tbl.add_column("Score", width=7)
            for tr in transitions:
                alt_s  = f"{tr.get('alt_drop_m', 0) or 0:.1f} m"   if tr.get('alt_drop_m')      is not None else "—"
                att_s  = f"{max(tr.get('pitch_spike_deg',0) or 0, tr.get('roll_spike_deg',0) or 0):.1f} deg" if (tr.get('pitch_spike_deg') or tr.get('roll_spike_deg')) else "—"
                ramp_s = f"{tr['ramp_time_s']:.1f}"                  if tr.get('ramp_time_s')     is not None else "—"
                ts_s   = f"T+{tr['timestamp_s']:.1f}"
                # AUTO > AUTO means transition happened inside AUTO (no mode change logged)
                _fm = tr['from_mode']; _tm = tr['to_mode']
                if _fm == _tm == "AUTO":
                    modes = "AUTO(VTOL) > AUTO(FW)" if tr["type"] == "vtol_to_fw" else "AUTO(FW) > AUTO(VTOL)"
                else:
                    modes = f"{_fm} > {_tm}"
                sc     = tr.get("score", 100)
                sc_col = "red" if sc < 60 else ("yellow" if sc < 80 else "green")
                tbl.add_row(ts_s, tr["type"].replace("_", " "), modes,
                            alt_s, att_s, ramp_s,
                            f"[{sc_col}]{sc:.0f}[/]")
            console.print(grid)
            console.print(tbl)
        else:
            console.print(grid)
    else:
        console.print(grid)

    if issues:
        for issue in issues:
            if issue.get("code") not in ("VTR-001",):   # VTR-001 info already in grid
                _print_issue_line(issue)

    console.print()


def _print_mission_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]MISSION EXECUTION QUALITY  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'Mission data not available.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    total  = m.get("total_waypoints", 0)
    done   = m.get("executed_waypoints", 0)
    pct    = m.get("completion_pct", 100.0)
    complete = m.get("mission_complete", True)
    c_color  = "green" if complete else ("yellow" if pct >= 50 else "red")
    grid.add_row("Waypoint completion",
                 f"[{c_color}]{done}/{total}  ({pct:.0f}%)[/]"
                 + ("  [green][complete][/]" if complete else "  [yellow][incomplete][/]"))

    mean_xte   = m.get("mean_cross_track_m")
    max_xte    = m.get("max_cross_track_m")
    max_xte_ts = m.get("max_xte_timestamp_s")
    max_xte_leg = m.get("max_xte_leg", "—")
    if mean_xte is not None:
        xte_color = "red" if mean_xte > 20 else ("yellow" if mean_xte > 10 else "green")
        max_loc = ""
        if max_xte is not None:
            max_loc = f"  (max {max_xte:.1f} m"
            if max_xte_ts is not None:
                max_loc += f" @ T+{max_xte_ts:.1f} s"
            if max_xte_leg and max_xte_leg != "—":
                max_loc += f"  {max_xte_leg}"
            max_loc += ")"
        grid.add_row("Cross-track error",
                     f"[{xte_color}]mean {mean_xte:.1f} m[/]{max_loc}")
    else:
        grid.add_row("Cross-track error", "[dim]N/A[/]")

    console.print(grid)
    if issues:
        for issue in issues:
            _print_issue_line(issue)
    console.print()


def _print_airspeed_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]AIRSPEED HEALTH  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'Airspeed data not available.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    mean_s = m.get("mean_speed_ms")
    min_s  = m.get("min_speed_ms")
    max_s  = m.get("max_speed_ms")
    if mean_s is not None:
        grid.add_row("Speed (mean / min / max)",
                     f"[bold]{mean_s:.1f}[/] / {min_s:.1f} / {max_s:.1f} m/s  [dim][GPS proxy][/]")

    stall_p = m.get("stall_time_pct", 0.0)
    stall_t = m.get("stall_proxy_ms")
    if stall_p is not None:
        st_color = "red" if stall_p > 5 else ("yellow" if stall_p > 1 else "green")
        grid.add_row("Below stall proxy",
                     f"[{st_color}]{stall_p:.1f}%[/]"
                     + (f"  [dim](proxy: {stall_t:.1f} m/s)[/]" if stall_t else ""))

    ospd_p = m.get("overspeed_pct", 0.0)
    ospd_t = m.get("overspeed_thr_ms")
    if ospd_p is not None:
        os_color = "yellow" if ospd_p > 2 else "green"
        grid.add_row("Above overspeed thr",
                     f"[{os_color}]{ospd_p:.1f}%[/]"
                     + (f"  [dim](thr: {ospd_t:.1f} m/s)[/]" if ospd_t else ""))

    vari = m.get("variability_ratio")
    if vari is not None:
        v_color = "yellow" if vari > 0.30 else "green"
        grid.add_row("Variability (std/mean)", f"[{v_color}]{vari:.3f}[/]")

    console.print(grid)
    if issues:
        for issue in issues:
            # suppress always-info proxy notice from issue list (already shown in grid)
            if issue.get("code") == "ASP-001":
                continue
            _print_issue_line(issue)
    console.print()


def _print_pid_tuning_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]PID TUNING QUALITY  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'PID data not available.')}[/]")
        console.print()
        return

    axes = m.get("axes_available", [])
    if not axes:
        console.print("  [dim yellow]No PID axes available for analysis.[/]")
        console.print()
        return

    # Per-axis metrics table
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1))
    tbl.add_column("Axis",     width=7)
    tbl.add_column("I-ratio",  width=9)
    tbl.add_column("D-noise",  width=9)
    tbl.add_column("Err-norm", width=10)
    tbl.add_column("I-bias",   width=8)
    tbl.add_column("Status", width=20)

    axis_keys = [
        ("roll",  "pid_roll",  "Roll"),
        ("pitch", "pid_pitch", "Pitch"),
        ("yaw",   "pid_yaw",   "Yaw"),
    ]
    for ax_key, m_key, ax_label in axis_keys:
        if ax_key not in axes:
            continue
        ax_stats = m.get(m_key, {})
        if not ax_stats:
            continue

        i_ratio  = ax_stats.get("i_ratio")
        d_noise  = ax_stats.get("d_noise_idx")
        err_norm = ax_stats.get("err_norm")
        i_bias   = ax_stats.get("i_bias")

        def _fmt(v, warn, crit):
            if v is None:
                return "[dim]—[/]"
            if crit is not None and v > crit:
                return f"[bold red]{v:.3f}[/]"
            if warn is not None and v > warn:
                return f"[yellow]{v:.3f}[/]"
            return f"[green]{v:.3f}[/]"

        i_ratio_s  = _fmt(i_ratio,  0.45, 0.65)
        d_noise_s  = _fmt(d_noise,  0.40, 0.80)
        err_norm_s = _fmt(err_norm, 0.25, 0.40)
        i_bias_s   = _fmt(i_bias,   1.5,  None)

        # Status text
        flags = []
        if i_ratio  is not None and i_ratio  > 0.45: flags.append("I-dom")
        if d_noise  is not None and d_noise  > 0.40: flags.append("D-noisy")
        if err_norm is not None and err_norm > 0.25: flags.append("err-high")
        if i_bias   is not None and i_bias   > 1.5:  flags.append("I-bias")
        status = "[yellow]" + ", ".join(flags) + "[/]" if flags else "[green]OK[/]"

        tbl.add_row(ax_label, i_ratio_s, d_noise_s, err_norm_s, i_bias_s, status)

    console.print(tbl)

    # P-gain asymmetry
    asym = m.get("rp_p_asymmetry_ratio")
    if asym is not None:
        a_color = "yellow" if asym > 2.0 else "green"
        console.print(
            f"  [dim]Roll/Pitch P-asymmetry:[/]  [{a_color}]{asym:.2f}[/]"
            + ("  [dim](> 2.0 = flag)[/]" if asym > 2.0 else "  [dim](OK)[/]")
        )

    if issues:
        for issue in issues:
            _print_issue_line(issue)

    console.print()


def _print_power_rail_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]POWER RAIL HEALTH  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'Power rail data not available.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=26)
    grid.add_column()

    # ── Main bus voltage stats ─────────────────────────────────────────────
    volt_mean = m.get("volt_mean_v")
    volt_min  = m.get("volt_min_v")
    volt_max  = m.get("volt_max_v")
    if volt_mean is not None:
        grid.add_row("Bus voltage (mean / min / max)",
                     f"[bold]{volt_mean:.1f} V[/]  /  {volt_min:.1f} V  /  {volt_max:.1f} V")

    # ── Brownout ───────────────────────────────────────────────────────────
    brownout_avail = m.get("brownout_check_available", False)
    if brownout_avail:
        brownout_n = m.get("brownout_count", 0)
        if m.get("brownout_pre_existing"):
            grid.add_row("Brownout check",
                         "[dim]Vcc change flag pre-existing (set before arming — pre-flight power event, not in-flight)[/]")
        elif brownout_n > 0:
            first_ts = m.get("brownout_first_ts_s")
            ts_str = f"  [dim](first @ T+{first_ts:.1f} s)[/]" if first_ts is not None else ""
            grid.add_row("Brownout events",
                         f"[bold red]DETECTED[/]{ts_str}")
        else:
            grid.add_row("Brownout events", "[green]None[/]  [dim](POWR.Flags VCC_CHANGED clear)[/]")
    else:
        grid.add_row("Brownout check", "[dim]POWR data not available[/]")

    # ── Voltage sag (peak current) ─────────────────────────────────────────
    if m.get("sag_available", False):
        peak_sag = m.get("peak_sag_pct")
        if peak_sag is not None:
            pk_thresh = m.get("peak_current_threshold_a")
            volt_pk   = m.get("volt_at_peak_current_v")
            ps_color  = "red" if peak_sag > 15.0 else ("yellow" if peak_sag > 8.0 else "green")
            thr_str   = f"  [dim](above {pk_thresh:.0f} A -> {volt_pk:.2f} V)[/]" if pk_thresh else ""
            grid.add_row("Peak-current sag",
                         f"[{ps_color}]{peak_sag:.1f}%[/]{thr_str}"
                         + ("  [dim](warn >8%, crit >15%)[/]" if peak_sag > 0 else ""))

        sus_sag = m.get("sustained_sag_pct")
        if sus_sag is not None:
            sus_thresh = m.get("sustained_current_threshold_a")
            volt_sus   = m.get("volt_at_sustained_current_v")
            ss_color   = "yellow" if sus_sag > 5.0 else "green"
            thr_str    = f"  [dim](above {sus_thresh:.0f} A -> {volt_sus:.2f} V)[/]" if sus_thresh else ""
            grid.add_row("Sustained-load sag",
                         f"[{ss_color}]{sus_sag:.1f}%[/]{thr_str}")
    else:
        grid.add_row("Voltage sag", "[dim]Current data not available — sag analysis skipped[/]")

    # ── Bus noise ──────────────────────────────────────────────────────────
    noise_cv = m.get("bus_noise_cv")
    if noise_cv is not None:
        nc_color = "yellow" if noise_cv > 0.06 else "green"
        grid.add_row("Bus voltage noise (CV)",
                     f"[{nc_color}]{noise_cv:.4f}[/]  [dim](warn >0.060)[/]")

    console.print(grid)

    if issues:
        for issue in issues:
            _print_issue_line(issue)

    console.print()


def _print_telemetry_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]TELEMETRY LINK QUALITY  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'Telemetry radio data not available.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=28)
    grid.add_column()

    # ── RSSI ─────────────────────────────────────────────────────────────────
    rssi_mean = m.get("rssi_mean")
    rssi_min  = m.get("rssi_min")
    if rssi_mean is not None:
        r_color = "red" if rssi_mean < 50 else ("yellow" if rssi_mean < 80 else "green")
        rem_str = ""
        if m.get("rem_rssi_mean") is not None:
            rem_str = f"  [dim](GCS: {m['rem_rssi_mean']:.0f})[/]"
        grid.add_row("RSSI (vehicle / min)",
                     f"[{r_color}]{rssi_mean:.0f}[/]  /  {rssi_min:.0f}  [dim](0–255)[/]{rem_str}")

    # ── SNR ──────────────────────────────────────────────────────────────────
    snr_mean = m.get("snr_mean")
    if snr_mean is not None:
        s_color = "red" if snr_mean < 5 else ("yellow" if snr_mean < 10 else "green")
        noise_str = f"  [dim](noise: {m['noise_mean']:.0f})[/]" if m.get("noise_mean") is not None else ""
        rem_snr_str = f"  [dim](GCS SNR: {m['rem_snr_mean']:.1f} dB)[/]" if m.get("rem_snr_mean") is not None else ""
        grid.add_row("SNR (signal margin)",
                     f"[{s_color}]{snr_mean:.1f} dB[/]{noise_str}{rem_snr_str}")

    # ── TX Buffer ─────────────────────────────────────────────────────────────
    txbuf_mean = m.get("txbuf_mean_pct")
    if txbuf_mean is not None:
        t_color = "red" if txbuf_mean < 20 else ("yellow" if txbuf_mean < 50 else "green")
        grid.add_row("TX buffer (space remaining)",
                     f"[{t_color}]{txbuf_mean:.0f}%[/]  [dim](min: {m.get('txbuf_min_pct', 0):.0f}%, 100=empty=best)[/]")

    # ── Packet errors ─────────────────────────────────────────────────────────
    err_rate = m.get("rxerror_rate_per_min")
    if err_rate is not None:
        e_color = "red" if err_rate > 20 else ("yellow" if err_rate > 5 else "green")
        fixed_str = f"  [dim]({m.get('fixed_total', 0)} corrected)[/]" if m.get("fixed_total", 0) > 0 else ""
        grid.add_row("Packet errors",
                     f"[{e_color}]{err_rate:.1f}/min[/]  "
                     f"[dim]({m.get('rxerrors_total', 0)} total)[/]{fixed_str}")

    # ── RSSI dropouts ─────────────────────────────────────────────────────────
    dropouts = m.get("rssi_dropout_count", 0)
    d_color = "red" if dropouts > 1 else ("yellow" if dropouts == 1 else "green")
    grid.add_row("RSSI dropouts", f"[{d_color}]{dropouts}[/]  [dim](RSSI < 30 for >= 2 s)[/]")

    console.print(grid)

    if issues:
        for issue in issues:
            _print_issue_line(issue)

    console.print()


def _print_esc_telemetry_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]ESC TELEMETRY HEALTH  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'ESC telemetry not available (BLHeli32/AM32 passthrough not configured).')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=28)
    grid.add_column()

    # ── Motor count ───────────────────────────────────────────────────────────
    motor_count = m.get("esc_count")
    if motor_count is not None:
        grid.add_row("Motors reporting", f"{motor_count}")

    # ── Temperature ───────────────────────────────────────────────────────────
    temp_max = m.get("max_esc_temp_c")
    rtemp_max = m.get("motor_winding_max_c")
    if temp_max is not None:
        t_color = "red" if temp_max >= 80 else ("yellow" if temp_max >= 60 else "green")
        grid.add_row("ESC board temp (max)", f"[{t_color}]{temp_max:.0f} °C[/]  [dim](warn >= 60, crit >= 80)[/]")
    if rtemp_max is not None:
        rt_color = "red" if rtemp_max >= 100 else ("yellow" if rtemp_max >= 70 else "green")
        grid.add_row("Motor winding temp (max)", f"[{rt_color}]{rtemp_max:.0f} °C[/]  [dim](warn >= 70, crit >= 100)[/]")

    # ── Current imbalance ────────────────────────────────────────────────────
    curr_imb = m.get("current_imbalance_pct")
    if curr_imb is not None:
        ci_color = "red" if curr_imb >= 40 else ("yellow" if curr_imb >= 20 else "green")
        grid.add_row("Current imbalance", f"[{ci_color}]{curr_imb:.1f}%[/]  [dim](warn >= 20%, crit >= 40%)[/]")

    # ── RPM imbalance ────────────────────────────────────────────────────────
    rpm_imb = m.get("rpm_imbalance_pct")
    if rpm_imb is not None:
        ri_color = "red" if rpm_imb >= 30 else ("yellow" if rpm_imb >= 15 else "green")
        grid.add_row("RPM imbalance", f"[{ri_color}]{rpm_imb:.1f}%[/]  [dim](warn >= 15%, crit >= 30%)[/]")

    # ── ESC errors ────────────────────────────────────────────────────────────
    esc_errors = m.get("esc_errors_total", 0)
    e_color = "red" if esc_errors > 0 else "green"
    grid.add_row("ESC error count", f"[{e_color}]{esc_errors}[/]  [dim](cumulative in armed window)[/]")

    console.print(grid)

    if issues:
        for issue in issues:
            _print_issue_line(issue)

    console.print()


def _print_altitude_control_result(result: dict):
    score     = result.get("score", 0)
    grade     = result.get("grade", "?")
    available = result.get("available", True)
    issues    = result.get("issues", [])
    m         = result.get("metrics", {})

    color = _grade_color(grade)
    console.rule(f"[bold]ALTITUDE CONTROL QUALITY  [{color}]{grade}[/] {score:.0f}/100[/]")

    if not available:
        console.print(f"  [dim]{m.get('unavailable_reason', 'CTUN altitude control data not available.')}[/]")
        console.print()
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=28)
    grid.add_column()

    # ── Altitude tracking ────────────────────────────────────────────────────
    alt_rms = m.get("alt_tracking_rms_m")
    if alt_rms is not None:
        warn_t = m.get("alt_warn_threshold_m", 2.0)
        crit_t = m.get("alt_crit_threshold_m", 5.0)
        a_color = "red" if alt_rms >= crit_t else ("yellow" if alt_rms >= warn_t else "green")
        grid.add_row("Altitude tracking RMS",
                     f"[{a_color}]{alt_rms:.1f} m[/]  "
                     f"[dim](warn >= {warn_t:.0f} m, crit >= {crit_t:.0f} m)[/]")

    # ── Peak deviation ───────────────────────────────────────────────────────
    peak_err = m.get("alt_peak_error_m")
    peak_ts  = m.get("alt_peak_ts")
    if peak_err is not None:
        p_color = "red" if peak_err >= m.get("alt_crit_threshold_m", 10.0) else (
                  "yellow" if peak_err >= m.get("alt_warn_threshold_m", 5.0) else "green")
        ts_str = f"  [dim]@ T+{peak_ts:.0f}s[/]" if peak_ts is not None else ""
        grid.add_row("Peak altitude error", f"[{p_color}]{peak_err:.1f} m[/]{ts_str}")

    # ── Climb rate tracking ──────────────────────────────────────────────────
    crt_rms = m.get("crt_tracking_rms_ms")
    if crt_rms is not None:
        crt_w = m.get("crt_warn_threshold", 0.5)
        crt_c = m.get("crt_crit_threshold", 1.5)
        c_color = "red" if crt_rms >= crt_c else ("yellow" if crt_rms >= crt_w else "green")
        grid.add_row("Climb-rate tracking RMS",
                     f"[{c_color}]{crt_rms:.2f} m/s[/]  "
                     f"[dim](warn >= {crt_w:.1f}, crit >= {crt_c:.1f} m/s)[/]")

    # ── Hold stability ───────────────────────────────────────────────────────
    hold_std = m.get("alt_hold_std_m")
    if hold_std is not None:
        h_color = "red" if hold_std >= 3.0 else ("yellow" if hold_std >= 1.0 else "green")
        hold_n  = m.get("hold_samples", 0)
        grid.add_row("Hold stability (std dev)",
                     f"[{h_color}]{hold_std:.2f} m[/]  [dim]({hold_n} samples in hold modes)[/]")
    elif m.get("hold_samples", 0) == 0:
        grid.add_row("Hold stability", "[dim]No hold-mode samples found[/]")

    # ── Mean altitude ────────────────────────────────────────────────────────
    alt_mean  = m.get("alt_mean_m")
    dalt_mean = m.get("dalt_mean_m")
    if alt_mean is not None and dalt_mean is not None:
        grid.add_row("Mean alt (actual / desired)",
                     f"{alt_mean:.0f} m  /  {dalt_mean:.0f} m")

    console.print(grid)

    if issues:
        for issue in issues:
            _print_issue_line(issue)

    console.print()


def _print_all_issues(all_issues: list):
    if not all_issues:
        console.print("[green]  No issues found.[/]")
        return

    console.rule("[bold]ALL ISSUES LOG[/]")
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("Sev", width=8)
    table.add_column("Code", width=9)
    table.add_column("Module", width=16)
    table.add_column("Message", no_wrap=False)

    for issue in all_issues:
        sev = issue.get("severity", "info")
        sev_text = {
            "critical": "[bold red]CRITICAL[/]",
            "warning":  "[yellow]WARNING[/]",
            "info":     "[dim]INFO[/]",
        }.get(sev, sev)
        table.add_row(
            sev_text,
            issue.get("code", ""),
            issue.get("module", ""),
            issue.get("message", ""),
        )

    console.print(table)
    console.print()


def _print_recommendations(recommendations: list):
    if not recommendations:
        return

    console.rule("[bold]MAINTENANCE RECOMMENDATIONS[/]")
    for i, rec in enumerate(recommendations, 1):
        priority = rec.get("priority", "LOW")
        color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}.get(priority, "dim")
        console.print(
            f"  [bold {color}]{i:2d}. [{priority}][/]  "
            f"[dim]{rec.get('code', '')}[/]  {rec.get('recommendation', '')}"
        )
    console.print()


def _print_verdict(scoring_result: dict):
    score = scoring_result["overall_score"]
    grade = scoring_result["grade"]
    verdict = scoring_result["verdict"]
    action = scoring_result["action"]
    color = scoring_result.get("color", "white")
    issue_sum = scoring_result["issue_summary"]

    issues_line = (
        f"[red]{issue_sum['critical']} critical[/]  "
        f"[yellow]{issue_sum['warning']} warning[/]  "
        f"[dim]{issue_sum['info']} info[/]"
    )

    console.print(Panel(
        f"[bold {color}]OVERALL FLIGHT SCORE:  {grade}  {score:.1f} / 100[/]\n"
        f"[bold]Verdict:[/] [{color}]{verdict}[/]\n"
        f"[bold]Action:[/]  {action}\n"
        f"[bold]Issues:[/]  {issues_line}",
        border_style=color,
        title="[bold]FLIGHT HEALTH REPORT[/]",
        padding=(1, 4),
    ))
    console.print()


def _print_issue_line(issue: dict):
    sev = issue.get("severity", "info")
    prefix = {
        "critical": "[bold red]  [CRITICAL][/]",
        "warning":  "[yellow]  [WARNING][/]",
        "info":     "[dim]  [INFO][/]",
    }.get(sev, "  [INFO]")
    msg = issue.get("message", "")
    ts = issue.get("timestamp_s")
    ts_str = f" [dim]@{ts:.1f}s[/]" if ts is not None else ""
    console.print(f"{prefix}{ts_str} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grade_color(grade: str) -> str:
    return {
        "A": "green",
        "B": "bright_green",
        "C": "yellow",
        "D": "orange3",
        "F": "red",
    }.get(grade, "white")


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Fleet commands
# ─────────────────────────────────────────────────────────────────────────────

_GRADE_COLOR = {"A": "green", "B": "yellow", "C": "orange3", "D": "red", "F": "bold red"}
_GRADE_ICON  = {"A": "[A+]", "B": "[B]", "C": "[C]", "D": "[D]", "F": "[F!]"}
_TREND_ARROW = {"up_fast": "^^", "up": "^", "stable": "--", "down": "v", "down_fast": "vv"}


def _fmt_score(score, grade) -> Text:
    color = _GRADE_COLOR.get(grade, "white")
    icon  = _GRADE_ICON.get(grade, "")
    t = Text()
    t.append(f"{score:.1f}", style=f"bold {color}")
    t.append(f"  {grade} {icon}", style=color)
    return t


def _fmt_trend(arrow: str) -> Text:
    # Convert Unicode arrows to ASCII for Windows terminal compatibility
    ascii_map = {"↑↑": "^^", "↑": "^ ", "→": "--", "↓": "v ", "↓↓": "vv"}
    colors    = {"↑↑": "bold red", "↑": "red", "→": "white", "↓": "green", "↓↓": "bold green"}
    disp = ascii_map.get(arrow, arrow or "--")
    return Text(disp, style=colors.get(arrow, "white"))


@cli.command()
@click.option("--drone", "-d", default=None, help="Show history for a specific drone name.")
@click.option("--all",  "show_all", is_flag=True, default=False, help="List every flight record.")
def fleet(drone: Optional[str], show_all: bool):
    """
    Show fleet health status and per-drone flight history.

    \b
    Examples:
      python cli.py fleet                      # Fleet overview (all drones)
      python cli.py fleet --drone "UAV-001"    # History for one drone
      python cli.py fleet --all                # All flight records
    """
    try:
        from fleet import db as fleet_db, trend as fleet_trend
    except ImportError as e:
        console.print(f"[red]Fleet module not available: {e}[/]")
        return

    db_path = fleet_db.get_db_path()
    if not db_path.exists():
        console.print(
            Panel(
                "[yellow]No fleet database found.\n\n"
                "Run [bold]python cli.py analyze[/] on at least one .BIN file first.\n"
                f"Database will be created at: [cyan]{db_path}[/]",
                title="Fleet DB — Empty",
                border_style="yellow",
            )
        )
        return

    # ── Single drone history ─────────────────────────────────────────────────
    if drone:
        history = fleet_db.get_drone_history(drone, limit=50)
        if not history:
            console.print(f"[yellow]No flights found for drone: {drone}[/]")
            return

        trends = fleet_trend.compute_trends(history)
        alerts = fleet_trend.fleet_alerts(history)

        console.print()
        console.print(
            Panel(
                f"[bold]{drone}[/]  |  {history[0].get('drone_type','').title()}  "
                f"|  [cyan]{len(history)} flights recorded[/]",
                title="Drone Flight History",
                border_style="blue",
            )
        )

        # Flight history table
        tbl = Table(box=box.ASCII, show_header=True, header_style="bold cyan")
        tbl.add_column("#",           style="dim",       width=4,  no_wrap=True, overflow="ignore")
        tbl.add_column("Log File",    style="white",     width=28, no_wrap=True, overflow="ignore")
        tbl.add_column("Date",        style="cyan",      width=12, no_wrap=True, overflow="ignore")
        tbl.add_column("Duration",    style="white",     width=8,  no_wrap=True, overflow="ignore")
        tbl.add_column("Score",       style="white",     width=14, no_wrap=True, overflow="ignore")
        tbl.add_column("Bat IR mohm", style="white",     width=11, no_wrap=True, overflow="ignore")
        tbl.add_column("End Cell V",  style="white",     width=10, no_wrap=True, overflow="ignore")
        tbl.add_column("Vibe X",      style="white",     width=8,  no_wrap=True, overflow="ignore")
        tbl.add_column("Issues",      style="white",     width=12, no_wrap=True, overflow="ignore")

        for i, f in enumerate(history, 1):
            grade = f.get("grade", "?")
            score = f.get("overall_score") or 0
            date  = (f.get("flight_date") or "")[:10]
            dur_s = int(f.get("duration_s") or 0)
            dur   = f"{dur_s//60}m {dur_s%60:02d}s" if dur_s else "-"
            ir    = f.get("bat_internal_resistance_mohm")
            cv    = f.get("bat_end_cell_v")
            vx    = f.get("vibe_x")
            crit  = f.get("crit_count", 0)
            warn  = f.get("warn_count", 0)

            ir_str = f"{ir:.1f}" if ir is not None else "—"
            cv_str = f"{cv:.3f}" if cv is not None else "—"
            vx_str = f"{vx:.2f}" if vx is not None else "—"

            issues_text = Text()
            if crit:  issues_text.append(f"x{crit} ", style="red")
            if warn:  issues_text.append(f"!{warn}",  style="orange3")
            if not crit and not warn: issues_text.append("[OK]", style="green")

            tbl.add_row(
                str(i),
                f.get("log_file", "—"),
                date,
                dur,
                _fmt_score(score, grade),
                ir_str,
                cv_str,
                vx_str,
                issues_text,
            )

        console.print(tbl)

        # Trend summary
        if trends:
            console.print("[bold cyan]Trend Analysis[/] (last flights, ^^=rising vv=falling --=stable):\n")
            # Convert Unicode arrows to ASCII for Windows terminal
            _arrow_ascii = {"↑↑": "^^", "↑": "^ ", "→": "--", "↓": "v ", "↓↓": "vv"}
            for col, t in trends.items():
                if t.latest is None:
                    continue
                a_ascii = _arrow_ascii.get(t.arrow, t.arrow or "--")
                unit_safe = t.unit.replace("mΩ", "mohm").replace("m/s²", "m/s2")
                console.print(
                    f"  {a_ascii:3s}  [bold]{t.label:30s}[/]  "
                    f"Latest: [cyan]{t.latest:.2f}{unit_safe}[/]  "
                    f"Slope: {t.slope:+.3f}{unit_safe}/flight"
                )

        # Alerts
        if alerts:
            console.print()
            console.print(Panel(
                "\n".join(f"  [!]  {a}" for a in alerts),
                title="[bold red]Maintenance Alerts[/]",
                border_style="red",
            ))
        else:
            console.print("\n[green][OK] No maintenance alerts - drone is trending healthy.[/]")
        return

    # ── All records flat list ────────────────────────────────────────────────
    if show_all:
        all_flights = fleet_db.get_all_flights()
        if not all_flights:
            console.print("[yellow]No flights in database.[/]")
            return
        tbl = Table(title=f"All Flights ({len(all_flights)} records)",
                    box=box.ASCII, header_style="bold cyan")
        tbl.add_column("ID",       width=4,  no_wrap=True, overflow="ignore")
        tbl.add_column("Drone",    width=20, no_wrap=True, overflow="ignore")
        tbl.add_column("Type",     width=11, no_wrap=True, overflow="ignore")
        tbl.add_column("Log File", width=22, no_wrap=True, overflow="ignore")
        tbl.add_column("Date",     width=11, no_wrap=True, overflow="ignore")
        tbl.add_column("Score",    width=13, no_wrap=True, overflow="ignore")
        for f in all_flights:
            dn = (f.get("drone_name") or "-")[:19]
            lf = (f.get("log_file")   or "-")[:21]
            tbl.add_row(
                str(f["id"]),
                dn,
                (f.get("drone_type") or "-").title()[:10],
                lf,
                (f.get("flight_date") or "")[:10],
                _fmt_score(f.get("overall_score") or 0, f.get("grade","?")),
            )
        console.print(tbl)
        return

    # ── Default: Fleet overview ──────────────────────────────────────────────
    fleet_rows = fleet_db.get_fleet_summary()
    if not fleet_rows:
        console.print("[yellow]No flights in database.[/]")
        return

    from datetime import datetime as _dt, timezone as _tz
    now_str = _dt.now(_tz.utc).strftime("%Y-%m-%d  %H:%M UTC")

    tbl = Table(
        title=f"Fleet Health Status  -  {now_str}",
        box=box.ASCII,
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Drone",         style="bold white", width=20, no_wrap=True, overflow="ignore")
    tbl.add_column("Type",          style="cyan",       width=10, no_wrap=True, overflow="ignore")
    tbl.add_column("Score",         style="white",      width=13, no_wrap=True, overflow="ignore")
    tbl.add_column("Trend",         style="white",      width=5,  no_wrap=True, overflow="ignore")
    tbl.add_column("Flights",       style="white",      width=7,  no_wrap=True, overflow="ignore")
    tbl.add_column("Last Flight",   style="white",      width=11, no_wrap=True, overflow="ignore")
    tbl.add_column("Status",        style="white",      width=22, no_wrap=True, overflow="ignore")

    for f in fleet_rows:
        drone_name = f.get("drone_name", "?")
        grade      = f.get("grade", "?")
        score      = f.get("overall_score") or 0
        dtype      = (f.get("drone_type") or "—").title()
        n_flights  = f.get("total_flights", 1)
        last_date  = (f.get("flight_date") or "")[:10]

        # Get trend arrow for this drone
        history = fleet_db.get_drone_history(drone_name, limit=10)
        arrow, alerts = fleet_trend.summary_for_cli(history)

        # Status text
        if grade in ("F", "D"):
            status = Text("[!!] Inspect / Ground", style="bold red")
        elif grade == "C":
            status = Text("[!] Inspect before flight", style="orange3")
        elif alerts:
            status = Text(f"[!] {alerts[0][:28]}", style="yellow")
        else:
            status = Text("[OK] Healthy", style="green")

        # Truncate long names to fit fixed-width columns cleanly on Windows
        dn_short  = drone_name[:21] if len(drone_name) > 21 else drone_name
        dt_short  = dtype[:11]      if len(dtype)      > 11 else dtype

        tbl.add_row(
            dn_short,
            dt_short,
            _fmt_score(score, grade),
            _fmt_trend(arrow),
            str(n_flights),
            last_date,
            status,
        )

    console.print()
    console.print(tbl)
    console.print(
        f"\n[dim]Fleet DB: {fleet_db.get_db_path()}"
        f"  |  Use [bold]--drone NAME[/] for detailed history[/]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
