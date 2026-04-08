"""
HTML Report Generator
=====================
Phase 5 — Produces a self-contained, single-file HTML report from a JSON
report dict (as returned by json_report.generate()).

Tier-1 enhancements (all six implemented):
  1. Module Health Radar Chart (SVG polygon, available modules only)
  2. Executive Summary block (auto-generated plain-English top findings)
  3. Maintenance Checklist (printable checkbox form for technicians)
  4. Sticky section navigation bar with scrollspy
  5. Expand All / Collapse All for module detail accordion
  6. Module row color highlights for D/F grade rows

Tier-2 enhancements (all implemented):
  1. Battery voltage discharge visual (SVG cell-voltage + remaining bar)
  2. Motor output comparison chart (SVG per-motor avg throttle bars)
  3. Vibration level bars (SVG X/Y/Z vs warn/crit thresholds)
  4. Maintenance verdict banner (prominent go/no-go at top of report)
  5. Issue detail expand on click (value, threshold, timestamp reveal)
  6. Module scorecard drill-down (click row → open accordion + scroll)
  7. Dark mode toggle (CSS variables, persisted in localStorage)
  8. Print / Save PDF button (already in header from Tier 1 work)
  9. Issue severity filter (already implemented in Tier 1 work)

No external dependencies — all CSS, JS, and SVG are inlined.
"""

from __future__ import annotations

import html
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    report: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Build a self-contained HTML report from a JSON report dict.

    Parameters
    ----------
    report      : dict returned by json_report.generate()
    output_path : if provided, writes HTML to this file path

    Returns
    -------
    HTML string
    """
    html_str = _render_html(report)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_str, encoding="utf-8")
        logger.info("HTML report written to: %s", path)
    return html_str


# ─────────────────────────────────────────────────────────────────────────────
# Colour / label constants
# ─────────────────────────────────────────────────────────────────────────────

_GRADE_COLOR = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}
_GRADE_BG    = {"A": "#dcfce7", "B": "#ecfccb", "C": "#fef9c3", "D": "#ffedd5", "F": "#fee2e2"}
_VERDICT_COLOR = {
    "EXCELLENT": "#22c55e", "GOOD": "#84cc16",
    "FAIR": "#eab308", "POOR": "#f97316", "CRITICAL": "#ef4444",
}
_SEV_COLOR = {"critical": "#ef4444", "warning": "#f97316", "info": "#3b82f6"}
_SEV_BG    = {"critical": "#fee2e2", "warning": "#ffedd5", "info": "#eff6ff"}
_SEV_ICON  = {"critical": "✕", "warning": "⚠", "info": "ℹ"}
_PRIO_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#3b82f6"}

_PHASE_COLORS = {
    "AUTO": "#3b82f6", "LOITER": "#8b5cf6", "STABILIZE": "#f59e0b",
    "RTL": "#f97316", "LAND": "#10b981", "GUIDED": "#06b6d4",
    "QHOVER": "#a855f7", "QLOITER": "#6366f1", "QSTABILIZE": "#f59e0b",
    "QRTL": "#f97316", "QLAND": "#10b981", "FBWA": "#3b82f6",
    "FBWB": "#0ea5e9", "CRUISE": "#22c55e", "CIRCLE": "#8b5cf6",
    "ACRO": "#ef4444", "ALTHOLD": "#f59e0b", "POSHOLD": "#14b8a6",
    "TAKEOFF": "#84cc16", "AUTOTUNE": "#ec4899",
}
_PHASE_DEFAULT = "#64748b"

_MODULE_LABELS: Dict[str, str] = {
    "flight_overview": "Flight Overview", "battery": "Battery",
    "motors": "Motors / Propulsion", "vibration": "Vibration",
    "sensors": "Sensors (GPS / EKF / IMU)", "control": "Control Performance",
    "efficiency": "Power Efficiency", "rc_link": "RC Link & Failsafes",
    "landing_wind": "Landing & Wind", "fc_health": "FC Power & Error Health",
    "vtol_transition": "VTOL Transition", "mission": "Mission Execution",
    "airspeed": "Airspeed Health", "pid_tuning": "PID Tuning Quality",
    "power_rail": "Power Rail Health", "telemetry": "Telemetry Link",
    "esc_telemetry": "ESC Telemetry", "altitude_control": "Altitude Control",
}

# Abbreviated names for the radar chart (8-10 chars max)
_MODULE_SHORT: Dict[str, str] = {
    "flight_overview": "Overview", "battery": "Battery",
    "motors": "Motors", "vibration": "Vibration",
    "sensors": "Sensors", "control": "Control",
    "efficiency": "Efficiency", "rc_link": "RC Link",
    "landing_wind": "Landing", "fc_health": "FC Health",
    "vtol_transition": "VTOL Trans", "mission": "Mission",
    "airspeed": "Airspeed", "pid_tuning": "PID",
    "power_rail": "Power Rail", "telemetry": "Telemetry",
    "esc_telemetry": "ESC", "altitude_control": "Altitude",
}

# Nav section definitions: (anchor_id, label)
_NAV_SECTIONS = [
    ("sec-summary",    "Summary"),
    ("sec-phases",     "Phases"),
    ("sec-metrics",    "Metrics"),
    ("sec-charts",     "Charts"),
    ("sec-scorecard",  "Scorecard"),
    ("sec-radar",      "Health Radar"),
    ("sec-detail",     "Module Detail"),
    ("sec-issues",     "Issues"),
    ("sec-recos",      "Recommendations"),
    ("sec-history",    "Fleet History"),
    ("sec-checklist",  "Checklist"),
]

# Keys that are implementation flags — not meaningful to display
_SKIP_KEYS = {
    "tracking_mode_filtered", "segment_excl_s", "segment_excl_insufficient",
    "is_vtol", "arm_time_s", "disarm_time_s",
}

# Row highlight style by grade
_ROW_HL = {"D": "background:#fff7ed", "F": "background:#fef2f2"}


def _gc(grade: str) -> str:
    return _GRADE_COLOR.get(grade, "#6b7280")

def _gb(grade: str) -> str:
    return _GRADE_BG.get(grade, "#f3f4f6")

def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "")


# ─────────────────────────────────────────────────────────────────────────────
# Metric value formatter
# ─────────────────────────────────────────────────────────────────────────────

def _unit_hint(key: str) -> str:
    k = key.lower()
    if "wh_per_km" in k:                                       return " Wh/km"
    if k.endswith("_wh"):                                      return " Wh"
    if k.endswith("_mah"):                                     return " mAh"
    if k.endswith("_pct") or k.endswith("_percent"):          return "%"
    if k.endswith("_v") and not k.endswith("_mv"):            return " V"
    if k.endswith("_mv"):                                      return " mV"
    if k.endswith("_a") and not k.endswith("_ma") and "count" not in k: return " A"
    if k.endswith("_ma"):                                      return " mA"
    if k.endswith("_hz"):                                      return " Hz"
    if k.endswith("_c") and "count" not in k and "rc" not in k: return " °C"
    if k.endswith("_ms") and "speed" in k:                    return " m/s"
    if k.endswith("_ms") and "count" not in k:               return " ms"
    if k.endswith("_km"):                                      return " km"
    if k.endswith("_m") and "count" not in k and "mode" not in k: return " m"
    if k.endswith("_s") and "timestamp" not in k and "count" not in k: return " s"
    if k.endswith("_deg"):                                     return "°"
    if k.endswith("_ohm"):                                     return " mΩ"
    return ""


def _fmt_scalar(key: str, val: Any) -> str:
    if val is None:
        return "<span class='na-val'>N/A</span>"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    unit = _unit_hint(key)
    if isinstance(val, float):
        if abs(val) >= 1000: return f"{val:,.0f}{unit}"
        if abs(val) >= 10:   return f"{val:.1f}{unit}"
        return f"{val:.3g}{unit}"
    if isinstance(val, int):
        return f"{val:,}{unit}"
    return _esc(str(val)) + unit


def _fmt_key(key: str) -> str:
    skip_suffixes = ("_s","_v","_a","_pct","_m","_km","_hz","_c",
                     "_mah","_wh","_ms","_deg","_ma","_mv","_ohm")
    label = key
    for suf in skip_suffixes:
        if label.endswith(suf):
            label = label[:-len(suf)]
            break
    return label.replace("_", " ").title()


def _render_list_of_dicts(items: List[Dict]) -> str:
    if not items:
        return "<span class='na-val'>None</span>"
    cols: List[str] = []
    for item in items:
        for k in item:
            if k not in cols:
                cols.append(k)
    header = "".join(f"<th>{_esc(_fmt_key(c))}</th>" for c in cols)
    rows   = "".join(
        "<tr>" + "".join(f"<td>{_fmt_scalar(c, item.get(c))}</td>" for c in cols) + "</tr>"
        for item in items
    )
    return (
        f'<div class="mini-table-wrap"><table class="mini-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _render_metric(key: str, val: Any, depth: int = 0) -> str:
    if isinstance(val, dict):
        inner = "".join(
            _render_metric(k, v, depth + 1)
            for k, v in val.items() if k not in _SKIP_KEYS
        )
        if not inner:
            return ""
        return (
            f'<div class="detail-subgroup">'
            f'<div class="detail-subgroup-label">{_esc(_fmt_key(key))}</div>'
            f'<div class="detail-subgroup-body">{inner}</div>'
            f'</div>'
        )
    if isinstance(val, list):
        if len(val) == 0:
            return (
                f'<div class="detail-item">'
                f'<span class="detail-label">{_esc(_fmt_key(key))}</span>'
                f'<span class="detail-val na-val">None</span></div>'
            )
        if isinstance(val[0], dict):
            return (
                f'<div class="detail-item detail-item-wide">'
                f'<span class="detail-label">{_esc(_fmt_key(key))}</span>'
                f'<div class="detail-val">{_render_list_of_dicts(val)}</div></div>'
            )
        display = ", ".join(_fmt_scalar("", v) for v in val[:8])
        if len(val) > 8: display += f" … +{len(val)-8} more"
        return (
            f'<div class="detail-item">'
            f'<span class="detail-label">{_esc(_fmt_key(key))}</span>'
            f'<span class="detail-val">{display}</span></div>'
        )
    return (
        f'<div class="detail-item">'
        f'<span class="detail-label">{_esc(_fmt_key(key))}</span>'
        f'<span class="detail-val">{_fmt_scalar(key, val)}</span></div>'
    )


def _render_module_metrics(metrics: Dict) -> str:
    if not metrics:
        return '<div class="na-val" style="padding:12px">No detailed metrics available.</div>'
    items_html = "".join(
        _render_metric(key, val)
        for key, val in metrics.items()
        if key not in _SKIP_KEYS
    )
    return f'<div class="detail-grid">{items_html}</div>' if items_html else \
           '<div class="na-val" style="padding:12px">No detailed metrics available.</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Main renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(report: Dict[str, Any]) -> str:
    meta    = report.get("meta", {})
    summary = report.get("flight_summary", {})
    modules = report.get("module_results", {})
    issues  = report.get("all_issues", [])
    recos   = report.get("recommendations", [])

    score   = summary.get("overall_score", 0)
    grade   = summary.get("grade", "?")
    verdict = summary.get("verdict", "")
    action  = summary.get("action", "")
    isums   = summary.get("issue_summary", {})

    log_file   = _esc(meta.get("log_file", "Unknown"))
    drone_name = _esc(meta.get("drone_name", "Unknown"))
    drone_type = _esc(meta.get("drone_type", "Unknown"))
    duration_s = meta.get("log_duration_s", 0)
    generated  = meta.get("generated_at", "")
    version    = _esc(meta.get("analyzer_version", ""))

    dur_min = int(duration_s // 60)
    dur_sec = int(duration_s % 60)
    dur_str = f"{dur_min}m {dur_sec:02d}s"

    try:
        dt = datetime.fromisoformat(generated.replace("Z", "+00:00"))
        gen_str = dt.strftime("%Y-%m-%d  %H:%M UTC")
    except Exception:
        gen_str = _esc(generated)

    crit_count  = isums.get("critical", 0)
    warn_count  = isums.get("warning", 0)
    info_count  = isums.get("info", 0)
    total_count = isums.get("total", 0)

    verdict_color = _VERDICT_COLOR.get(verdict, "#6b7280")
    score_disp    = f"{score:.1f}" if score != int(score) else f"{int(score)}"

    # Build all sections
    score_svg   = _score_gauge(score, grade)
    banner_html = _maintenance_banner(grade, verdict, score)
    exec_html   = _executive_summary(issues, summary, modules)
    phase_html  = _phase_bar(modules)
    metrics_html= _key_metrics(modules)
    charts_html = _charts_section(modules)
    mod_scores  = summary.get("module_scores", {})
    table_html  = _module_table(mod_scores, modules)
    radar_html  = _radar_chart(mod_scores)
    detail_html = _module_detail_section(modules)
    issues_html = _issues_section(issues)
    recos_html  = _recommendations_section(recos)
    check_html  = _maintenance_checklist(recos, meta)
    history_html= _fleet_history_section(report)
    nav_html    = _nav_bar()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UAV Flight Report — {log_file}</title>
<style>{_CSS}</style>
</head>
<body>

<!-- ══ HEADER ══ -->
<header class="site-header">
  <div class="header-inner">
    <div class="header-brand">
      <span class="brand-icon">✈</span>
      <div>
        <div class="brand-title">UAV Flight Log Analyzer</div>
        <div class="brand-sub">Preventive Maintenance Report  ·  v{version}</div>
      </div>
    </div>
    <div class="header-meta">
      <div class="meta-item"><span class="meta-label">Log File</span><span class="meta-val">{log_file}</span></div>
      <div class="meta-item"><span class="meta-label">Drone</span><span class="meta-val">{drone_name}</span></div>
      <div class="meta-item"><span class="meta-label">Type</span><span class="meta-val">{drone_type.upper()}</span></div>
      <div class="meta-item"><span class="meta-label">Duration</span><span class="meta-val">{dur_str}</span></div>
      <div class="meta-item"><span class="meta-label">Generated</span><span class="meta-val">{gen_str}</span></div>
      <div class="meta-item no-print" style="display:flex;gap:8px;align-items:center">
        <button class="print-btn" onclick="window.print()">🖨 Print / PDF</button>
        <button class="darkmode-btn" id="dm-toggle" onclick="toggleDark()" title="Toggle dark mode">🌙</button>
      </div>
    </div>
  </div>
</header>

<!-- ══ STICKY NAV ══ -->
{nav_html}

<!-- ══ MAINTENANCE VERDICT BANNER ══ -->
{banner_html}

<!-- ══ HERO ══ -->
<section class="hero" id="sec-summary">
  <div class="hero-inner">
    <div class="score-card">
      {score_svg}
      <div class="score-label">Flight Health Score</div>
    </div>
    <div class="verdict-block">
      <div class="verdict-badge" style="background:{verdict_color}22;border:2px solid {verdict_color};color:{verdict_color}">
        {_esc(verdict)}
      </div>
      <p class="verdict-action">{_esc(action)}</p>
      <div class="issue-pills">
        <div class="pill pill-crit"><span class="pill-count">{crit_count}</span><span class="pill-label">Critical</span></div>
        <div class="pill pill-warn"><span class="pill-count">{warn_count}</span><span class="pill-label">Warning</span></div>
        <div class="pill pill-info"><span class="pill-count">{info_count}</span><span class="pill-label">Info</span></div>
      </div>
    </div>
  </div>
</section>

<!-- ══ EXECUTIVE SUMMARY ══ -->
{exec_html}

<!-- ══ PHASE TIMELINE ══ -->
<div id="sec-phases">{phase_html}</div>

<!-- ══ KEY METRICS ══ -->
<div id="sec-metrics">{metrics_html}</div>

<!-- ══ CHARTS ══ -->
<div id="sec-charts">{charts_html}</div>

<!-- ══ MODULE SCORECARD ══ -->
<section class="section" id="sec-scorecard">
  <div class="section-inner">
    <h2 class="section-title">Module Scorecard</h2>
    {table_html}
  </div>
</section>

<!-- ══ HEALTH RADAR ══ -->
<section class="section section-alt" id="sec-radar">
  <div class="section-inner">
    <h2 class="section-title">Module Health Radar</h2>
    <p class="section-desc">Polygon shows each module's score (0–100). Outer ring = 100. Only available modules shown.</p>
    {radar_html}
  </div>
</section>

<!-- ══ DETAILED MODULE REPORTS ══ -->
<div id="sec-detail">{detail_html}</div>

<!-- ══ ISSUES ══ -->
<section class="section section-alt" id="sec-issues">
  <div class="section-inner">
    <h2 class="section-title">
      Flight Issues
      <span class="issue-count-badge">{total_count} total</span>
    </h2>
    <div class="filter-tabs no-print">
      <button class="filter-btn active"      onclick="filterIssues('all',this)">All ({total_count})</button>
      <button class="filter-btn filter-crit" onclick="filterIssues('critical',this)">&#x2715; Critical ({crit_count})</button>
      <button class="filter-btn filter-warn" onclick="filterIssues('warning',this)">&#x26A0; Warning ({warn_count})</button>
      <button class="filter-btn filter-info" onclick="filterIssues('info',this)">&#x2139; Info ({info_count})</button>
    </div>
    {issues_html}
  </div>
</section>

<!-- ══ RECOMMENDATIONS ══ -->
<section class="section" id="sec-recos">
  <div class="section-inner">
    <h2 class="section-title">Maintenance Recommendations</h2>
    {recos_html}
  </div>
</section>

<!-- ══ FLEET HISTORY ══ -->
<div id="sec-history">{history_html}</div>

<!-- ══ MAINTENANCE CHECKLIST ══ -->
<div id="sec-checklist">{check_html}</div>

<!-- ══ FOOTER ══ -->
<footer class="site-footer">
  <p>UAV Flight Log Analyzer v{version}  ·  Generated {gen_str}  ·  {log_file}</p>
</footer>

<script>{_JS}</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Score gauge SVG
# ─────────────────────────────────────────────────────────────────────────────

def _score_gauge(score: float, grade: str) -> str:
    r, cx, cy = 70, 90, 90
    stroke     = 14
    circ       = 2 * math.pi * r
    sweep      = 220
    frac       = max(0.0, min(1.0, score / 100.0))
    dash_val   = frac * (sweep / 360) * circ
    dash_gap   = circ - dash_val
    color      = _gc(grade)
    rot        = "rotate(145 90 90)"
    score_disp = f"{score:.1f}" if score != int(score) else str(int(score))
    return f"""<svg class="gauge-svg" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="{stroke}"
    stroke-dasharray="{sweep/360*circ:.1f} {circ:.1f}" stroke-linecap="round" transform="{rot}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
    stroke-dasharray="{dash_val:.1f} {dash_gap:.1f}" stroke-linecap="round" transform="{rot}"/>
  <text x="{cx}" y="{cy-10}" text-anchor="middle" font-size="28" font-weight="700" fill="{color}">{score_disp}</text>
  <text x="{cx}" y="{cy+12}" text-anchor="middle" font-size="12" fill="#6b7280">/ 100</text>
  <text x="{cx}" y="{cy+38}" text-anchor="middle" font-size="22" font-weight="800" fill="{color}">{_esc(grade)}</text>
</svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Executive Summary  (Tier 1 #2)
# ─────────────────────────────────────────────────────────────────────────────

def _executive_summary(
    issues: List[Dict],
    summary: Dict,
    modules: Dict,
) -> str:
    grade   = summary.get("grade", "?")
    verdict = summary.get("verdict", "")
    action  = summary.get("action", "")

    criticals = [i for i in issues if i.get("severity") == "critical"]
    warnings  = [i for i in issues if i.get("severity") == "warning"]

    # Top findings: up to 3 criticals, then warnings to fill up to 4
    findings: List[Dict] = []
    for iss in criticals[:3]:
        findings.append(iss)
    remaining = 4 - len(findings)
    for iss in warnings[:remaining]:
        findings.append(iss)

    # Worst module (available only)
    worst_key, worst_score = None, 101.0
    for key, res in modules.items():
        if res.get("available", True):
            s = res.get("score", 100.0) or 100.0
            if s < worst_score:
                worst_score, worst_key = s, key

    # Grade icon
    grade_icons = {"A": "✅", "B": "✔", "C": "⚠", "D": "⛔", "F": "🚨"}
    gicon = grade_icons.get(grade, "•")
    gc    = _gc(grade)
    gb    = _gb(grade)

    # Findings rows
    rows = ""
    if not findings:
        rows = f'<div class="exec-finding exec-ok">✅ No significant issues detected — drone is in excellent health.</div>'
    else:
        for iss in findings:
            sev  = iss.get("severity", "info")
            sc   = _SEV_COLOR.get(sev, "#6b7280")
            icon = _SEV_ICON.get(sev, "•")
            mod  = _esc(iss.get("module", "").replace("_", " ").title())
            msg  = _esc(iss.get("message", ""))
            code = _esc(iss.get("code", ""))
            rows += (
                f'<div class="exec-finding" style="border-left:4px solid {sc}">'
                f'<span class="exec-sev" style="color:{sc}">{icon}</span>'
                f'<div class="exec-finding-body">'
                f'<span class="exec-code" style="color:{sc}">{code}</span>'
                f'<span class="exec-mod">{mod}</span>'
                f'<p class="exec-msg">{msg}</p>'
                f'</div></div>'
            )

    # Worst module callout
    worst_html = ""
    if worst_key and worst_score < 75:
        wlabel = _esc(_MODULE_LABELS.get(worst_key, worst_key))
        wgrade = modules[worst_key].get("grade", "?")
        wc     = _gc(wgrade)
        wb     = _gb(wgrade)
        worst_html = (
            f'<div class="exec-worst">'
            f'Lowest scoring module: <strong>{wlabel}</strong> — '
            f'<span style="color:{wc};font-weight:700">{wgrade} {worst_score:.0f}/100</span>'
            f'</div>'
        )

    return f"""<section class="exec-section">
  <div class="section-inner">
    <div class="exec-card">
      <div class="exec-header" style="background:{gb};border-bottom:2px solid {gc}22">
        <span class="exec-gicon">{gicon}</span>
        <div>
          <div class="exec-title" style="color:{gc}">Executive Summary — {_esc(verdict)}</div>
          <div class="exec-action">{_esc(action)}</div>
        </div>
      </div>
      <div class="exec-body">
        <div class="exec-findings-label">Key Findings</div>
        {rows}
        {worst_html}
      </div>
    </div>
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Phase bar
# ─────────────────────────────────────────────────────────────────────────────

def _phase_bar(modules: Dict) -> str:
    ov        = modules.get("flight_overview", {}).get("metrics", {})
    phase_seq = ov.get("phase_sequence", "")
    mode_chg  = ov.get("mode_change_count", 0) or 0
    if not phase_seq or phase_seq == "—":
        return ""
    # flight_overview uses " > " as separator (may also use "→" — handle both)
    sep = " > " if " > " in phase_seq else "→"
    phases = [p.strip() for p in phase_seq.split(sep) if p.strip()]
    if not phases:
        return ""

    _BAR_THRESHOLD = 15   # above this, switch to grouped distribution view

    if len(phases) <= _BAR_THRESHOLD:
        # ── Normal sequential bar ──────────────────────────────────────────
        n  = len(phases)
        sw = round(100 / n, 2)
        segs = ""
        for i, phase in enumerate(phases):
            color = _PHASE_COLORS.get(phase.upper(), _PHASE_DEFAULT)
            left  = round(i * sw, 2)
            w     = sw if i < n - 1 else round(100 - left, 2)
            segs += (
                f'<div class="phase-seg" style="left:{left}%;width:{w}%;background:{color}" title="{_esc(phase)}">'
                f'<span class="phase-label">{_esc(phase)}</span></div>'
            )
        bar_note = ""
    else:
        # ── Grouped distribution bar (many repeated mode switches) ─────────
        # Count occurrences of each unique phase
        from collections import Counter
        counts = Counter(phases)
        total  = len(phases)
        segs   = ""
        left   = 0.0
        unique_ordered = []
        seen_set: set = set()
        for p in phases:
            if p not in seen_set:
                unique_ordered.append(p)
                seen_set.add(p)
        for i, phase in enumerate(unique_ordered):
            color = _PHASE_COLORS.get(phase.upper(), _PHASE_DEFAULT)
            cnt   = counts[phase]
            w     = round(cnt / total * 100, 2)
            label = f"{_esc(phase)} ×{cnt}"
            segs += (
                f'<div class="phase-seg" style="left:{left}%;width:{w}%;background:{color}" title="{_esc(phase)} ({cnt} times)">'
                f'<span class="phase-label">{label}</span></div>'
            )
            left = round(left + w, 2)
        bar_note = (
            f'<div class="phase-note">'
            f'⚠ {mode_chg} mode changes detected — bar shows mode distribution (proportion of occurrences). '
            f'This is a normal pattern for automated survey/mission flights.'
            f'</div>'
        )

    # Legend: unique phases with counts
    from collections import Counter
    counts  = Counter(phases)
    legend  = ""
    seen2: set = set()
    for p in phases:
        if p in seen2: continue
        seen2.add(p)
        color = _PHASE_COLORS.get(p.upper(), _PHASE_DEFAULT)
        cnt   = counts[p]
        chip  = f"{_esc(p)}" + (f" ×{cnt}" if cnt > 1 else "")
        legend += f'<span class="phase-chip" style="background:{color}22;border:1px solid {color};color:{color}">{chip}</span>'

    return f"""<section class="section section-alt">
  <div class="section-inner">
    <h2 class="section-title">Flight Phase Sequence</h2>
    <div class="phase-bar">{segs}</div>
    <div class="phase-legend">{legend}</div>
    {bar_note}
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Key metrics
# ─────────────────────────────────────────────────────────────────────────────

def _key_metrics(modules: Dict) -> str:
    ov  = modules.get("flight_overview", {}).get("metrics", {})
    bat = modules.get("battery",         {}).get("metrics", {})
    eff = modules.get("efficiency",      {}).get("metrics", {})
    lnd = modules.get("landing_wind",    {}).get("metrics", {})
    vib = modules.get("vibration",       {}).get("metrics", {})
    ctl = modules.get("control",         {}).get("metrics", {})

    airborne = ov.get("airborne_s")
    air_str  = f"{airborne/60:.1f} min" if airborne else "N/A"

    endo      = ov.get("endurance", {})
    act_min   = endo.get("actual_min")
    exp_min   = endo.get("expected_min")
    delta_pct = endo.get("delta_pct")
    endo_str  = f"{act_min:.1f} min" if act_min else "N/A"
    if exp_min:   endo_str += f" <span class='sub'>/ {exp_min:.0f} min</span>"
    if delta_pct is not None:
        sign = "+" if delta_pct >= 0 else ""
        cls  = "metric-good" if delta_pct >= -5 else "metric-warn" if delta_pct >= -20 else "metric-crit"
        endo_str += f" <span class='{cls}'>{sign}{delta_pct:.1f}%</span>"

    agl, msl = ov.get("max_altitude_agl_m"), ov.get("max_altitude_msl_m")
    alt_parts = []
    if agl is not None: alt_parts.append(f"{agl:.0f} m AGL")
    if msl is not None: alt_parts.append(f"{msl:.0f} m MSL")
    alt_str  = " / ".join(alt_parts) if alt_parts else "N/A"
    spd      = ov.get("max_speed_ms")
    dist     = ov.get("distance_km")
    mode_chg = ov.get("mode_change_count")
    spd_str  = f"{spd:.1f} m/s"        if spd      is not None else "N/A"
    dist_str = f"{dist:.3f} km"         if dist     is not None else "N/A"
    mode_str = str(mode_chg)            if mode_chg is not None else "N/A"

    v_start = bat.get("start_voltage_v")
    v_end   = bat.get("end_voltage_v")
    mah     = bat.get("capacity_consumed_mah")
    avg_cur = bat.get("avg_current_a")
    bat_rem = bat.get("capacity_remaining_pct")
    # Derive cell count from full_voltage_v ÷ 4.2 (LiPo max cell voltage)
    full_v  = bat.get("full_voltage_v")
    cell_cnt= round(full_v / 4.2) if full_v else None
    ir      = bat.get("internal_resistance_mohm")
    v_str   = f"{v_start:.2f} V → {v_end:.2f} V" if (v_start and v_end) else "N/A"
    mah_str = f"{mah:.0f} mAh"    if mah      is not None else "N/A"
    cur_str = f"{avg_cur:.1f} A"  if avg_cur  is not None else "N/A"
    cell_str= f"{cell_cnt}S"      if cell_cnt             else "N/A"
    ir_str  = f"{ir:.1f} mΩ"     if ir       is not None else "N/A"
    if bat_rem is not None:
        cls = "metric-good" if bat_rem >= 30 else "metric-warn" if bat_rem >= 15 else "metric-crit"
        rem_str = f"<span class='{cls}'>{bat_rem:.0f}%</span>"
    else:
        rem_str = "N/A"

    whkm  = eff.get("wh_per_km")
    wh    = eff.get("energy_wh")
    eff_str = f"{whkm:.1f} Wh/km" if whkm is not None else "N/A"
    wh_str  = f"{wh:.1f} Wh"      if wh   is not None else "N/A"

    sink    = lnd.get("descent_rate_ms")
    horiz   = lnd.get("horizontal_speed_ms")
    wind    = lnd.get("wind_speed_ms")
    sink_str  = f"{sink:.1f} m/s"  if sink  is not None else "N/A"
    horiz_str = f"{horiz:.1f} m/s" if horiz is not None else "N/A"
    wind_str  = f"{wind:.1f} m/s"  if wind  is not None else "N/A"

    vx, vy, vz = vib.get("mean_vibe_x"), vib.get("mean_vibe_y"), vib.get("mean_vibe_z")
    vib_str = f"X:{vx:.1f} Y:{vy:.1f} Z:{vz:.1f}" if all(v is not None for v in (vx,vy,vz)) else "N/A"
    clips   = vib.get("total_clipping_events", 0)
    clip_str= str(clips) if clips is not None else "N/A"

    rr = ctl.get("roll_rms_deg"); pr = ctl.get("pitch_rms_deg")
    ctl_str = f"Roll {rr:.2f}°  Pitch {pr:.2f}°" if (rr is not None and pr is not None) else "N/A"

    def _card(icon, label, value):
        return (
            f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
            f'<div class="metric-label">{_esc(label)}</div>'
            f'<div class="metric-value">{value}</div></div>'
        )

    cards = (
        _card("✈","Flight Time",     air_str)
      + _card("⏱","Endurance",       endo_str)
      + _card("⬆","Max Altitude",    alt_str)
      + _card("⚡","Max Speed",       spd_str)
      + _card("📍","Distance",        dist_str)
      + _card("🔄","Mode Changes",    mode_str)
      + _card("🔋","Battery Voltage", v_str)
      + _card("🔋","Pack Config",     cell_str)
      + _card("💧","Consumed",        mah_str)
      + _card("⚡","Avg Current",     cur_str)
      + _card("✅","Bat Remaining",   rem_str)
      + _card("🔩","Internal Resist", ir_str)
      + _card("🌿","Efficiency",      eff_str)
      + _card("🔌","Energy Used",     wh_str)
      + _card("🛬","Sink Rate",       sink_str)
      + _card("➡","Horiz at Land",   horiz_str)
      + _card("💨","Est Wind",        wind_str)
      + _card("📳","Vibration (m/s²)",vib_str)
      + _card("🔢","IMU Clips",       clip_str)
      + _card("🎯","ATT Tracking",    ctl_str)
    )
    return f"""<section class="section section-alt">
  <div class="section-inner">
    <h2 class="section-title">Key Flight Metrics</h2>
    <div class="metrics-grid">{cards}</div>
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Module scorecard table  (Tier 1 #6 — D/F row highlights)
# ─────────────────────────────────────────────────────────────────────────────

def _module_table(module_scores: Dict, module_results: Dict) -> str:
    rows = ""
    for mod_key, info in module_scores.items():
        label     = _MODULE_LABELS.get(mod_key, mod_key.replace("_"," ").title())
        score     = info.get("score", 0)
        grade     = info.get("grade", "?")
        available = info.get("available", True)
        summary   = _esc(info.get("summary", ""))

        if not available:
            rows += f"""
      <tr class="row-na">
        <td class="mod-name">{_esc(label)}</td>
        <td class="mod-score na-val">—</td>
        <td><span class="grade-badge na-badge">N/A</span></td>
        <td class="mod-bar-cell"><div class="bar-track"><div class="bar-fill bar-na" style="width:0%"></div></div></td>
        <td class="mod-summary na-val">Data not available in this log</td>
      </tr>"""
            continue

        gc  = _gc(grade)
        gb  = _gb(grade)
        bar = max(2, int(score))
        sd  = f"{score:.1f}" if score != int(score) else str(int(score))
        row_style = _ROW_HL.get(grade, "")
        row_style_attr = f' style="{row_style}"' if row_style else ""
        grade_icon = " 🔴" if grade == "F" else (" 🟠" if grade == "D" else "")

        rows += f"""
      <tr{row_style_attr} class="mod-row" onclick="drillDown('acc-{mod_key}')" title="Click to open {_esc(label)} details">
        <td class="mod-name">{_esc(label)}{grade_icon} <span class="drill-hint no-print">↗</span></td>
        <td class="mod-score" style="color:{gc};font-weight:700">{sd}</td>
        <td><span class="grade-badge" style="background:{gb};color:{gc};border-color:{gc}">{_esc(grade)}</span></td>
        <td class="mod-bar-cell">
          <div class="bar-track"><div class="bar-fill" style="width:{bar}%;background:{gc}"></div></div>
        </td>
        <td class="mod-summary">{summary}</td>
      </tr>"""

    return f"""<div class="table-wrap">
  <table class="mod-table">
    <thead><tr>
      <th>Module</th><th>Score</th><th>Grade</th>
      <th style="width:140px">Health Bar</th><th>Summary</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Radar chart  (Tier 1 #1)
# ─────────────────────────────────────────────────────────────────────────────

def _radar_chart(module_scores: Dict) -> str:
    # Only available modules
    avail = [(k, v) for k, v in module_scores.items() if v.get("available", True)]
    n = len(avail)
    if n < 3:
        return '<div class="na-val">Not enough modules to draw radar chart.</div>'

    cx, cy  = 280, 240
    r_max   = 170
    label_r = r_max + 32

    # Angles starting from top (-π/2), clockwise
    angles = [math.pi * (-0.5 + 2 * i / n) for i in range(n)]

    # Score polygon
    pts = []
    for i, (key, info) in enumerate(avail):
        s = max(0, min(100, info.get("score", 0) or 0))
        r = r_max * s / 100.0
        pts.append((cx + r * math.cos(angles[i]), cy + r * math.sin(angles[i])))
    polygon_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Concentric grid rings
    rings = ""
    for pct in [25, 50, 75, 100]:
        r = r_max * pct / 100.0
        rpts = " ".join(
            f"{cx + r*math.cos(a):.1f},{cy + r*math.sin(a):.1f}"
            for a in angles
        )
        stroke_w = "1.5" if pct == 100 else "0.8"
        rings += f'<polygon points="{rpts}" fill="none" stroke="#e2e8f0" stroke-width="{stroke_w}"/>'
        # Ring label (at top)
        lx = cx + r * math.cos(angles[0])
        ly = cy + r * math.sin(angles[0]) - 5
        rings += f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="9" fill="#94a3b8" text-anchor="middle">{pct}</text>'

    # Axis lines
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + r_max*math.cos(a):.1f}" y2="{cy + r_max*math.sin(a):.1f}" stroke="#e2e8f0" stroke-width="0.8"/>'
        for a in angles
    )

    # Labels
    labels = ""
    for i, (key, info) in enumerate(avail):
        a   = angles[i]
        lx  = cx + label_r * math.cos(a)
        ly  = cy + label_r * math.sin(a)
        short = _MODULE_SHORT.get(key, key)
        anchor = "middle"
        if lx < cx - 8:   anchor = "end"
        elif lx > cx + 8: anchor = "start"
        grade  = info.get("grade", "?")
        color  = _gc(grade)
        labels += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10.5" font-weight="600" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{_esc(short)}</text>'
        )

    # Score dots
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{_gc(avail[i][1].get("grade","?"))}" stroke="#fff" stroke-width="1.5"/>'
        for i, (x, y) in enumerate(pts)
    )

    # Legend: grade bands
    legend = (
        '<g font-size="10" fill="#6b7280">'
        f'<rect x="10" y="10" width="10" height="10" fill="#3b82f620" stroke="#3b82f6" stroke-width="1"/>'
        f'<text x="24" y="19">Score polygon</text>'
        '</g>'
    )

    vw = cx * 2 + 80
    vh = cy * 2 + 60

    return f"""<div class="radar-wrap">
<svg viewBox="0 0 {vw} {vh}" class="radar-svg" xmlns="http://www.w3.org/2000/svg">
  {rings}
  {axes}
  <polygon points="{polygon_pts}" fill="#3b82f618" stroke="#3b82f6" stroke-width="2" stroke-linejoin="round"/>
  {dots}
  {labels}
  {legend}
</svg>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Module detail accordion  (Tier 1 #5 — Expand All / Collapse All)
# ─────────────────────────────────────────────────────────────────────────────

def _module_detail_section(modules: Dict) -> str:
    accordions = ""
    for mod_key, res in modules.items():
        available = res.get("available", True)
        label     = _MODULE_LABELS.get(mod_key, mod_key.replace("_"," ").title())
        grade     = res.get("grade", "?")
        score     = res.get("score", 0)
        metrics   = res.get("metrics", {})
        summary   = _esc(res.get("summary", ""))

        acc_id = f"acc-{mod_key}"
        if not available:
            body = '<div class="acc-na">Data not available in this log — module excluded from scoring.</div>'
        else:
            body = _render_module_metrics(metrics)

        gc = _gc(grade) if available else "#94a3b8"
        gb = _gb(grade) if available else "#f1f5f9"
        sd = f"{score:.1f}" if score != int(score) else str(int(score))

        accordions += f"""
    <div class="accordion" id="{acc_id}">
      <button class="acc-header" onclick="toggleAcc('{acc_id}')">
        <span class="acc-label">{_esc(label)}</span>
        <span class="acc-summary">{summary}</span>
        <span class="acc-grade" style="background:{gb};color:{gc};border-color:{gc}">{_esc(grade)} {sd}/100</span>
        <span class="acc-arrow">▼</span>
      </button>
      <div class="acc-body" id="{acc_id}-body">{body}</div>
    </div>"""

    return f"""<section class="section">
  <div class="section-inner">
    <h2 class="section-title">
      Detailed Module Reports
      <span class="section-subtitle no-print">Click a module row to expand</span>
    </h2>
    <div class="expand-btns no-print">
      <button class="exp-btn" onclick="expandAll()">⊞ Expand All</button>
      <button class="exp-btn" onclick="collapseAll()">⊟ Collapse All</button>
    </div>
    <div class="accordion-group">{accordions}
    </div>
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Fleet History section
# ─────────────────────────────────────────────────────────────────────────────

def _fleet_history_section(report: Dict) -> str:
    """Render a flight history table + trend indicators from fleet DB data."""
    history = report.get("fleet_history", [])
    trends  = report.get("fleet_trends",  {})
    meta    = report.get("meta", {})

    if not history or len(history) < 2:
        return ""   # Only show once drone has 2+ flights

    drone_name = _esc(meta.get("drone_name", ""))
    current_log = meta.get("log_file", "")

    # ── Trend alert banners ───────────────────────────────────────────────────
    alerts_html = ""
    for key, t in trends.items():
        if t.get("alert") and t.get("alert_msg"):
            alerts_html += (
                f'<div class="hist-alert">'
                f'<span class="hist-alert-icon">⚠</span>'
                f'<span>{_esc(t["alert_msg"])}</span>'
                f'</div>'
            )

    # ── History table ─────────────────────────────────────────────────────────
    def _gc_hist(grade: str) -> str:
        return _GRADE_COLOR.get(grade, "#6b7280")

    def _gb_hist(grade: str) -> str:
        return _GRADE_BG.get(grade, "#f3f4f6")

    def _arrow_color(arrow: str) -> str:
        return {"↑↑": "#ef4444", "↑": "#f97316", "→": "#94a3b8",
                "↓": "#22c55e",  "↓↓": "#22c55e"}.get(arrow, "#94a3b8")

    rows = ""
    for f in history:
        log   = f.get("log_file", "—")
        date  = (f.get("flight_date") or "")[:10] or "—"
        dur_s = int(f.get("duration_s") or 0)
        dur   = f"{dur_s//60}m {dur_s%60:02d}s" if dur_s else "—"
        score = f.get("overall_score")
        grade = f.get("grade", "?")
        ir    = f.get("bat_internal_resistance_mohm")
        cv    = f.get("bat_end_cell_v")
        vx    = f.get("vibe_x")
        crit  = f.get("crit_count", 0)
        warn  = f.get("warn_count", 0)
        gc    = _gc_hist(grade)
        gb    = _gb_hist(grade)

        score_str = f"{score:.1f}" if score is not None else "—"
        ir_str    = f"{ir:.1f} mΩ"   if ir  is not None else "—"
        cv_str    = f"{cv:.3f} V"    if cv  is not None else "—"
        vx_str    = f"{vx:.2f}"      if vx  is not None else "—"

        issue_badges = ""
        if crit:
            issue_badges += f'<span class="hist-badge hist-crit">✕{crit}</span>'
        if warn:
            issue_badges += f'<span class="hist-badge hist-warn">⚠{warn}</span>'
        if not crit and not warn:
            issue_badges = '<span class="hist-badge hist-ok">✅</span>'

        is_current = "hist-row-current" if log == current_log else ""

        rows += (
            f'<tr class="{is_current}">'
            f'<td class="hist-log">{_esc(log)}'
            + (' <span class="hist-cur-tag">current</span>' if log == current_log else '')
            + f'</td>'
            f'<td class="hist-date">{date}</td>'
            f'<td class="hist-dur">{dur}</td>'
            f'<td class="hist-score">'
            f'<span style="color:{gc};font-weight:700">{score_str}</span>'
            f' <span class="hist-grade" style="background:{gb};color:{gc};border-color:{gc}">{_esc(grade)}</span>'
            f'</td>'
            f'<td class="hist-val">{ir_str}</td>'
            f'<td class="hist-val">{cv_str}</td>'
            f'<td class="hist-val">{vx_str}</td>'
            f'<td>{issue_badges}</td>'
            f'</tr>'
        )

    # ── Trend sparklines ──────────────────────────────────────────────────────
    spark_cards = ""
    trend_keys = [
        ("overall_score",                "Overall Score",    ""),
        ("bat_internal_resistance_mohm", "Battery IR",       "mΩ"),
        ("bat_end_cell_v",               "End Cell Voltage", "V"),
        ("vibe_x",                       "Vibration X",      "m/s²"),
        ("motor_imbalance_pct",          "Motor Imbalance",  "%"),
    ]
    for key, label, unit in trend_keys:
        t = trends.get(key)
        if not t:
            continue
        vals = t.get("values", [])
        if len(vals) < 2:
            continue
        arrow     = t.get("arrow", "→")
        arrow_clr = _arrow_color(arrow)
        latest    = t.get("latest")
        slope     = t.get("slope", 0)
        alert     = t.get("alert", False)
        border    = "#ef4444" if alert else "#e2e8f0"
        latest_str = f"{latest:.2f}{unit}" if latest is not None else "—"
        slope_str  = f"{slope:+.3f}{unit}/flight"
        spark_cards += (
            f'<div class="spark-card" style="border-color:{border}">'
            f'<div class="spark-label">{_esc(label)}</div>'
            f'<div class="spark-val">'
            f'<span style="color:{arrow_clr};font-weight:800;font-size:1.1rem">{arrow}</span>'
            f'  {_esc(latest_str)}'
            f'</div>'
            f'<div class="spark-slope">{_esc(slope_str)}</div>'
            f'{"<div class=\"spark-alert\">⚠ Attention needed</div>" if alert else ""}'
            f'</div>'
        )

    return f"""<section class="section section-alt" id="sec-history">
  <div class="section-inner">
    <h2 class="section-title">
      Flight History — {drone_name}
      <span class="section-subtitle">{len(history)} flights recorded</span>
    </h2>
    {f'<div class="hist-alerts">{alerts_html}</div>' if alerts_html else ''}
    {f'<div class="spark-grid">{spark_cards}</div>' if spark_cards else ''}
    <div class="table-wrap" style="margin-top:16px">
      <table class="hist-table">
        <thead><tr>
          <th>Log File</th><th>Date</th><th>Duration</th><th>Score</th>
          <th>Bat IR</th><th>End Cell V</th><th>Vibe X</th><th>Issues</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Maintenance checklist  (Tier 1 #3)
# ─────────────────────────────────────────────────────────────────────────────

def _maintenance_checklist(recos: List[Dict], meta: Dict) -> str:
    log_file   = _esc(meta.get("log_file", "Unknown"))
    drone_name = _esc(meta.get("drone_name", "Unknown"))
    generated  = meta.get("generated_at", "")
    try:
        dt = datetime.fromisoformat(generated.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d")
    except Exception:
        date_str = ""

    if not recos:
        body = '<p class="na-val" style="padding:12px">No maintenance actions required.</p>'
    else:
        rows = ""
        for idx, reco in enumerate(recos, 1):
            prio   = reco.get("priority", "LOW")
            code   = _esc(reco.get("code", ""))
            module = _esc(reco.get("module", "").replace("_", " ").title())
            text   = _esc(reco.get("recommendation", ""))
            pc     = _PRIO_COLOR.get(prio, "#6b7280")
            rows += f"""
        <tr>
          <td class="cl-idx">{idx}</td>
          <td><span class="reco-prio" style="background:{pc}22;color:{pc};border:1px solid {pc}44">{_esc(prio)}</span></td>
          <td class="cl-code"><code>{code}</code></td>
          <td class="cl-module">{module}</td>
          <td class="cl-action">{text}</td>
          <td class="cl-check"><input type="checkbox" class="cl-checkbox"></td>
          <td class="cl-notes"></td>
        </tr>"""
        body = f"""<div class="table-wrap">
  <table class="cl-table">
    <thead><tr>
      <th>#</th><th>Priority</th><th>Code</th><th>Module</th>
      <th>Action Required</th><th>Done</th><th>Notes / Initials</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""

    return f"""<section class="section section-alt">
  <div class="section-inner">
    <h2 class="section-title">
      Maintenance Checklist
      <button class="exp-btn no-print" onclick="window.print()" style="margin-left:12px">🖨 Print Checklist</button>
    </h2>
    <!-- Header fields (filled in by technician) -->
    <div class="cl-header-grid">
      <div class="cl-field">
        <span class="cl-field-label">Drone / Asset</span>
        <span class="cl-field-val">{drone_name}</span>
      </div>
      <div class="cl-field">
        <span class="cl-field-label">Log File</span>
        <span class="cl-field-val">{log_file}</span>
      </div>
      <div class="cl-field">
        <span class="cl-field-label">Report Date</span>
        <span class="cl-field-val">{date_str}</span>
      </div>
      <div class="cl-field">
        <span class="cl-field-label">Reviewed By</span>
        <span class="cl-field-val cl-blank">________________________</span>
      </div>
      <div class="cl-field">
        <span class="cl-field-label">Date Completed</span>
        <span class="cl-field-val cl-blank">________________________</span>
      </div>
      <div class="cl-field">
        <span class="cl-field-label">Signature</span>
        <span class="cl-field-val cl-blank">________________________</span>
      </div>
    </div>
    {body}
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sticky nav bar  (Tier 1 #4)
# ─────────────────────────────────────────────────────────────────────────────

def _nav_bar() -> str:
    links = "".join(
        f'<a href="#{aid}" class="nav-link" data-target="{aid}">{_esc(label)}</a>'
        for aid, label in _NAV_SECTIONS
    )
    return f"""<nav class="nav-bar no-print" id="main-nav">
  <div class="nav-inner">{links}</div>
</nav>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tier-2 #1 — Maintenance verdict banner
# ─────────────────────────────────────────────────────────────────────────────

def _maintenance_banner(grade: str, verdict: str, score: float) -> str:
    configs = {
        "F": ("⛔", "GROUND IMMEDIATELY — DO NOT FLY",
              "Critical issues detected. Aircraft must not be flown until all faults are resolved and verified.",
              "#ef4444", "#fef2f2", "#fee2e2"),
        "D": ("🔴", "DO NOT FLY — MAINTENANCE REQUIRED",
              "Significant faults detected. Aircraft must be inspected and repaired before next mission.",
              "#f97316", "#fff7ed", "#ffedd5"),
        "C": ("⚠", "INSPECT BEFORE NEXT FLIGHT",
              "Multiple issues require attention. Schedule a maintenance inspection before next mission.",
              "#eab308", "#fefce8", "#fef9c3"),
        "B": ("✔", "AIRWORTHY — MINOR ATTENTION RECOMMENDED",
              "Aircraft is safe to fly. Address warnings at the next scheduled maintenance interval.",
              "#84cc16", "#f7fee7", "#ecfccb"),
        "A": ("✅", "OK TO FLY",
              "Aircraft is in excellent health. No maintenance action required before next flight.",
              "#22c55e", "#f0fdf4", "#dcfce7"),
    }
    icon, label, desc, color, bg, border_bg = configs.get(
        grade, ("?", verdict, "", "#6b7280", "#f9fafb", "#e5e7eb"))
    score_disp = f"{score:.1f}" if score != int(score) else str(int(score))
    return f"""<div class="maint-banner no-print-hide" style="background:{bg};border-left:6px solid {color};border-bottom:2px solid {border_bg}">
  <div class="maint-inner">
    <span class="maint-icon" style="color:{color}">{icon}</span>
    <div class="maint-content">
      <div class="maint-label" style="color:{color}">{label}</div>
      <div class="maint-desc">{desc}</div>
    </div>
    <div class="maint-score-block" style="border:2px solid {color}33;background:{border_bg}">
      <div class="maint-score-val" style="color:{color}">{score_disp}</div>
      <div class="maint-score-sub">/ 100</div>
      <div class="maint-grade" style="color:{color}">{_esc(grade)}</div>
    </div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tier-2 #2–4 — Charts section (battery visual, motor bars, vibration bars)
# ─────────────────────────────────────────────────────────────────────────────

def _charts_section(modules: Dict) -> str:
    bat = modules.get("battery",   {}).get("metrics", {})
    mot = modules.get("motors",    {}).get("metrics", {})
    vib = modules.get("vibration", {}).get("metrics", {})

    bat_html = _battery_visual(bat)
    mot_html = _motor_chart(mot)
    vib_html = _vibration_chart(vib)

    if not any([bat_html, mot_html, vib_html]):
        return ""

    cards = ""
    if bat_html:
        cards += f'<div class="chart-card"><h3 class="chart-title">🔋 Battery Status</h3>{bat_html}</div>'
    if mot_html:
        cards += f'<div class="chart-card"><h3 class="chart-title">⚙ Motor Output (Avg Throttle %)</h3>{mot_html}</div>'
    if vib_html:
        cards += f'<div class="chart-card"><h3 class="chart-title">📳 Vibration Levels (m/s²)</h3>{vib_html}</div>'

    return f"""<section class="section">
  <div class="section-inner">
    <h2 class="section-title">Flight Data Charts</h2>
    <div class="charts-grid">{cards}</div>
  </div>
</section>"""


def _battery_visual(bat: Dict) -> str:
    if not bat:
        return ""
    v_start  = bat.get("start_voltage_v")
    v_end    = bat.get("end_voltage_v")
    cv_start = bat.get("start_cell_voltage_v")
    cv_end   = bat.get("end_cell_voltage_v")
    rem_pct  = bat.get("capacity_remaining_pct")
    consumed = bat.get("capacity_consumed_mah")
    total    = bat.get("capacity_total_mah")
    ir       = bat.get("internal_resistance_mohm")
    full_v   = bat.get("full_voltage_v")
    cell_cnt = round(full_v / 4.2) if full_v else None

    if v_start is None and rem_pct is None:
        return ""

    # Remaining % bar — use gray/0% when rem_pct unavailable to avoid misleading fill
    if rem_pct is not None:
        pct = max(0.0, min(100.0, rem_pct))
        bar_color = "#22c55e" if pct >= 30 else "#f97316" if pct >= 15 else "#ef4444"
    else:
        pct = 0.0
        bar_color = "#94a3b8"
    cv_end_color = "#22c55e" if cv_end and cv_end >= 3.7 else "#f97316" if cv_end and cv_end >= 3.5 else "#ef4444"

    cell_str    = f"{cell_cnt}S" if cell_cnt else "—"
    v_start_str = f"{v_start:.2f} V" if v_start is not None else "—"
    v_end_str   = f"{v_end:.2f} V"   if v_end   is not None else "—"
    cv_s_str    = f"{cv_start:.3f} V/cell" if cv_start is not None else "—"
    cv_e_str    = f"{cv_end:.3f} V/cell"   if cv_end   is not None else "—"
    rem_str     = f"{rem_pct:.1f}%" if rem_pct is not None else "—"
    cons_str    = f"{consumed:,.0f} mAh" if consumed is not None else "—"
    total_str   = f"{total:,.0f} mAh" if total is not None else "—"
    ir_str      = f"{ir:.1f} mΩ" if ir is not None else "—"

    return f"""<div class="bat-visual">
  <div class="bat-header">
    <div class="bat-stat"><span class="bat-stat-l">Pack</span><span class="bat-stat-v">{cell_str}</span></div>
    <div class="bat-stat"><span class="bat-stat-l">Start Voltage</span><span class="bat-stat-v">{v_start_str}</span></div>
    <div class="bat-stat"><span class="bat-stat-l">End Voltage</span><span class="bat-stat-v" style="color:{cv_end_color}">{v_end_str}</span></div>
    <div class="bat-stat"><span class="bat-stat-l">Cell Start</span><span class="bat-stat-v">{cv_s_str}</span></div>
    <div class="bat-stat"><span class="bat-stat-l">Cell End</span><span class="bat-stat-v" style="color:{cv_end_color}">{cv_e_str}</span></div>
    <div class="bat-stat"><span class="bat-stat-l">Int Resistance</span><span class="bat-stat-v">{ir_str}</span></div>
  </div>
  <div class="bat-bar-section">
    <div class="bat-bar-label-row">
      <span class="bat-bar-lbl">Remaining Capacity</span>
      <span class="bat-bar-pct" style="color:{bar_color}">{rem_str}</span>
    </div>
    <div class="bat-bar-track">
      <div class="bat-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
      <div class="bat-bar-markers">
        <span class="bat-marker bat-marker-crit" style="left:15%">15%</span>
        <span class="bat-marker bat-marker-warn" style="left:30%">30%</span>
      </div>
    </div>
    <div class="bat-consumed">{cons_str} used of {total_str}</div>
  </div>
</div>"""


def _motor_chart(mot: Dict) -> str:
    per_motor = mot.get("per_motor") or []
    # Fallback for single-motor fixed-wing where per_motor is None/empty:
    # synthesise a list from aggregate motor metrics + motor_channels
    if not per_motor:
        channels = mot.get("motor_channels") or []
        avg = mot.get("avg_throttle_pct")
        p95 = mot.get("p95_throttle_pct")
        if avg is not None and channels:
            per_motor = [
                {
                    "channel": ch,
                    "avg_throttle_pct": avg,
                    "max_throttle_pct": p95 if p95 is not None else avg,
                    "min_throttle_pct": None,
                    "std_throttle_pct": None,
                }
                for ch in channels
            ]
    if not per_motor:
        return ""
    hover_ref_raw = mot.get("expected_hover_throttle_pct", 50)
    hover_ref = hover_ref_raw if hover_ref_raw is not None else 50

    n       = len(per_motor)
    row_h   = 32
    pad_l   = 38   # left for channel labels
    pad_r   = 52   # right for value labels
    bar_w   = 220  # max bar width in px (SVG units)
    h_axis  = 30   # space below bars for axis labels
    h_total = n * row_h + h_axis
    w_total = pad_l + bar_w + pad_r

    # Grid lines at 25 / 50 / 75 / 100 %
    grids = ""
    for pct in [25, 50, 75, 100]:
        x = pad_l + bar_w * pct / 100
        grids += (
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{n*row_h}"'
            f' stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{n*row_h+16}" font-size="9" fill="#94a3b8"'
            f' text-anchor="middle">{pct}%</text>'
        )

    # Expected hover reference
    hx = pad_l + bar_w * hover_ref / 100
    hover_line = (
        f'<line x1="{hx:.1f}" y1="0" x2="{hx:.1f}" y2="{n*row_h}"'
        f' stroke="#f97316" stroke-width="1.5" stroke-dasharray="4,3"/>'
        f'<text x="{hx:.1f}" y="{n*row_h+28}" font-size="8" fill="#f97316"'
        f' text-anchor="middle">Hover {hover_ref:.0f}%</text>'
    )

    bars = ""
    for i, m in enumerate(per_motor):
        y   = i * row_h
        ch  = _esc(str(m.get("channel", f"M{i+1}")))
        avg = m.get("avg_throttle_pct", 0) or 0
        bar_len = bar_w * min(avg, 100) / 100
        color = "#22c55e" if avg < 70 else "#f97316" if avg < 85 else "#ef4444"
        cy_mid = y + row_h / 2 + 4
        bars += (
            f'<text x="{pad_l-5}" y="{cy_mid:.1f}" font-size="11" fill="#475569"'
            f' text-anchor="end" font-weight="600">{ch}</text>'
            f'<rect x="{pad_l}" y="{y+6}" width="{bar_len:.1f}" height="{row_h-12}"'
            f' rx="4" fill="{color}" opacity="0.82"/>'
            f'<text x="{pad_l+bar_len+5}" y="{cy_mid:.1f}" font-size="10.5"'
            f' fill="#334155" font-weight="700">{avg:.1f}%</text>'
        )

    return (
        f'<div class="chart-svg-wrap">'
        f'<svg viewBox="0 0 {w_total} {h_total}" class="chart-svg"'
        f' xmlns="http://www.w3.org/2000/svg">'
        f'{grids}{hover_line}{bars}'
        f'</svg></div>'
    )


def _vibration_chart(vib: Dict) -> str:
    vx = vib.get("mean_vibe_x")
    vy = vib.get("mean_vibe_y")
    vz = vib.get("mean_vibe_z")
    if all(v is None for v in (vx, vy, vz)):
        return ""

    WARN = 15.0
    CRIT = 30.0
    vals = [v for v in (vx, vy, vz) if v is not None]
    max_val = max(CRIT * 1.15, max(vals) * 1.15) if vals else CRIT * 1.15

    axes_data = [("X", vx), ("Y", vy), ("Z", vz)]
    n       = 3
    row_h   = 32
    pad_l   = 22
    pad_r   = 60
    bar_w   = 220
    h_axis  = 28
    h_total = n * row_h + h_axis
    w_total = pad_l + bar_w + pad_r

    # Threshold zone backgrounds
    wx = pad_l + bar_w * WARN / max_val
    cx = pad_l + bar_w * CRIT / max_val
    zones = (
        f'<rect x="{pad_l}" y="0" width="{wx-pad_l:.1f}" height="{n*row_h}" fill="#22c55e18"/>'
        f'<rect x="{wx:.1f}" y="0" width="{cx-wx:.1f}" height="{n*row_h}" fill="#f9731618"/>'
        f'<rect x="{cx:.1f}" y="0" width="{pad_l+bar_w-cx:.1f}" height="{n*row_h}" fill="#ef444418"/>'
        f'<line x1="{wx:.1f}" y1="0" x2="{wx:.1f}" y2="{n*row_h}"'
        f' stroke="#f97316" stroke-width="1.2" stroke-dasharray="3,2"/>'
        f'<line x1="{cx:.1f}" y1="0" x2="{cx:.1f}" y2="{n*row_h}"'
        f' stroke="#ef4444" stroke-width="1.2" stroke-dasharray="3,2"/>'
        f'<text x="{wx:.1f}" y="{n*row_h+16}" font-size="8.5" fill="#f97316"'
        f' text-anchor="middle">Warn {WARN:.0f}</text>'
        f'<text x="{cx:.1f}" y="{n*row_h+16}" font-size="8.5" fill="#ef4444"'
        f' text-anchor="middle">Crit {CRIT:.0f}</text>'
    )

    bars = ""
    for i, (label, val) in enumerate(axes_data):
        y = i * row_h
        if val is None:
            continue
        bar_len = bar_w * min(val, max_val) / max_val
        color   = "#22c55e" if val < WARN else "#f97316" if val < CRIT else "#ef4444"
        cy_mid  = y + row_h / 2 + 4
        bars += (
            f'<text x="{pad_l-4}" y="{cy_mid:.1f}" font-size="11" fill="#475569"'
            f' text-anchor="end" font-weight="700">{label}</text>'
            f'<rect x="{pad_l}" y="{y+6}" width="{bar_len:.1f}" height="{row_h-12}"'
            f' rx="4" fill="{color}" opacity="0.82"/>'
            f'<text x="{pad_l+bar_len+5}" y="{cy_mid:.1f}" font-size="10.5"'
            f' fill="#334155" font-weight="700">{val:.2f}</text>'
        )

    return (
        f'<div class="chart-svg-wrap">'
        f'<svg viewBox="0 0 {w_total} {h_total}" class="chart-svg"'
        f' xmlns="http://www.w3.org/2000/svg">'
        f'{zones}{bars}'
        f'</svg></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issues section
# ─────────────────────────────────────────────────────────────────────────────

def _issues_section(issues: List[Dict]) -> str:
    if not issues:
        return '<div class="empty-state">✅ No issues detected in this flight.</div>'
    cards = ""
    for idx, iss in enumerate(issues):
        sev    = iss.get("severity", "info")
        code   = _esc(iss.get("code", ""))
        module = _esc(iss.get("module", "").replace("_", " ").title())
        msg    = _esc(iss.get("message", ""))
        ts     = iss.get("timestamp_s")
        value  = iss.get("value")
        thresh = iss.get("threshold")
        sc, sb = _SEV_COLOR.get(sev, "#6b7280"), _SEV_BG.get(sev, "#f9fafb")
        icon   = _SEV_ICON.get(sev, "•")
        card_id = f"iss-{idx}"

        # Build expandable detail rows
        detail_rows = ""
        if ts is not None:
            detail_rows += f'<div class="iss-detail-row"><span class="iss-dl">Timestamp</span><span class="iss-dv">T+{ts:.1f} s into flight</span></div>'
        if value is not None:
            detail_rows += f'<div class="iss-detail-row"><span class="iss-dl">Measured Value</span><span class="iss-dv">{_esc(str(value))}</span></div>'
        if thresh is not None:
            detail_rows += f'<div class="iss-detail-row"><span class="iss-dl">Threshold</span><span class="iss-dv">{_esc(str(thresh))}</span></div>'
        detail_rows += f'<div class="iss-detail-row"><span class="iss-dl">Severity</span><span class="iss-dv" style="color:{sc};font-weight:700">{sev.upper()}</span></div>'
        detail_rows += f'<div class="iss-detail-row"><span class="iss-dl">Module</span><span class="iss-dv">{module}</span></div>'
        detail_rows += f'<div class="iss-detail-row"><span class="iss-dl">Code</span><span class="iss-dv"><code>{code}</code></span></div>'

        expand_hint = '<span class="iss-expand-hint no-print">▸ click for details</span>'

        cards += (
            f'<div class="issue-card" id="{card_id}" data-sev="{sev}"'
            f' style="border-left:4px solid {sc};background:{sb}0a"'
            f' onclick="toggleIssue(\'{card_id}\')" role="button" tabindex="0">'
            f'<div class="issue-top">'
            f'<span class="sev-dot" style="background:{sc};color:#fff">{icon}</span>'
            f'<span class="issue-code" style="background:{sb};color:{sc};border:1px solid {sc}33">{code}</span>'
            f'<span class="issue-module">{module}</span>'
            f'{expand_hint}</div>'
            f'<p class="issue-msg">{msg}</p>'
            f'<div class="iss-detail" id="{card_id}-detail">'
            f'{detail_rows}'
            f'</div>'
            f'</div>'
        )
    return f'<div class="issue-list" id="issue-list">{cards}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────────────────────

def _recommendations_section(recos: List[Dict]) -> str:
    if not recos:
        return '<div class="empty-state">No maintenance actions required.</div>'
    items = ""
    for reco in recos:
        prio   = reco.get("priority", "LOW")
        code   = _esc(reco.get("code", ""))
        module = _esc(reco.get("module", "").replace("_", " ").title())
        text   = _esc(reco.get("recommendation", ""))
        pc     = _PRIO_COLOR.get(prio, "#6b7280")
        items += (
            f'<div class="reco-card"><div class="reco-left">'
            f'<span class="reco-prio" style="background:{pc}22;color:{pc};border:1px solid {pc}44">{_esc(prio)}</span>'
            f'</div><div class="reco-body"><div class="reco-top">'
            f'<span class="reco-code">{code}</span><span class="reco-module">{module}</span>'
            f'</div><p class="reco-text">{text}</p></div></div>'
        )
    return f'<div class="reco-list">{items}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Embedded CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px;scroll-behavior:smooth}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  background:#f1f5f9;color:#1e293b;line-height:1.55}

/* ── Header ── */
.site-header{background:#0f172a;color:#fff;padding:0 24px}
.header-inner{max-width:1200px;margin:0 auto;display:flex;align-items:center;
  justify-content:space-between;flex-wrap:wrap;gap:12px;padding:16px 0}
.header-brand{display:flex;align-items:center;gap:14px}
.brand-icon{font-size:2rem}
.brand-title{font-size:1.2rem;font-weight:700;letter-spacing:.5px}
.brand-sub{font-size:.78rem;color:#94a3b8;margin-top:2px}
.header-meta{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center}
.meta-item{display:flex;flex-direction:column}
.meta-label{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
.meta-val{font-size:.82rem;font-weight:600;color:#e2e8f0}
.print-btn{padding:6px 14px;background:#1d4ed8;color:#fff;border:none;border-radius:6px;
  font-size:.82rem;font-weight:600;cursor:pointer}
.print-btn:hover{background:#2563eb}

/* ── Sticky nav (Tier1 #4) ── */
.nav-bar{position:sticky;top:0;z-index:200;background:#1e293b;
  border-bottom:2px solid #334155;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.nav-inner{max-width:1200px;margin:0 auto;display:flex;flex-wrap:nowrap;
  overflow-x:auto;gap:0;padding:0 8px}
.nav-link{display:block;padding:10px 14px;color:#94a3b8;text-decoration:none;
  font-size:.8rem;font-weight:600;white-space:nowrap;border-bottom:3px solid transparent;
  transition:all .15s}
.nav-link:hover{color:#e2e8f0;background:#334155}
.nav-link.active{color:#38bdf8;border-bottom-color:#38bdf8}

/* ── Hero ── */
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);padding:40px 24px}
.hero-inner{max-width:1200px;margin:0 auto;display:flex;align-items:center;gap:56px;flex-wrap:wrap}
.score-card{display:flex;flex-direction:column;align-items:center;gap:8px}
.gauge-svg{width:180px;height:180px;filter:drop-shadow(0 4px 16px rgba(0,0,0,.35))}
.score-label{color:#94a3b8;font-size:.82rem;font-weight:500}
.verdict-block{flex:1;min-width:240px}
.verdict-badge{display:inline-block;padding:6px 22px;border-radius:100px;
  font-size:1.15rem;font-weight:800;letter-spacing:1.5px;margin-bottom:14px}
.verdict-action{color:#cbd5e1;font-size:.92rem;margin-bottom:22px;max-width:500px}
.issue-pills{display:flex;gap:12px;flex-wrap:wrap}
.pill{display:flex;flex-direction:column;align-items:center;padding:8px 20px;border-radius:12px;min-width:72px}
.pill-crit{background:#fee2e222;border:2px solid #ef444455}
.pill-warn{background:#ffedd522;border:2px solid #f9731655}
.pill-info{background:#eff6ff22;border:2px solid #3b82f655}
.pill-count{font-size:1.6rem;font-weight:800}
.pill-label{font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px}
.pill-crit .pill-count{color:#ef4444}
.pill-warn .pill-count{color:#f97316}
.pill-info .pill-count{color:#3b82f6}

/* ── Executive Summary (Tier1 #2) ── */
.exec-section{background:#fff;padding:28px 24px;border-bottom:1px solid #e2e8f0}
.exec-card{border:1px solid #e2e8f0;border-radius:12px;overflow:hidden}
.exec-header{display:flex;align-items:flex-start;gap:14px;padding:16px 20px}
.exec-gicon{font-size:1.8rem;flex-shrink:0}
.exec-title{font-size:1.05rem;font-weight:800;margin-bottom:4px}
.exec-action{font-size:.88rem;color:#475569}
.exec-body{padding:16px 20px;background:#fafafa}
.exec-findings-label{font-size:.72rem;font-weight:700;color:#64748b;
  text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.exec-finding{display:flex;gap:12px;align-items:flex-start;padding:10px 14px;
  margin-bottom:8px;border-radius:8px;background:#fff;border:1px solid #e2e8f0}
.exec-finding.exec-ok{border-left:4px solid #22c55e;color:#166534;font-weight:600}
.exec-sev{font-size:1rem;flex-shrink:0;margin-top:2px}
.exec-finding-body{flex:1}
.exec-code{font-family:monospace;font-size:.78rem;font-weight:700;
  padding:1px 6px;background:#f1f5f9;border-radius:4px;margin-right:6px}
.exec-mod{font-size:.78rem;color:#64748b;font-weight:600;margin-right:6px}
.exec-msg{font-size:.87rem;color:#334155;margin-top:4px;line-height:1.45}
.exec-worst{margin-top:10px;font-size:.87rem;color:#475569;
  padding:8px 12px;background:#fff;border:1px solid #e2e8f0;border-radius:8px}

/* ── Phase bar ── */
.phase-bar{position:relative;height:36px;background:#e2e8f0;border-radius:8px;
  overflow:hidden;margin-bottom:12px}
.phase-seg{position:absolute;top:0;height:100%;display:flex;align-items:center;
  justify-content:center;overflow:hidden}
.phase-seg:hover{opacity:.85}
.phase-label{font-size:.72rem;font-weight:700;color:#fff;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;padding:0 6px;text-shadow:0 1px 2px rgba(0,0,0,.5)}
.phase-legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.phase-chip{padding:3px 10px;border-radius:100px;font-size:.75rem;font-weight:600}
.phase-note{margin-top:10px;font-size:.8rem;color:#64748b;padding:8px 12px;
  background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px}

/* ── Sections ── */
.section{padding:36px 24px}
.section-alt{background:#fff}
.section-inner{max-width:1200px;margin:0 auto}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #e2e8f0;
  display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.section-subtitle{font-size:.78rem;color:#94a3b8;font-weight:400}
.section-desc{font-size:.85rem;color:#64748b;margin-bottom:16px}
.issue-count-badge{font-size:.75rem;padding:2px 8px;background:#f1f5f9;
  color:#64748b;border-radius:100px;font-weight:600;border:1px solid #e2e8f0}

/* ── Metrics grid ── */
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px}
.metric-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px}
.metric-icon{font-size:1.1rem;margin-bottom:5px}
.metric-label{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:3px}
.metric-value{font-size:.9rem;font-weight:600;color:#1e293b}
.metric-good{color:#22c55e;font-weight:700}
.metric-warn{color:#f97316;font-weight:700}
.metric-crit{color:#ef4444;font-weight:700}
.sub{color:#94a3b8;font-weight:400;font-size:.85em}
.na-val{color:#94a3b8;font-style:italic}

/* ── Module table ── */
.table-wrap{overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0}
.mod-table{width:100%;border-collapse:collapse;font-size:.88rem}
.mod-table thead th{text-align:left;padding:10px 12px;background:#f8fafc;
  border-bottom:2px solid #e2e8f0;color:#64748b;font-weight:600;font-size:.75rem;
  text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
.mod-table tbody tr{border-bottom:1px solid #f1f5f9;transition:background .1s}
.mod-table tbody tr:hover{filter:brightness(.97)}
.mod-table td{padding:9px 12px;vertical-align:middle}
.mod-name{font-weight:600;color:#1e293b;white-space:nowrap}
.mod-score{font-size:1rem;white-space:nowrap}
.grade-badge{display:inline-block;padding:2px 10px;border-radius:6px;
  font-weight:700;font-size:.85rem;border:1px solid;white-space:nowrap}
.na-badge{background:#f1f5f9;color:#94a3b8;border-color:#e2e8f0}
.row-na{opacity:.65}
.mod-bar-cell{width:140px;min-width:90px}
.bar-track{background:#e2e8f0;border-radius:100px;height:8px;width:100%}
.bar-fill{height:8px;border-radius:100px;transition:width .5s ease}
.bar-na{background:#e2e8f0}
.mod-summary{color:#475569;font-size:.83rem}

/* ── Radar chart (Tier1 #1) ── */
.radar-wrap{display:flex;justify-content:center;padding:8px 0}
.radar-svg{width:100%;max-width:620px;height:auto}

/* ── Expand/collapse buttons (Tier1 #5) ── */
.expand-btns{display:flex;gap:8px;margin-bottom:12px}
.exp-btn{padding:5px 14px;border:1px solid #e2e8f0;background:#f8fafc;
  border-radius:6px;font-size:.8rem;font-weight:600;cursor:pointer;color:#475569}
.exp-btn:hover{background:#f1f5f9;border-color:#cbd5e1}

/* ── Accordion ── */
.accordion-group{display:flex;flex-direction:column;gap:6px}
.accordion{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;background:#fff}
.acc-header{width:100%;background:#f8fafc;border:none;cursor:pointer;
  padding:12px 16px;display:flex;align-items:center;gap:12px;text-align:left;transition:background .15s}
.acc-header:hover{background:#f1f5f9}
.acc-label{font-weight:700;color:#1e293b;font-size:.92rem;flex-shrink:0;min-width:190px}
.acc-summary{color:#64748b;font-size:.82rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.acc-grade{padding:3px 10px;border-radius:6px;font-size:.8rem;font-weight:700;
  border:1px solid;white-space:nowrap;flex-shrink:0}
.acc-arrow{color:#64748b;font-size:.75rem;flex-shrink:0;transition:transform .2s}
.acc-header.open .acc-arrow{transform:rotate(180deg)}
.acc-body{display:none;padding:16px;border-top:1px solid #f1f5f9}
.acc-body.open{display:block}
.acc-na{color:#94a3b8;font-style:italic;padding:4px}

/* ── Detail grid ── */
.detail-grid{display:flex;flex-wrap:wrap;gap:0}
.detail-item{display:flex;flex-direction:column;gap:3px;padding:8px 12px;
  min-width:175px;border-right:1px solid #f1f5f9;border-bottom:1px solid #f1f5f9;flex:0 0 auto}
.detail-item-wide{flex:1 1 100%;flex-direction:column}
.detail-label{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px}
.detail-val{font-size:.88rem;font-weight:600;color:#1e293b}
.detail-subgroup{width:100%;padding:10px 12px;border-bottom:1px solid #f1f5f9}
.detail-subgroup-label{font-size:.75rem;font-weight:700;color:#475569;
  text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.detail-subgroup-body{display:flex;flex-wrap:wrap;gap:0}
.mini-table-wrap{overflow-x:auto;margin-top:4px}
.mini-table{border-collapse:collapse;font-size:.8rem;min-width:200px}
.mini-table thead th{background:#f1f5f9;padding:4px 8px;text-align:left;
  font-weight:600;color:#475569;border-bottom:1px solid #e2e8f0;white-space:nowrap}
.mini-table tbody td{padding:4px 8px;border-bottom:1px solid #f8fafc;color:#334155}
.mini-table tbody tr:last-child td{border-bottom:none}

/* ── Issue filter ── */
.filter-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.filter-btn{padding:6px 16px;border-radius:100px;border:1px solid #e2e8f0;
  background:#f8fafc;color:#475569;font-size:.82rem;font-weight:600;cursor:pointer;transition:all .15s}
.filter-btn:hover{background:#f1f5f9}
.filter-btn.active{background:#1e293b;color:#fff;border-color:#1e293b}
.filter-crit.active{background:#ef4444;border-color:#ef4444}
.filter-warn.active{background:#f97316;border-color:#f97316}
.filter-info.active{background:#3b82f6;border-color:#3b82f6}

/* ── Issue cards ── */
.issue-list{display:flex;flex-direction:column;gap:8px}
.issue-card{padding:12px 16px;border-radius:8px;border:1px solid #e2e8f0}
.issue-card.hidden{display:none}
.issue-top{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px}
.sev-dot{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.75rem;font-weight:800;flex-shrink:0}
.issue-code{display:inline-block;padding:2px 8px;border-radius:5px;
  font-size:.75rem;font-weight:700;font-family:monospace}
.issue-module{font-size:.78rem;color:#64748b;font-weight:600}
.iss-tag{display:inline-block;background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:5px;font-size:.75rem}
.issue-msg{font-size:.87rem;color:#334155;line-height:1.5}
.empty-state{padding:32px;text-align:center;color:#64748b;
  border:2px dashed #e2e8f0;border-radius:12px}

/* ── Recommendations ── */
.reco-list{display:flex;flex-direction:column;gap:10px}
.reco-card{display:flex;gap:14px;background:#f8fafc;
  border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
.reco-left{flex-shrink:0}
.reco-prio{display:inline-block;padding:4px 10px;border-radius:6px;
  font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
.reco-body{flex:1}
.reco-top{display:flex;align-items:center;gap:10px;margin-bottom:5px}
.reco-code{font-family:monospace;font-size:.78rem;font-weight:700;
  color:#1e293b;background:#e2e8f0;padding:2px 7px;border-radius:4px}
.reco-module{font-size:.78rem;color:#64748b;font-weight:600}
.reco-text{font-size:.87rem;color:#334155;line-height:1.5}

/* ── Maintenance checklist (Tier1 #3) ── */
.cl-header-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:10px;margin-bottom:16px;padding:16px;background:#f8fafc;
  border:1px solid #e2e8f0;border-radius:8px}
.cl-field{display:flex;flex-direction:column;gap:3px}
.cl-field-label{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px}
.cl-field-val{font-size:.88rem;font-weight:600;color:#1e293b;border-bottom:1px solid #cbd5e1;padding-bottom:3px}
.cl-blank{color:#94a3b8;font-weight:400;font-style:italic}
.cl-table{width:100%;border-collapse:collapse;font-size:.85rem}
.cl-table thead th{text-align:left;padding:9px 10px;background:#f8fafc;
  border-bottom:2px solid #e2e8f0;color:#64748b;font-weight:600;font-size:.75rem;
  text-transform:uppercase;letter-spacing:.5px}
.cl-table tbody tr{border-bottom:1px solid #f1f5f9}
.cl-table tbody tr:hover{background:#fafafa}
.cl-table td{padding:9px 10px;vertical-align:top}
.cl-idx{color:#94a3b8;font-weight:600;width:28px}
.cl-code code{font-family:monospace;font-size:.78rem;background:#f1f5f9;
  padding:2px 6px;border-radius:4px}
.cl-module{color:#475569;font-size:.82rem;white-space:nowrap}
.cl-action{color:#334155;line-height:1.45;max-width:400px}
.cl-check{text-align:center;width:50px}
.cl-checkbox{width:18px;height:18px;cursor:pointer;accent-color:#3b82f6}
.cl-notes{min-width:120px;border-bottom:1px dashed #94a3b8}

/* ── Footer ── */
.site-footer{background:#0f172a;color:#475569;text-align:center;
  padding:16px 24px;font-size:.78rem}

/* ── Print ── */
@media print{
  .no-print{display:none!important}
  body{background:#fff;font-size:13px}
  .site-header,.hero{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .section-alt{background:#fff}
  .acc-body{display:block!important}
  .acc-arrow,.expand-btns{display:none}
  .phase-bar,.phase-seg,.bar-fill,.grade-badge,.sev-dot,.exec-header
    {-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .issue-card,reco-card,.accordion{page-break-inside:avoid;border:1px solid #ccc}
  .cl-table{page-break-inside:auto}
  .cl-table tr{page-break-inside:avoid}
  .cl-checkbox{display:inline-block!important}
  .cl-notes{min-height:20px}
  .radar-svg{max-width:460px}
  .maint-banner{-webkit-print-color-adjust:exact;print-color-adjust:exact;border-left-width:6px!important}
  .iss-detail{display:block!important}
  .iss-expand-hint,.drill-hint{display:none}
  .chart-svg-wrap,.bat-visual,.charts-grid{page-break-inside:avoid}
}

/* ══════════════════════════════════════════════════════
   TIER-2 STYLES
   ══════════════════════════════════════════════════════ */

/* ── Dark mode toggle button ── */
.darkmode-btn{padding:5px 10px;background:#334155;color:#e2e8f0;border:1px solid #475569;
  border-radius:6px;font-size:.85rem;cursor:pointer;transition:background .15s;line-height:1}
.darkmode-btn:hover{background:#475569}

/* ── Maintenance verdict banner ── */
.maint-banner{padding:0;border-bottom:2px solid transparent}
.maint-inner{max-width:1200px;margin:0 auto;display:flex;align-items:center;
  gap:18px;padding:14px 24px;flex-wrap:wrap}
.maint-icon{font-size:2rem;flex-shrink:0}
.maint-content{flex:1;min-width:200px}
.maint-label{font-size:.98rem;font-weight:800;letter-spacing:.4px;margin-bottom:3px}
.maint-desc{font-size:.84rem;color:#475569;line-height:1.4}
.maint-score-block{flex-shrink:0;text-align:center;padding:8px 18px;border-radius:10px;min-width:80px}
.maint-score-val{font-size:1.6rem;font-weight:800;line-height:1}
.maint-score-sub{font-size:.72rem;color:#64748b}
.maint-grade{font-size:1.3rem;font-weight:800;margin-top:2px}

/* ── Charts grid ── */
.charts-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.chart-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px}
.chart-title{font-size:.88rem;font-weight:700;color:#1e293b;margin-bottom:12px;
  padding-bottom:8px;border-bottom:1px solid #e2e8f0}
.chart-svg-wrap{width:100%;overflow-x:auto}
.chart-svg{width:100%;height:auto;display:block}

/* ── Battery visual ── */
.bat-visual{display:flex;flex-direction:column;gap:12px}
.bat-header{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px}
.bat-stat{display:flex;flex-direction:column;gap:2px;padding:6px 10px;
  background:#fff;border:1px solid #e2e8f0;border-radius:7px}
.bat-stat-l{font-size:.66rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px}
.bat-stat-v{font-size:.88rem;font-weight:700;color:#1e293b}
.bat-bar-section{display:flex;flex-direction:column;gap:5px}
.bat-bar-label-row{display:flex;justify-content:space-between;align-items:center}
.bat-bar-lbl{font-size:.75rem;color:#475569;font-weight:600}
.bat-bar-pct{font-size:.95rem;font-weight:800}
.bat-bar-track{position:relative;height:20px;background:#e2e8f0;border-radius:10px;overflow:visible}
.bat-bar-fill{height:100%;border-radius:10px;transition:width .6s ease}
.bat-bar-markers{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.bat-marker{position:absolute;top:-18px;font-size:.65rem;color:#94a3b8;
  transform:translateX(-50%);white-space:nowrap}
.bat-marker::after{content:'';position:absolute;top:18px;left:50%;width:1px;
  height:20px;background:#cbd5e1;transform:translateX(-50%)}
.bat-consumed{font-size:.77rem;color:#64748b;margin-top:2px}

/* ── Issue expandable detail ── */
.issue-card{cursor:pointer;transition:box-shadow .15s}
.issue-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.08)}
.iss-expand-hint{font-size:.72rem;color:#94a3b8;margin-left:auto;flex-shrink:0;font-style:italic}
.iss-detail{display:none;margin-top:10px;padding-top:10px;
  border-top:1px solid #e2e8f0}
.iss-detail.open{display:grid!important;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px}
.iss-detail-row{display:flex;flex-direction:column;gap:2px;padding:6px 10px;
  background:#fff;border:1px solid #e2e8f0;border-radius:6px}
.iss-dl{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px}
.iss-dv{font-size:.84rem;font-weight:600;color:#1e293b}
.iss-dv code{font-family:monospace;font-size:.8rem;background:#f1f5f9;
  padding:1px 5px;border-radius:3px}

/* ── Scorecard drill-down ── */
.mod-row{cursor:pointer}
.mod-row:hover td{background:#f8fafc!important}
.drill-hint{color:#94a3b8;font-size:.75rem;font-weight:400;margin-left:4px}

/* ══════════════════════════════════════════════════════
   DARK MODE
   ══════════════════════════════════════════════════════ */
body.dark{background:#0f172a;color:#e2e8f0}
body.dark .section-alt{background:#1e293b}
body.dark .section{background:#0f172a}
body.dark .exec-section{background:#1e293b}
body.dark .exec-card{border-color:#334155}
body.dark .exec-header{border-bottom-color:#334155}
body.dark .exec-finding{background:#162032;border-color:#334155}
body.dark .exec-worst{background:#162032;border-color:#334155;color:#94a3b8}
body.dark .exec-action{color:#94a3b8}
body.dark .exec-msg{color:#cbd5e1}
body.dark .section-title{color:#e2e8f0;border-bottom-color:#334155}
body.dark .section-desc{color:#94a3b8}
body.dark .metric-card{background:#1e293b;border-color:#334155}
body.dark .metric-label{color:#94a3b8}
body.dark .metric-value{color:#e2e8f0}
body.dark .table-wrap{border-color:#334155}
body.dark .mod-table thead th{background:#162032;color:#94a3b8;border-bottom-color:#334155}
body.dark .mod-table tbody tr{border-bottom-color:#1e293b}
body.dark .mod-table tbody tr:hover td{background:#162032!important}
body.dark .mod-row:hover td{background:#162032!important}
body.dark .mod-table td{color:#e2e8f0}
body.dark .mod-name{color:#e2e8f0}
body.dark .mod-summary{color:#94a3b8}
body.dark .na-val{color:#475569}
body.dark .accordion{background:#1e293b;border-color:#334155}
body.dark .acc-header{background:#162032}
body.dark .acc-header:hover{background:#1e293b}
body.dark .acc-label{color:#e2e8f0}
body.dark .acc-summary{color:#64748b}
body.dark .acc-body{border-top-color:#334155}
body.dark .detail-item{border-right-color:#334155;border-bottom-color:#334155}
body.dark .detail-label{color:#94a3b8}
body.dark .detail-val{color:#e2e8f0}
body.dark .detail-subgroup{border-bottom-color:#334155}
body.dark .detail-subgroup-label{color:#94a3b8}
body.dark .mini-table thead th{background:#162032;color:#94a3b8;border-bottom-color:#334155}
body.dark .mini-table tbody td{color:#cbd5e1;border-bottom-color:#1e293b}
body.dark .issue-card{background:#1e293b!important;border-color:#334155}
body.dark .issue-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.4)}
body.dark .iss-detail{border-top-color:#334155}
body.dark .iss-detail-row{background:#162032;border-color:#334155}
body.dark .iss-dl{color:#64748b}
body.dark .iss-dv{color:#e2e8f0}
body.dark .iss-dv code{background:#0f172a}
body.dark .issue-module{color:#94a3b8}
body.dark .issue-msg{color:#cbd5e1}
body.dark .iss-tag{background:#162032;color:#94a3b8}
body.dark .reco-card{background:#1e293b;border-color:#334155}
body.dark .reco-code{background:#334155;color:#e2e8f0}
body.dark .reco-module{color:#94a3b8}
body.dark .reco-text{color:#cbd5e1}
body.dark .filter-btn{background:#1e293b;border-color:#334155;color:#94a3b8}
body.dark .filter-btn:hover{background:#162032}
body.dark .filter-btn.active{background:#1e293b;color:#fff;border-color:#1e293b}
body.dark .issue-count-badge{background:#1e293b;color:#94a3b8;border-color:#334155}
body.dark .empty-state{border-color:#334155;color:#64748b}
body.dark .cl-header-grid{background:#1e293b;border-color:#334155}
body.dark .cl-field-label{color:#64748b}
body.dark .cl-field-val{color:#e2e8f0;border-bottom-color:#475569}
body.dark .cl-table thead th{background:#162032;color:#94a3b8;border-bottom-color:#334155}
body.dark .cl-table tbody tr{border-bottom-color:#1e293b}
body.dark .cl-table tbody tr:hover{background:#162032}
body.dark .cl-action{color:#cbd5e1}
body.dark .cl-module{color:#94a3b8}
body.dark .cl-notes{border-bottom-color:#475569}
body.dark .chart-card{background:#1e293b;border-color:#334155}
body.dark .chart-title{color:#e2e8f0;border-bottom-color:#334155}
body.dark .bat-stat{background:#162032;border-color:#334155}
body.dark .bat-stat-l{color:#64748b}
body.dark .bat-stat-v{color:#e2e8f0}
body.dark .bat-bar-track{background:#334155}
body.dark .bat-marker{color:#475569}
body.dark .bat-consumed{color:#64748b}
body.dark .phase-note{background:#1e293b;border-color:#334155;color:#94a3b8}
body.dark .expand-btns .exp-btn{background:#1e293b;border-color:#334155;color:#94a3b8}
body.dark .expand-btns .exp-btn:hover{background:#162032}
body.dark .maint-desc{color:#94a3b8}

/* ── Fleet History (Phase 6) ── */
.hist-alerts{display:flex;flex-direction:column;gap:8px;margin-bottom:16px}
.hist-alert{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;
  background:#fff7ed;border:1px solid #f9731644;border-left:4px solid #f97316;
  border-radius:8px;font-size:.87rem;color:#c2410c}
.hist-alert-icon{font-size:1rem;flex-shrink:0;margin-top:1px}
.spark-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
  gap:10px;margin-bottom:16px}
.spark-card{padding:12px 14px;background:#f8fafc;border:1px solid #e2e8f0;
  border-radius:10px;transition:border-color .2s}
.spark-label{font-size:.68rem;color:#64748b;text-transform:uppercase;
  letter-spacing:.4px;margin-bottom:6px}
.spark-val{font-size:.92rem;font-weight:600;color:#1e293b;margin-bottom:3px}
.spark-slope{font-size:.72rem;color:#94a3b8}
.spark-alert{font-size:.72rem;color:#f97316;font-weight:600;margin-top:4px}
.hist-table{width:100%;border-collapse:collapse;font-size:.85rem}
.hist-table thead th{text-align:left;padding:9px 12px;background:#f8fafc;
  border-bottom:2px solid #e2e8f0;color:#64748b;font-weight:600;font-size:.74rem;
  text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
.hist-table tbody tr{border-bottom:1px solid #f1f5f9;transition:background .1s}
.hist-table tbody tr:hover{background:#f8fafc}
.hist-row-current{background:#eff6ff!important}
.hist-row-current:hover{background:#dbeafe!important}
.hist-log{font-size:.8rem;color:#475569;max-width:220px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-cur-tag{display:inline-block;background:#3b82f6;color:#fff;
  font-size:.62rem;font-weight:700;padding:1px 6px;border-radius:4px;
  margin-left:6px;vertical-align:middle}
.hist-date{color:#64748b;white-space:nowrap;font-size:.82rem}
.hist-dur{color:#64748b;white-space:nowrap;font-size:.82rem}
.hist-score{white-space:nowrap}
.hist-grade{display:inline-block;padding:1px 7px;border-radius:5px;
  font-size:.72rem;font-weight:700;border:1px solid}
.hist-val{font-size:.82rem;color:#334155;white-space:nowrap}
.hist-badge{display:inline-block;padding:2px 7px;border-radius:5px;
  font-size:.75rem;font-weight:700;margin-right:3px}
.hist-crit{background:#fee2e2;color:#ef4444}
.hist-warn{background:#ffedd5;color:#f97316}
.hist-ok{background:#dcfce7;color:#16a34a}
/* dark mode for history */
body.dark .hist-alert{background:#431407;border-color:#92400e;color:#fed7aa}
body.dark .spark-card{background:#1e293b;border-color:#334155}
body.dark .spark-label{color:#64748b}
body.dark .spark-val{color:#e2e8f0}
body.dark .spark-slope{color:#475569}
body.dark .hist-table thead th{background:#162032;color:#64748b;border-bottom-color:#334155}
body.dark .hist-table tbody tr{border-bottom-color:#1e293b}
body.dark .hist-table tbody tr:hover{background:#162032}
body.dark .hist-row-current{background:#1e3a5f!important}
body.dark .hist-log{color:#94a3b8}
body.dark .hist-date,.hist-dur{color:#64748b}
body.dark .hist-val{color:#cbd5e1}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Embedded JS
# ─────────────────────────────────────────────────────────────────────────────

_JS = """
// ── Animate health bars on load ──
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.bar-fill:not(.bar-na)').forEach(function(el) {
    var w = el.style.width; el.style.width='0%';
    setTimeout(function(){el.style.width=w;},120);
  });
  initScrollSpy();
});

// ── Accordion toggle ──
function toggleAcc(id) {
  var body   = document.getElementById(id+'-body');
  var header = document.querySelector('#'+id+' .acc-header');
  if (!body) return;
  var open = body.classList.contains('open');
  body.classList.toggle('open',!open);
  if (header) header.classList.toggle('open',!open);
}

// ── Expand / Collapse All (Tier1 #5) ──
function expandAll() {
  document.querySelectorAll('.acc-body').forEach(function(b){b.classList.add('open')});
  document.querySelectorAll('.acc-header').forEach(function(h){h.classList.add('open')});
}
function collapseAll() {
  document.querySelectorAll('.acc-body').forEach(function(b){b.classList.remove('open')});
  document.querySelectorAll('.acc-header').forEach(function(h){h.classList.remove('open')});
}

// ── Issue severity filter ──
function filterIssues(sev, btn) {
  document.querySelectorAll('.filter-btn').forEach(function(b){b.classList.remove('active')});
  if (btn) btn.classList.add('active');
  document.querySelectorAll('.issue-card').forEach(function(card){
    if (sev==='all' || card.dataset.sev===sev) card.classList.remove('hidden');
    else card.classList.add('hidden');
  });
}

// ── Scrollspy for sticky nav (Tier1 #4) ──
function initScrollSpy() {
  var links = document.querySelectorAll('.nav-link');
  if (!links.length) return;

  var targets = [];
  links.forEach(function(link) {
    var id = link.getAttribute('data-target');
    var el = document.getElementById(id);
    if (el) targets.push({link:link, el:el});
  });

  var obs = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        targets.forEach(function(t){t.link.classList.remove('active')});
        targets.forEach(function(t){
          if (t.el===entry.target) t.link.classList.add('active');
        });
      }
    });
  },{rootMargin:'-20% 0px -70% 0px'});

  targets.forEach(function(t){obs.observe(t.el)});
}

// ── Tier-2: Dark mode toggle ──
function toggleDark() {
  var isDark = document.body.classList.toggle('dark');
  try { localStorage.setItem('uav-dark', isDark ? '1' : '0'); } catch(e){}
  document.getElementById('dm-toggle').textContent = isDark ? '☀' : '🌙';
}
(function() {
  try {
    if (localStorage.getItem('uav-dark') === '1') {
      document.body.classList.add('dark');
      document.addEventListener('DOMContentLoaded', function() {
        var btn = document.getElementById('dm-toggle');
        if (btn) btn.textContent = '☀';
      });
    }
  } catch(e) {}
})();

// ── Tier-2: Issue detail expand on click ──
function toggleIssue(id) {
  var detail = document.getElementById(id + '-detail');
  if (!detail) return;
  var isOpen = detail.classList.contains('open');
  // Close all others first
  document.querySelectorAll('.iss-detail.open').forEach(function(d) {
    d.classList.remove('open');
    var card = d.parentElement;
    if (card) {
      var hint = card.querySelector('.iss-expand-hint');
      if (hint) hint.textContent = '▸ click for details';
    }
  });
  if (!isOpen) {
    detail.classList.add('open');
    var card = document.getElementById(id);
    if (card) {
      var hint = card.querySelector('.iss-expand-hint');
      if (hint) hint.textContent = '▾ click to close';
    }
  }
}

// ── Tier-2: Module scorecard drill-down ──
function drillDown(accId) {
  var body   = document.getElementById(accId + '-body');
  var header = document.querySelector('#' + accId + ' .acc-header');
  var detail = document.getElementById('sec-detail');
  if (!body) return;
  // Open the accordion
  body.classList.add('open');
  if (header) header.classList.add('open');
  // Scroll to module detail section, then to the specific accordion
  if (detail) {
    detail.scrollIntoView({behavior:'smooth', block:'start'});
    setTimeout(function() {
      var acc = document.getElementById(accId);
      if (acc) acc.scrollIntoView({behavior:'smooth', block:'nearest'});
    }, 400);
  }
}
"""
