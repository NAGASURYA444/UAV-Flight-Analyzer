"""
UAV Flight Analyzer — Documentation PDF Generator
===================================================
Converts UAV_Flight_Analyzer_User_Manual.md into a professionally styled
HTML file and (optionally) a PDF using weasyprint.

Usage
-----
    # Install dependencies first (one time):
    pip install markdown2 weasyprint

    # Run from the project root:
    python docs/generate_pdf.py

Outputs
-------
    docs/UAV_Flight_Analyzer_User_Manual.html   (always generated)
    docs/UAV_Flight_Analyzer_User_Manual.pdf    (if weasyprint is available)

If weasyprint is not installed or fails (e.g. missing GTK on Windows),
the script still writes the HTML file and prints browser-print instructions.
"""

import sys
import os
from pathlib import Path
from datetime import date

DOCS_DIR  = Path(__file__).parent
MD_FILE   = DOCS_DIR / "UAV_Flight_Analyzer_User_Manual.md"
HTML_FILE = DOCS_DIR / "UAV_Flight_Analyzer_User_Manual.html"
PDF_FILE  = DOCS_DIR / "UAV_Flight_Analyzer_User_Manual.pdf"

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ===== Reset & Page ===== */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

@page {
    size: A4;
    margin: 22mm 18mm 22mm 22mm;
}

body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #1a1a2e;
    background: #ffffff;
}

/* ===== Cover Page ===== */
.cover {
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    min-height: 240mm;
    padding: 30mm 20mm;
    background: #0f1b35;
    color: #ffffff;
}

.cover .logo-line {
    font-size: 11pt;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7eb3e8;
    margin-bottom: 8mm;
}

.cover h1 {
    font-size: 28pt;
    font-weight: 700;
    line-height: 1.2;
    color: #ffffff;
    margin-bottom: 4mm;
}

.cover .subtitle {
    font-size: 13pt;
    color: #a8c4e0;
    margin-bottom: 16mm;
}

.cover .meta-table {
    border-collapse: collapse;
    margin-top: 4mm;
}

.cover .meta-table td {
    padding: 2mm 6mm 2mm 0;
    font-size: 10pt;
    color: #c8d8ec;
    border: none;
    background: transparent;
}

.cover .meta-table td:first-child {
    color: #7eb3e8;
    font-weight: 600;
    min-width: 30mm;
}

.cover .accent-bar {
    width: 60mm;
    height: 1mm;
    background: #3a7bd5;
    margin-bottom: 6mm;
}

/* ===== TOC Page ===== */
.toc-section {
    page-break-after: always;
    padding: 0 0 8mm 0;
}

.toc-section h2 {
    font-size: 16pt;
    color: #0f1b35;
    border-bottom: 2px solid #3a7bd5;
    padding-bottom: 3mm;
    margin-bottom: 5mm;
}

.toc-section ul {
    list-style: none;
    padding: 0;
}

.toc-section li {
    padding: 1.5mm 0;
    font-size: 10pt;
    border-bottom: 1px dotted #d0d8e8;
    display: flex;
    justify-content: space-between;
}

.toc-section li a {
    color: #1a3a6e;
    text-decoration: none;
}

/* ===== Headings ===== */
h1, h2, h3, h4, h5 {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: #0f1b35;
    line-height: 1.3;
}

h1 {
    font-size: 20pt;
    margin: 0 0 4mm 0;
    color: #0f1b35;
}

h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #0f1b35;
    border-bottom: 2px solid #3a7bd5;
    padding-bottom: 2mm;
    margin: 10mm 0 4mm 0;
    page-break-before: always;
    page-break-after: avoid;
}

h2:first-of-type {
    page-break-before: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #1a3a6e;
    margin: 6mm 0 2mm 0;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #2a4a7e;
    margin: 4mm 0 2mm 0;
    page-break-after: avoid;
}

/* ===== Paragraphs & Text ===== */
p {
    margin: 0 0 3mm 0;
    text-align: justify;
}

strong { color: #0f1b35; }

em { color: #2a4a7e; }

/* ===== Tables ===== */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 3mm 0 5mm 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

thead tr {
    background: #0f1b35;
    color: #ffffff;
}

thead th {
    padding: 2.5mm 3mm;
    text-align: left;
    font-weight: 600;
    font-size: 8.5pt;
    letter-spacing: 0.3px;
}

tbody tr:nth-child(even) {
    background: #f0f4fb;
}

tbody tr:nth-child(odd) {
    background: #ffffff;
}

tbody td {
    padding: 2mm 3mm;
    border-bottom: 1px solid #dde4f0;
    vertical-align: top;
}

/* Severity column color-coding */
tbody td:nth-child(2) {
    font-weight: 600;
    white-space: nowrap;
}

/* ===== Code Blocks ===== */
pre {
    background: #1a1a2e;
    color: #c8d8ec;
    border-radius: 3px;
    padding: 4mm;
    margin: 3mm 0 4mm 0;
    font-size: 8.5pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
    border-left: 3px solid #3a7bd5;
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
}

p code, li code, td code {
    background: #e8eef8;
    color: #1a3a6e;
    padding: 0.3mm 1.5mm;
    border-radius: 2px;
    font-size: 8.5pt;
}

/* ===== Lists ===== */
ul, ol {
    margin: 2mm 0 3mm 5mm;
    padding-left: 5mm;
}

li {
    margin: 1mm 0;
    line-height: 1.5;
}

/* ===== Horizontal Rule ===== */
hr {
    border: none;
    border-top: 1px solid #c0cce0;
    margin: 6mm 0;
}

/* ===== Severity Color-coding ===== */
.sev-critical { color: #c0392b; }
.sev-warning  { color: #d68910; }
.sev-info     { color: #1a6ea8; }

/* ===== Note Blocks ===== */
blockquote {
    background: #eef3fb;
    border-left: 3px solid #3a7bd5;
    padding: 3mm 4mm;
    margin: 3mm 0;
    font-size: 9pt;
    color: #2a4a7e;
}

/* ===== Footer / Page numbers handled by @page ===== */
"""

# ── HTML Template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UAV Flight Analyzer — Technical Documentation & User Manual</title>
<style>
{css}
</style>
</head>
<body>

<!-- ========== COVER PAGE ========== -->
<div class="cover">
  <div class="logo-line">Technical Documentation</div>
  <h1>UAV Flight Analyzer</h1>
  <div class="subtitle">Technical Documentation &amp; User Manual</div>
  <div class="accent-bar"></div>
  <table class="meta-table">
    <tr><td>Version</td><td>1.0</td></tr>
    <tr><td>Date</td><td>{date}</td></tr>
    <tr><td>Platform</td><td>ArduPilot DataFlash .BIN Log Analysis</td></tr>
    <tr><td>Modules</td><td>18 Analysis Modules</td></tr>
    <tr><td>Vehicle Types</td><td>Quadcopter &bull; VTOL &bull; Fixed-Wing</td></tr>
    <tr><td>Output Formats</td><td>HTML Report &bull; JSON &bull; Fleet SQLite</td></tr>
  </table>
</div>

<!-- ========== DOCUMENT BODY ========== -->
<div class="doc-body">
{body}
</div>

</body>
</html>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _post_process_html(html: str) -> str:
    """Light post-processing: colour-code severity words inside table cells."""
    html = html.replace(">Critical<", ' class="sev-critical">Critical<')
    html = html.replace(">Warning<",  ' class="sev-warning">Warning<')
    html = html.replace(">Info<",     ' class="sev-info">Info<')
    return html


def build_html(md_text: str) -> str:
    try:
        import markdown2
    except ImportError:
        print("[ERROR] markdown2 is not installed.")
        print("        Run: pip install markdown2")
        sys.exit(1)

    body_html = markdown2.markdown(
        md_text,
        extras=[
            "tables",
            "fenced-code-blocks",
            "header-ids",
            "strike",
            "footnotes",
        ],
    )
    body_html = _post_process_html(body_html)

    today = date.today().strftime("%B %Y")
    return HTML_TEMPLATE.format(css=CSS, body=body_html, date=today)


def write_html(html: str) -> None:
    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"[OK] HTML written: {HTML_FILE}")


def write_pdf(html: str) -> bool:
    # Try xhtml2pdf first (pure Python, no GTK needed on Windows)
    try:
        from xhtml2pdf import pisa
        with open(PDF_FILE, "wb") as f:
            result = pisa.CreatePDF(html.encode("utf-8"), dest=f,
                                    encoding="utf-8")
        if result.err:
            print(f"[WARN] xhtml2pdf reported errors: {result.err}")
            return False
        print(f"[OK] PDF written:  {PDF_FILE}")
        return True
    except ImportError:
        pass
    except Exception as exc:
        print(f"[WARN] xhtml2pdf failed: {exc}")

    # Try weasyprint as secondary option (needs GTK on Windows)
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(string=html, base_url=str(DOCS_DIR)).write_pdf(str(PDF_FILE))
        print(f"[OK] PDF written:  {PDF_FILE}")
        return True
    except ImportError:
        pass
    except Exception as exc:
        print(f"[WARN] weasyprint failed: {exc}")

    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not MD_FILE.exists():
        print(f"[ERROR] Source not found: {MD_FILE}")
        sys.exit(1)

    print(f"Reading: {MD_FILE}")
    md_text = MD_FILE.read_text(encoding="utf-8")

    print("Converting Markdown -> HTML...")
    html = build_html(md_text)

    write_html(html)

    print("Attempting PDF generation via weasyprint...")
    ok = write_pdf(html)

    if not ok:
        print()
        print("=" * 60)
        print("  PDF generation via weasyprint was not available.")
        print()
        print("  Option A — Install weasyprint:")
        print("    pip install weasyprint")
        print()
        print("  Option B — Print from browser (no install needed):")
        print(f"    1. Open: {HTML_FILE}")
        print("    2. Press Ctrl+P  (or Cmd+P on Mac)")
        print('    3. Destination -> "Save as PDF"')
        print("    4. Paper size: A4, Margins: Default")
        print("    5. Click Save")
        print("=" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
