# UAV Flight Analyzer

A local web application for analyzing ArduPilot `.bin` DataFlash logs, scoring flight health, and tracking fleet maintenance trends.

---

## Features

- Analyze ArduPilot `.bin` flight logs (QuadCopter, VTOL QuadPlane, Fixed-Wing)
- 18 analysis modules: battery, motors, vibration, GPS, control, efficiency, and more
- Flight health score with A–F grade
- Downloadable HTML report per flight
- Fleet dashboard with trend graphs and maintenance alerts
- Fully local — no internet required after first setup, data never leaves your machine

---

## Quick Start — Windows

**Step 1 — Get the project** (one time only)

Option A — using Git:
```bash
git clone https://github.com/NAGASURYA444/UAV-Flight-Analyzer.git
cd UAV-Flight-Analyzer
```

Option B — no Git needed:
1. Click the green **Code** button on this page → **Download ZIP**
2. Extract the ZIP anywhere on your machine (e.g. `C:\UAV-Flight-Analyzer`)
3. Open that folder

**Step 2 — Double-click `run.bat`**

The launcher handles everything from here automatically:

1. Detects a compatible Python (3.10–3.13) or installs Python 3.13 via winget
2. Creates an isolated `.venv` virtual environment on first run
3. Downloads and installs all dependencies (takes 1–2 min the first time)
4. Starts the server and opens `http://localhost:5000` in your browser

**Every run after the first:** just double-click `run.bat` again — starts in seconds.

> **Note:** Python 3.14 (alpha) is intentionally skipped — it has a known DLL
> incompatibility with `pydantic_core` on Windows. The launcher targets 3.10–3.13.

---

## Quick Start — macOS / Linux

```bash
# Clone and enter the project
git clone https://github.com/NAGASURYA444/UAV-Flight-Analyzer.git
cd UAV-Flight-Analyzer

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the server
python run_web.py
```

Then open **http://localhost:5000** in your browser.

---

## Requirements

| Platform | Requirement |
|---|---|
| Windows | Python 3.10–3.13 (auto-installed by `run.bat` if missing) |
| macOS / Linux | Python 3.10–3.13 installed manually |
| All | Internet connection on first run (to download dependencies) |

---

## Daily Use

**Windows:** double-click `run.bat`.

**macOS / Linux:**
```bash
source .venv/bin/activate
python run_web.py
```

Then open **http://localhost:5000** in your browser.

---

## How to Analyze a Flight

1. Go to the **Analyze** tab
2. Enter a Drone ID (e.g. `Quadcopter-01`)
3. Select the vehicle type (Quadcopter / VTOL QuadPlane / Fixed-Wing)
4. Upload your `.bin` log file
5. Click **Run Analysis**
6. View the score, issues, and download the HTML report

---

## Vehicle Profiles

Default profiles are in the `profiles/` folder:

| File | Vehicle |
|---|---|
| `profiles/quadcopter.yaml` | Quadcopter (4 motors) |
| `profiles/quadplane.yaml` | VTOL QuadPlane (4 lift + 1 pusher) |
| `profiles/fixed_wing.yaml` | Fixed-Wing (single motor) |

Edit these YAML files to match your specific airframe (battery capacity, motor channels, thresholds, etc.) — or adjust them directly in the Profile Settings section of the web UI before running analysis.

---

## Fleet Dashboard

The **Fleet Dashboard** tab shows:
- All drones analyzed so far
- Trend graphs across flights (score, battery IR, vibration, motor imbalance)
- Maintenance alerts when a metric is deteriorating
- Full flight history per drone with CSV export

Fleet data is stored locally in `fleet.db` (SQLite — auto-created on first analysis).

---

## Updating

```bash
git pull
```

Then double-click `run.bat` (Windows) or run `pip install -r requirements.txt && python run_web.py` (macOS/Linux). The launcher automatically reinstalls dependencies if `requirements.txt` changed.

---

## Supported Log Types

| Vehicle | ArduPilot Firmware | Profile |
|---|---|---|
| Quadcopter / Hex / Octo | ArduCopter | `quadcopter.yaml` |
| VTOL QuadPlane | ArduPlane (Q-modes) | `quadplane.yaml` |
| Fixed-Wing | ArduPlane | `fixed_wing.yaml` |

Logs must be ArduPilot DataFlash `.bin` format (from flight controller SD card).

---

## Troubleshooting

**Port already in use**
`run.bat` will warn you. If port 5000 is taken by another app, edit `run_web.py` and change `port=5000` to any free port (e.g. `5001`), then open `http://localhost:5001`.

**Dependencies not installing**
Delete the `.venv` folder and double-click `run.bat` again — it will rebuild from scratch.

**pymavlink install fails on Windows**
`run.bat` handles this automatically using pre-built binary wheels. If you are installing manually, use:
```bash
pip install -r requirements.txt --only-binary=pymavlink
```

**Log not parsing**
Make sure the file is a raw DataFlash `.bin` from the SD card — not a `.tlog` (telemetry log) or `.log` (text log).

---

## Project Structure

```
UAV-Flight-Analyzer/
├── run.bat                 # Windows one-click launcher
├── run_web.py              # Start the web server (used by run.bat)
├── cli.py                  # Command-line interface (optional)
├── requirements.txt
├── profiles/               # Airframe YAML profiles
│   ├── quadcopter.yaml
│   ├── quadplane.yaml
│   └── fixed_wing.yaml
├── analyzer/               # Analysis engine
│   ├── modules/            # 18 analysis modules
│   ├── parser/             # .bin log parser
│   ├── scoring/            # Score aggregation
│   └── report/             # HTML + JSON report generation
├── fleet/                  # Fleet DB + trend analysis
├── web/                    # FastAPI backend + frontend
│   ├── app.py
│   └── static/index.html
└── reports/                # Generated HTML reports (auto-created)
```

---

## Feedback & Bug Reports

Suggestions, bug reports, and real-world use cases are welcome.

- **GitHub Issues:** [Open an issue](https://github.com/NAGASURYA444/UAV-Flight-Analyzer/issues)
- **Email:** nagasurya@aereo.io

If you fly ArduPilot drones — try it, use it, and share your feedback. It helps make the tool better for everyone.

---

## Built With

Developed using **Claude AI** through vibe coding.

---

## License

For internal / operational use. Contact the maintainer for licensing questions.
