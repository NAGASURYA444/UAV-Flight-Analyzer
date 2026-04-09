# UAV Flight Analyzer

A local web application for analyzing ArduPilot `.bin` DataFlash logs, scoring flight health, and tracking fleet maintenance trends.

---

## Features

- Analyze ArduPilot `.bin` flight logs (QuadCopter, VTOL QuadPlane, Fixed-Wing)
- 18 analysis modules: battery, motors, vibration, GPS, control, efficiency, and more
- Flight health score with A–F grade
- Downloadable HTML report per flight
- Fleet dashboard with trend graphs and maintenance alerts
- Fully local — no internet required, data never leaves your machine

---

## Requirements

- Python 3.10 or newer
- Windows / macOS / Linux

---

## Setup (First Time)

**1. Clone the repository**
```bash
git clone https://github.com/NAGASURYA444/UAV-Flight-Analyzer.git
cd UAV-Flight-Analyzer
```

**2. (Recommended) Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the server**
```bash
python run_web.py
```

**5. Open your browser**
```
http://localhost:5000
```

---

## Daily Use

After the first-time setup, just run:
```bash
# Activate virtual environment (if you created one)
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Start the server
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

To get the latest version:
```bash
git pull
pip install -r requirements.txt   # only if dependencies changed
python run_web.py
```

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
If port 5000 is taken by another app, edit `run_web.py` and change `port=5000` to any free port (e.g. `5001`), then open `http://localhost:5001`.

**Missing dependencies**
```bash
pip install -r requirements.txt
```

**pymavlink install fails on Windows**
Install Microsoft C++ Build Tools first:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Log not parsing**
Make sure the file is a raw DataFlash `.bin` from the SD card — not a `.tlog` (telemetry log) or `.log` (text log).

---

## Project Structure

```
UAV-Flight-Analyzer/
├── run_web.py              # Start the web server
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

## License

For internal / operational use. Contact the maintainer for licensing questions.
