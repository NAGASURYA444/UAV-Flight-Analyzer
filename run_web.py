"""
UAV Flight Analyzer — Web Server Startup
=========================================
Run this script to start the web dashboard:

    python run_web.py

Then open http://localhost:5000 in your browser.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=5000,
    )
