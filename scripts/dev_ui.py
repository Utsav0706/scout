"""Run the Streamlit UI for local development, pointed at scripts/dev_api.py.

    py scripts/dev_ui.py             # http://localhost:8501

SCOUT_API_URL wins if set; otherwise the dev API's address (http://localhost:8001).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"

if __name__ == "__main__":
    os.environ.setdefault("SCOUT_API_URL", "http://localhost:8001")
    port = os.environ.get("PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        str(APP),
        "--server.port",
        port,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    sys.exit(cli.main())
