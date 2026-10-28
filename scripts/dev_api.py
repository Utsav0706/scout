"""Run the API for local development, without Docker or Redis.

    py scripts/dev_api.py            # http://localhost:8001 (docs at /docs)

Port 8001, not 8000: Windows often reserves 8000 (Hyper-V / WSL port ranges).
Defaults suited to a Windows laptop: jobs run inside the API (JOBS_MODE=inline, since RQ
workers can't run on Windows) and data goes to a local SQLite file (document search,
which needs Postgres + pgvector, is unavailable). Real environment variables win over
these defaults; these win over .env (which usually points at Docker's Postgres).
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

# Work from the repo root wherever this is started from: .env, config.yaml and prompts/
# are found relative to it.
os.chdir(Path(__file__).resolve().parents[1])
os.environ.setdefault("JOBS_MODE", "inline")
os.environ.setdefault("DATABASE_URL", "sqlite:///scout_dev.db")

if __name__ == "__main__":
    uvicorn.run(
        "scout.api.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "8001")),
    )
