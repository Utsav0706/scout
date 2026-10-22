"""Smoke test for the full Docker Compose stack: API -> Redis -> RQ worker -> Postgres.

Runs in CI with NO Anthropic key, so it costs nothing: the worker must pick up the job
and fail it with a clean, known error. That proves a job travels the whole real path
(queue, worker, database, events) without spending money on a model call.

Usage (stack already up):  python scripts/compose_smoke.py http://localhost:8000
Standard library only, so CI needn't install anything to run it.
"""

from __future__ import annotations

import json
import secrets
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def call(
    method: str, path: str, body: dict | None = None, token: str | None = None
) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def main() -> int:
    status, health = call("GET", "/health")
    assert status == 200 and health["db"] == "ok", health

    creds = {"email": "smoke@example.com", "password": secrets.token_urlsafe(16)}
    assert call("POST", "/auth/register", creds)[0] == 201
    token = call("POST", "/auth/login", creds)[1]["access_token"]

    status, created = call(
        "POST", "/research", {"question": "Smoke test question?", "use_docs": False}, token
    )
    assert status == 202, created
    job_id = created["job_id"]

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        _, job = call("GET", f"/research/{job_id}", token=token)
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(1)
    else:
        print("FAIL: job never finished; is the worker running?", job)
        return 1

    messages = [e["message"] for e in job["events"]]
    print("status:", job["status"], "| error:", job["error"], "| events:", messages)
    assert "Research started" in messages, "the worker never picked the job up"
    assert job["status"] == "failed", "expected a clean failure (no Anthropic key in CI)"
    assert job["error"] == "Research failed (RuntimeError).", job["error"]
    print("OK: API -> Redis -> worker -> Postgres works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
