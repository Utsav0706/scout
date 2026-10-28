"""Send a user's thumbs up/down to Langfuse as a score on that run's trace, so the runs
people disliked can be filtered in Langfuse and inspected step by step.

Best effort: Scout's own feedback table is the record; a Langfuse outage or missing keys
never fails the user's request.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx

from scout.config import Secrets

log = logging.getLogger(__name__)

SCORE_NAME = "user_feedback"

# (trace_id, job_id, rating, comment) -> None
ScoreSender = Callable[[str, str, str, str | None], None]


def langfuse_score_sender(secrets: Secrets, timeout_s: float = 5.0) -> ScoreSender | None:
    if not (secrets.langfuse_public_key and secrets.langfuse_secret_key):
        return None
    auth = (
        secrets.langfuse_public_key.get_secret_value(),
        secrets.langfuse_secret_key.get_secret_value(),
    )
    url = f"{secrets.langfuse_host.rstrip('/')}/api/public/scores"

    def send(trace_id: str, job_id: str, rating: str, comment: str | None) -> None:
        body = {
            # A stable id: re-rating the same report (same day) updates this score.
            "id": f"{job_id}-feedback",
            "traceId": trace_id,
            "name": SCORE_NAME,
            "value": 1 if rating == "up" else 0,
            "dataType": "BOOLEAN",
            "comment": comment,
        }
        try:
            resp = httpx.post(url, json=body, auth=auth, timeout=timeout_s)
            if resp.status_code >= 400:
                log.warning("langfuse score rejected", extra={"status": resp.status_code})
        except httpx.HTTPError as exc:
            log.warning("langfuse score not sent", extra={"error": type(exc).__name__})

    return send
