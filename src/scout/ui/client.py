"""ScoutClient: everything the UI does, as calls to the Scout HTTP API.

The UI never touches the database or the agent directly: it is just another client of
the same API that Dyno and any other tool use. Tests hand this class the real API
in-process (FastAPI's TestClient is an httpx.Client), so UI actions are tested end to end.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class ScoutError(Exception):
    """An API call failed; `message` is safe to show to the user."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def _detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return resp.text[:200] or resp.reason_phrase
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(d.get("msg", d)) for d in detail)
    if isinstance(detail, dict):
        return str(detail.get("message", detail))
    return str(detail)


class ScoutClient:
    def __init__(
        self,
        base_url: str | None = None,
        http: httpx.Client | None = None,
        token: str | None = None,
    ) -> None:
        self._http = http or httpx.Client(
            base_url=base_url or os.environ.get("SCOUT_API_URL", DEFAULT_API_URL), timeout=30
        )
        self.token = token
        self.email: str | None = None

    # --- plumbing ---

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._http.request(method, path, headers=self._headers(), **kwargs)
        if resp.status_code >= 400:
            if resp.status_code == 401:
                self.token = None  # expired or invalid: the UI shows the login page again
            raise ScoutError(resp.status_code, _detail(resp))
        return resp.json() if resp.content else None

    # --- account ---

    def register(self, email: str, password: str) -> None:
        self._request("POST", "/auth/register", json={"email": email, "password": password})

    def login(self, email: str, password: str) -> None:
        body = self._request("POST", "/auth/login", json={"email": email, "password": password})
        self.token, self.email = body["access_token"], email.strip().lower()

    def logout(self) -> None:
        self.token = self.email = None

    # --- research ---

    def start_research(self, question: str, use_web: bool = True, use_docs: bool = True) -> str:
        body = self._request(
            "POST",
            "/research",
            json={"question": question, "use_web": use_web, "use_docs": use_docs},
        )
        return body["job_id"]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/research/{job_id}")

    def history(self, limit: int = 20, before: str | None = None) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        return self._request("GET", "/research", params=params)

    def stats(self, days: int = 30) -> dict:
        return self._request("GET", "/research/stats", params={"days": days})

    def feedback(self, job_id: str, rating: str, comment: str | None = None) -> dict:
        return self._request(
            "POST", f"/research/{job_id}/feedback", json={"rating": rating, "comment": comment}
        )

    def stream(self, job_id: str, last_event_id: str | None = None) -> Iterator[tuple[str, dict]]:
        """Live progress as (event, data) pairs, ending with ("done", {"status": ...}).
        Reads the server-sent events of GET /research/{id}/stream."""
        headers = self._headers()
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        with self._http.stream(
            "GET", f"/research/{job_id}/stream", headers=headers, timeout=None
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise ScoutError(resp.status_code, _detail(resp))
            event, data = "message", ""
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = line[6:]
                elif not line and data:  # a blank line ends one event
                    yield event, json.loads(data)
                    event, data = "message", ""

    # --- documents ---

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, filename: str, content: bytes) -> dict:
        return self._request(
            "POST", "/documents", files={"file": (filename, content, "application/pdf")}
        )
