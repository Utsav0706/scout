"""Eval cases: one question each, with what a good report must contain.

The fields follow Dyno's case format so the same YAML file works there later:
id, input, expected, must_call, must_not_call, max_steps, tags.

Scout-specific meaning:
- expected: the key points a correct report covers. Empty for "unanswerable" cases,
  where the right answer is that there isn't enough evidence.
- tags: exactly one category (web, docs, both, unanswerable), plus optional extras
  such as "ci" for the small subset run in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Tool = Literal["web_search", "fetch_url", "rag_search", "save_note"]
CATEGORIES = ("web", "docs", "both", "unanswerable")


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    input: str = Field(min_length=5)
    expected: list[str] = Field(default_factory=list)
    must_call: list[Tool] = Field(default_factory=list)
    must_not_call: list[Tool] = Field(default_factory=list)
    max_steps: int | None = None
    tags: list[str]

    @property
    def category(self) -> str:
        return next(t for t in self.tags if t in CATEGORIES)

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @property
    def needs_docs(self) -> bool:
        return self.category in ("docs", "both") or "rag_search" in self.must_call

    @model_validator(mode="after")
    def check(self) -> Case:
        found = [t for t in self.tags if t in CATEGORIES]
        if len(found) != 1:
            raise ValueError(f"{self.id}: needs exactly one category tag of {CATEGORIES}")
        if self.answerable and not self.expected:
            raise ValueError(f"{self.id}: an answerable case needs expected key points")
        if not self.answerable and self.expected:
            raise ValueError(f"{self.id}: an unanswerable case has no expected key points")
        if overlap := set(self.must_call) & set(self.must_not_call):
            raise ValueError(f"{self.id}: {sorted(overlap)} both required and forbidden")
        return self


def load_cases(
    path: Path, tags: list[str] | None = None, ids: list[str] | None = None
) -> list[Case]:
    """All cases in the file, or those having every tag in `tags` / listed in `ids`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = [Case.model_validate(c) for c in raw["cases"]]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id}")
        seen.add(c.id)
    if tags:
        cases = [c for c in cases if set(tags) <= set(c.tags)]
    if ids:
        unknown = set(ids) - seen
        if unknown:
            raise ValueError(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c.id in ids]
    return cases
