"""Versioned prompt files.

Prompts live in prompts/<name>_v<N>.md, reviewed like code. Each report records the
version and a short content hash of every prompt it used, so a change in results can be
traced to the exact prompt text that produced it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


def _prompts_dir() -> Path:
    # SCOUT_PROMPTS_DIR wins; else the repo checkout (editable install, tests); else the
    # working directory (the Docker image copies prompts/ next to config.yaml).
    if env := os.environ.get("SCOUT_PROMPTS_DIR"):
        return Path(env)
    repo = Path(__file__).resolve().parents[3] / "prompts"
    return repo if repo.is_dir() else Path.cwd() / "prompts"


PROMPTS_DIR = _prompts_dir()


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str

    @property
    def tag(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:8]
        return f"v{self.version}:{digest}"

    def render(self, **values: object) -> str:
        return self.text.format(**values)


def load_prompt(name: str, version: int = 1, directory: Path = PROMPTS_DIR) -> Prompt:
    path = directory / f"{name}_v{version}.md"
    return Prompt(name, version, path.read_text(encoding="utf-8").strip())
