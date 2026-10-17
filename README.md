# Scout

An autonomous research agent. Ask a question; Scout plans the research, searches the
web and a private document store, reads sources, verifies every claim, and returns a
report where each claim links to its source.

> Headline (filled in once measured): *"Scout answers research questions with X% of
> claims verified against sources, in Y seconds, for £Z per report."*

Status: **M0 — project setup.** Nothing runs yet beyond config, logging and tests.

## Dev setup

Needs Python 3.12+.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Check everything:

```bash
ruff check .
ruff format --check .
pytest
```

## Layout

| Path | What |
|---|---|
| `config.yaml` | Models per agent node, cost/step limits. No secrets. |
| `.env` | API keys (copy from `.env.example`, never committed). |
| `src/scout/` | The package. |
| `prompts/` | Versioned prompt files (from M4). |
| `docs/` | Design notes. |
| `tests/` | pytest suite, run in CI on every push/PR. |
