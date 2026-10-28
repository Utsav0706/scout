"""Run the golden-set eval.

    py -m scout.eval run                          # every case (needs Postgres for docs cases)
    py -m scout.eval run --tags ci                # the CI subset
    py -m scout.eval run --tags web --database-url sqlite:///scout_eval.db

Each run writes runs/eval/<time>/results.jsonl (one line per case, written as it
finishes), summary.json and summary.md. Costs money: it runs the real agent and judges.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from scout.agent.runtime import Services
from scout.config import get_config, get_secrets
from scout.db import db_is_up, init_db, make_engine
from scout.eval.cases import load_cases
from scout.eval.runner import CaseResult, run_suite
from scout.eval.scorers import JUDGE_PROMPTS
from scout.eval.summary import summarise, to_markdown
from scout.log import setup_logging
from scout.telemetry import setup_tracing

REPO = Path(__file__).resolve().parents[3]


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scout.eval", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run cases and score them")
    run.add_argument("--cases", type=Path, default=REPO / "eval" / "golden.yaml")
    run.add_argument("--corpus", type=Path, default=REPO / "eval" / "corpus")
    run.add_argument("--tags", nargs="*", help="only cases having all these tags")
    run.add_argument("--ids", nargs="*", help="only these case ids")
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--max-cost-gbp", type=float, default=5.0, help="stop starting cases here")
    run.add_argument("--database-url", help="override DATABASE_URL")
    run.add_argument("--out", type=Path, default=REPO / "runs" / "eval")
    run.add_argument(
        "--min-task-success",
        type=float,
        help="exit 1 if task success is below this (0-1): the eval gate for CI",
    )
    run.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


async def _run(args: argparse.Namespace) -> int:
    secrets, config = get_secrets(), get_config()
    if secrets.anthropic_api_key is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set (see .env.example).")
    setup_tracing(secrets)  # each case's trace goes to Langfuse when its keys are set

    cases = load_cases(args.cases, tags=args.tags, ids=args.ids)
    if not cases:
        raise SystemExit("No cases match.")
    url = args.database_url or secrets.database_url
    engine = make_engine(url)
    if not db_is_up(engine):
        raise SystemExit(
            f"Database not reachable at {url}. Try --database-url sqlite:///scout_eval.db"
        )
    init_db(engine)
    use_docs = engine.dialect.name == "postgresql"
    if not use_docs and any(c.needs_docs for c in cases):
        print(
            "Note: SQLite has no document search; docs and both cases are skipped.", file=sys.stderr
        )

    started = datetime.now(UTC)
    out = args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    meta = {
        "started_at": started.isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "cases_file": str(args.cases),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest()[:12],
        "config_sha256": hashlib.sha256((REPO / "config.yaml").read_bytes()).hexdigest()[:12],
        "judge_model": config.llm.nodes["judge"].model,
        "judge_prompts": JUDGE_PROMPTS,
        "tags": args.tags,
        "ids": args.ids,
        "document_search": use_docs,
        "max_cost_gbp": args.max_cost_gbp,
    }
    print(f"Running {len(cases)} cases -> {out}", file=sys.stderr)

    results_file = (out / "results.jsonl").open("a", encoding="utf-8")

    def save(r: CaseResult) -> None:
        results_file.write(r.model_dump_json() + "\n")
        results_file.flush()
        mark = {True: "PASS", False: "FAIL"}.get(r.passed, r.outcome.upper())  # type: ignore[arg-type]
        print(f"  {mark:7} {r.id}  £{r.cost_gbp + r.judge_cost_gbp:.4f}", file=sys.stderr)

    services = Services.create(config, secrets, sessionmaker(engine, expire_on_commit=False))
    try:
        results = await run_suite(
            services,
            cases,
            corpus_dir=args.corpus,
            use_docs=use_docs,
            concurrency=args.concurrency,
            max_cost_gbp=args.max_cost_gbp,
            on_result=save,
        )
    finally:
        results_file.close()

    summary = summarise(results)
    meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, indent=2), encoding="utf-8"
    )
    report = to_markdown(summary, meta, results)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)

    success = summary["overall"]["task_success"]
    if args.min_task_success is not None and (success or 0) < args.min_task_success:
        print(
            f"Eval gate failed: task success {success} < {args.min_task_success}", file=sys.stderr
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # £ on Windows consoles
    setup_logging("INFO" if args.verbose else "WARNING", stream=sys.stderr)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
