"""Run one research question from the command line (the HTTP API arrives in M6).

    py -m scout.agent.run "Why did UK inflation rise in 2026?" --user you@example.com

Prints the report as markdown; --json saves the full report (claims, sources, steps,
cost) as JSON. Use --database-url sqlite:///scout_local.db --create-user to try it
without Postgres (web search only: document search needs pgvector).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from scout.agent.report import Report, to_markdown
from scout.agent.runtime import Services, research
from scout.config import get_config, get_secrets
from scout.db import db_is_up, init_db, make_engine
from scout.log import setup_logging
from scout.models import User


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scout.agent.run", description=__doc__.splitlines()[0])
    p.add_argument("question")
    p.add_argument("--user", required=True, help="email of the Scout account to run as")
    p.add_argument("--create-user", action="store_true", help="create the account if missing")
    p.add_argument("--database-url", help="override DATABASE_URL")
    p.add_argument("--no-web", action="store_true", help="don't search the web")
    p.add_argument("--no-docs", action="store_true", help="don't search uploaded documents")
    p.add_argument("--json", type=Path, help="also save the full report as JSON here")
    p.add_argument("--verbose", action="store_true", help="log every model and tool call")
    return p.parse_args(argv)


async def _main(args: argparse.Namespace) -> Report:
    secrets, config = get_secrets(), get_config()
    if secrets.anthropic_api_key is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set (see .env.example).")

    url = args.database_url or secrets.database_url
    engine = make_engine(url)
    if not db_is_up(engine):
        raise SystemExit(
            f"Database not reachable at {url}. Start it with `docker compose up -d db`, "
            "or pass --database-url sqlite:///scout_local.db"
        )
    init_db(engine)
    factory = sessionmaker(engine, expire_on_commit=False)

    email = args.user.strip().lower()
    with factory() as db:
        user_id = db.scalar(select(User.id).where(User.email == email))
        if user_id is None and args.create_user:
            # "!" is not a valid Argon2 hash, so this account can't log in by password.
            user = User(email=email, password_hash="!")
            db.add(user)
            db.commit()
            user_id = user.id
    if user_id is None:
        raise SystemExit(f"No user {email}. Register via POST /auth/register or add --create-user.")

    use_docs = not args.no_docs
    if use_docs and engine.dialect.name != "postgresql":
        print(
            "Note: document search needs Postgres + pgvector; searching the web only.",
            file=sys.stderr,
        )
        use_docs = False

    services = Services.create(config, secrets, factory)

    async def progress(node: str, message: str) -> None:
        print(f"  [{node}] {message}", file=sys.stderr)

    # Last-resort timeout on top of the graph's own time limit, which ends runs gracefully.
    return await asyncio.wait_for(
        research(
            services,
            user_id=user_id,
            job_id=uuid.uuid4(),
            question=args.question,
            use_web=not args.no_web,
            use_docs=use_docs,
            on_event=progress,
        ),
        timeout=config.limits.timeout_s + 60,
    )


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # £ and curly quotes on Windows consoles
    # Logs to stderr so stdout is just the report (pipe-able to a file).
    setup_logging("INFO" if args.verbose else "WARNING", stream=sys.stderr)
    report = asyncio.run(_main(args))
    print(to_markdown(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json}", file=sys.stderr)
    return 1 if report.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
