import threading
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from scout.db import init_db, make_engine
from scout.models import User


def test_only_in_memory_sqlite_shares_one_connection(tmp_path: Path) -> None:
    assert isinstance(make_engine("sqlite://").pool, StaticPool)
    assert not isinstance(make_engine(f"sqlite:///{tmp_path / 'a.db'}").pool, StaticPool)


def test_file_sqlite_handles_parallel_writes(tmp_path: Path) -> None:
    # Regression: file DBs used to share one connection, and parallel tool calls
    # (e.g. the reader saving notes) failed with "cannot start a transaction".
    engine = make_engine(f"sqlite:///{(tmp_path / 'b.db').as_posix()}")
    init_db(engine)
    factory = sessionmaker(engine)
    errors: list[Exception] = []

    def write(i: int) -> None:
        try:
            with factory() as db:
                db.add(User(email=f"u{i}@example.com", password_hash="x"))
                db.commit()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(User)) == 8
