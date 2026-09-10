"""Postgres connection pool + schema init for the app's structured state.

Everything that must survive a restart lives in Postgres:

* users / decks / sessions / session_decks / turns  — see ``schema.sql``.
* LangGraph checkpoints — owned by ``PostgresSaver`` (its own tables).

We keep a single process-wide ``psycopg_pool.ConnectionPool``. Connections are
configured with ``autocommit=True`` and ``dict_row`` because
``langgraph.checkpoint.postgres.PostgresSaver`` requires both (dict-style row
access, and autocommit for its ``.setup()`` DDL) and we share the same pool with
it. Application queries use short ``with pool.connection() as conn`` blocks.

Configuration is via ``PPT_DATABASE_URL`` (a standard libpq/psycopg conn
string, e.g. ``postgresql://user:pass@localhost:5432/pptagent``). If it is
unset, ``get_pool()`` returns ``None`` and callers fall back to their previous
in-memory / on-disk behaviour, so the app still runs without a database.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

try:  # psycopg / pool are optional until a DB URL is configured.
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    _HAVE_PSYCOPG = True
except Exception:  # pragma: no cover - import guard
    dict_row = None  # type: ignore[assignment]
    ConnectionPool = None  # type: ignore[assignment]
    _HAVE_PSYCOPG = False


SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

_pool: Optional["ConnectionPool"] = None
_pool_lock = threading.Lock()


def database_url() -> str | None:
    url = (os.getenv("PPT_DATABASE_URL") or "").strip()
    return url or None


def _configure(conn) -> None:
    # PostgresSaver.setup() needs autocommit; dict_row for its row access.
    conn.autocommit = True


def get_pool() -> Optional["ConnectionPool"]:
    """Return the process-wide connection pool, or ``None`` if no DB is configured.

    Lazily created on first use. Safe to call from multiple threads.
    """
    global _pool
    if _pool is not None:
        return _pool
    url = database_url()
    if not url:
        return None
    if not _HAVE_PSYCOPG:
        raise RuntimeError(
            "PPT_DATABASE_URL is set but psycopg / psycopg_pool are not installed. "
            "Install them (see requirements.txt)."
        )
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                url,
                min_size=1,
                max_size=10,
                kwargs={"autocommit": True, "row_factory": dict_row},
                configure=_configure,
                open=True,
            )
    return _pool


def db_enabled() -> bool:
    return get_pool() is not None


def init_db() -> None:
    """Create application tables from ``schema.sql`` (idempotent).

    LangGraph's checkpoint tables are created separately by ``PostgresSaver``.
    Intended to be run once (management command / startup), not per request.
    """
    pool = get_pool()
    if pool is None:
        raise RuntimeError("no database configured (set PPT_DATABASE_URL)")
    ddl = SCHEMA_SQL.read_text(encoding="utf-8")
    with pool.connection() as conn:
        conn.execute(ddl)


__all__ = ["get_pool", "db_enabled", "database_url", "init_db", "SCHEMA_SQL"]
