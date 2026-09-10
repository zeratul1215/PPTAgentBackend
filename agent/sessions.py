"""Session store for the conversational agent (DB-backed, memory fallback).

A *session* is one chat thread. It is not bound to a single deck: the user can
switch which deck (project) they are editing mid-conversation, either by
clicking in the UI or by asking in natural language. The store holds, per
``session_id``:

* ``user_id``            — who owns the session (decks are filtered by this).
* ``active_project_id``  — the deck the next tool call will operate on.
* ``recent_project_ids`` — decks touched in this session, most-recent-first.

Tools never receive ``project_id`` as an argument. They read the *live* active
deck from this store via ``session_id`` (injected through runtime context), so a
``set_active_deck`` call takes effect immediately for the rest of the same turn.

Persistence: when ``PPT_DATABASE_URL`` is configured, session state lives in the
``sessions`` / ``session_decks`` tables and survives restarts (mirroring the
LangGraph ``PostgresSaver`` that persists the conversation itself). Without a
DB, this transparently falls back to a process-local, thread-safe in-memory
store — same public API, lost on restart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from agent_backend.workspace.db import get_pool


@dataclass
class SessionState:
    session_id: str
    user_id: str
    active_project_id: Optional[str] = None
    # Decks touched in this session, most-recent-first (for "the previous deck").
    recent_project_ids: list[str] = field(default_factory=list)


class _MemoryBackend:
    """Process-local store; used when no database is configured."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            st = self._sessions.get(session_id)
            # Return a copy so callers can't mutate our state by reference.
            return _copy(st) if st else None

    def ensure(self, *, session_id: str, user_id: str) -> SessionState:
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                st = SessionState(session_id=session_id, user_id=user_id)
                self._sessions[session_id] = st
            return _copy(st)

    def set_active(self, *, session_id: str, project_id: str) -> None:
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                raise KeyError(f"unknown session: {session_id}")
            st.active_project_id = project_id
            st.recent_project_ids = [project_id] + [
                p for p in st.recent_project_ids if p != project_id
            ]


class _PostgresBackend:
    """Durable store backed by the ``sessions`` / ``session_decks`` tables."""

    def __init__(self, pool) -> None:
        self._pool = pool

    def _recent(self, conn, session_id: str) -> list[str]:
        rows = conn.execute(
            """
            SELECT project_id FROM session_decks
            WHERE session_id = %s
            ORDER BY last_used_at DESC
            """,
            (session_id,),
        ).fetchall()
        return [r["project_id"] for r in rows]

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT session_id, user_id, active_project_id FROM sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            recent = self._recent(conn, session_id)
        return SessionState(
            session_id=row["session_id"],
            user_id=row["user_id"],
            active_project_id=row["active_project_id"],
            recent_project_ids=recent,
        )

    def ensure(self, *, session_id: str, user_id: str) -> SessionState:
        with self._pool.connection() as conn:
            # Owning user must exist for the FK. Normally the row already
            # exists (created at login); this is a safety net. username is
            # NOT NULL, so seed it from user_id when we have nothing better.
            conn.execute(
                """
                INSERT INTO users (user_id, username) VALUES (%s, %s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (user_id, user_id),
            )
            conn.execute(
                """
                INSERT INTO sessions (session_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()
                """,
                (session_id, user_id),
            )
            row = conn.execute(
                "SELECT session_id, user_id, active_project_id FROM sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            recent = self._recent(conn, session_id)
        return SessionState(
            session_id=row["session_id"],
            user_id=row["user_id"],
            active_project_id=row["active_project_id"],
            recent_project_ids=recent,
        )

    def set_active(self, *, session_id: str, project_id: str) -> None:
        with self._pool.connection() as conn:
            updated = conn.execute(
                """
                UPDATE sessions
                SET active_project_id = %s, last_active_at = now()
                WHERE session_id = %s
                RETURNING session_id
                """,
                (project_id, session_id),
            ).fetchone()
            if updated is None:
                raise KeyError(f"unknown session: {session_id}")
            conn.execute(
                """
                INSERT INTO session_decks (session_id, project_id, last_used_at)
                VALUES (%s, %s, now())
                ON CONFLICT (session_id, project_id) DO UPDATE SET last_used_at = now()
                """,
                (session_id, project_id),
            )


def _copy(st: SessionState) -> SessionState:
    return SessionState(
        session_id=st.session_id,
        user_id=st.user_id,
        active_project_id=st.active_project_id,
        recent_project_ids=list(st.recent_project_ids),
    )


class SessionStore:
    """Public API used by the server and agent tools.

    Backend is chosen lazily per call: if a DB is configured we go to Postgres,
    otherwise to the in-memory backend. Choosing per call (rather than once at
    construction) keeps a single module-level singleton correct even if the DB
    URL becomes available after import.
    """

    def __init__(self) -> None:
        self._memory = _MemoryBackend()

    def _backend(self):
        pool = get_pool()
        if pool is not None:
            return _PostgresBackend(pool)
        return self._memory

    def create(self, *, session_id: str, user_id: str) -> SessionState:
        return self._backend().ensure(session_id=session_id, user_id=user_id)

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._backend().get(session_id)

    def ensure(self, *, session_id: str, user_id: str) -> SessionState:
        """Return the session, creating it if absent.

        If it exists with a different user_id, the stored one wins (the caller
        should treat session_id as authoritative for ownership).
        """
        return self._backend().ensure(session_id=session_id, user_id=user_id)

    def set_active(self, *, session_id: str, project_id: str) -> None:
        self._backend().set_active(session_id=session_id, project_id=project_id)

    def active_project_id(self, session_id: str) -> Optional[str]:
        st = self._backend().get(session_id)
        return st.active_project_id if st else None

    def user_id(self, session_id: str) -> Optional[str]:
        st = self._backend().get(session_id)
        return st.user_id if st else None


# Process-wide singleton. The server and the agent tools share this instance so
# a manual UI switch (server) and a natural-language switch (tool) write the
# same source of truth.
STORE = SessionStore()


__all__ = ["SessionState", "SessionStore", "STORE"]
