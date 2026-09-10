"""Data-access helpers for the application tables (users / decks / turns).

Thin functions over the shared connection pool in ``db.py``. Session state has
its own richer store (``agent/sessions.py``); this module covers the deck
library, user rows, and edit-history writes.

Every function is a no-op / returns ``None`` when no database is configured
(``PPT_DATABASE_URL`` unset), so the app keeps working off-disk without a DB.
Callers should treat a ``None`` return as "DB disabled, fall back".
"""

from __future__ import annotations

import uuid
import json
from pathlib import Path
from typing import Any, Optional

from agent_backend.workspace.db import get_pool


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def ensure_user(user_id: str, *, username: str | None = None, display_name: str | None = None) -> None:
    """Insert the user row if absent. No-op without a DB."""
    pool = get_pool()
    if pool is None:
        return
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, username, display_name)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id, username or user_id, display_name),
        )


def login_user(username: str) -> Optional[dict[str, Any]]:
    """Resolve a username to an account, creating it on first login.

    Passwordless: the same (case-insensitive) username always maps to the same
    ``user_id``. Returns ``{user_id, username, display_name}``, or ``None`` when
    no database is configured (caller should fall back to a local identity).
    """
    pool = get_pool()
    if pool is None:
        return None
    handle = (username or "").strip()
    if not handle:
        raise ValueError("username is required")
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT user_id, username, display_name FROM users WHERE lower(username) = lower(%s)",
            (handle,),
        ).fetchone()
        if row is None:
            new_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO users (user_id, username, display_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (new_id, handle, handle),
            )
            row = conn.execute(
                "SELECT user_id, username, display_name FROM users WHERE lower(username) = lower(%s)",
                (handle,),
            ).fetchone()
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "display_name": row["display_name"] or row["username"],
    }


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------


def upsert_deck(
    *,
    project_id: str,
    user_id: str,
    title: str,
    page_count: int,
    page_size_pt: dict[str, Any] | None = None,
    source_kind: str | None = None,
    source_path: str | None = None,
    workspace_root: str | None = None,
) -> None:
    """Create/update a deck row. Ensures the owning user row exists first."""
    pool = get_pool()
    if pool is None:
        return
    import json

    ensure_user(user_id)
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO decks (project_id, user_id, title, page_count, page_size_pt,
                               source_kind, source_path, workspace_root)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (project_id) DO UPDATE SET
                title = EXCLUDED.title,
                page_count = EXCLUDED.page_count,
                page_size_pt = EXCLUDED.page_size_pt,
                source_kind = EXCLUDED.source_kind,
                source_path = EXCLUDED.source_path,
                workspace_root = EXCLUDED.workspace_root,
                updated_at = now()
            """,
            (
                project_id,
                user_id,
                title,
                int(page_count),
                json.dumps(page_size_pt) if page_size_pt is not None else None,
                source_kind,
                source_path,
                workspace_root,
            ),
        )


def list_decks_by_user(user_id: str) -> Optional[list[dict[str, Any]]]:
    """Deck summaries for a user (newest first), or ``None`` when DB disabled."""
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT project_id, title, page_count
            FROM decks
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "project_id": r["project_id"],
            "title": r["title"] or r["project_id"],
            "page_count": int(r["page_count"] or 0),
        }
        for r in rows
    ]


def get_deck(project_id: str) -> Optional[dict[str, Any]]:
    """Full deck row as a dict, or ``None`` (unknown deck or DB disabled)."""
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM decks WHERE project_id = %s",
            (project_id,),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Sessions (chat threads) — listing + title. The live session *state*
# (active deck, recents) is owned by agent/sessions.py; these are the
# read/label helpers the frontend's session sidebar needs.
# ---------------------------------------------------------------------------


def list_sessions_by_user(user_id: str) -> Optional[list[dict[str, Any]]]:
    """Chat sessions for a user (most-recently-active first), or ``None``."""
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, title, active_project_id, created_at, last_active_at
            FROM sessions
            WHERE user_id = %s
            ORDER BY last_active_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "title": r["title"],
            "active_project_id": r["active_project_id"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "last_active_at": r["last_active_at"].isoformat() if r["last_active_at"] else None,
        }
        for r in rows
    ]


def get_session_title(session_id: str) -> Optional[str]:
    """Current title for a session (``None`` if unset or DB disabled)."""
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT title FROM sessions WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return (row["title"] if row else None) or None


def set_session_title(session_id: str, title: str, *, only_if_empty: bool = True) -> None:
    """Label a session. With ``only_if_empty`` (default) the first message wins
    and later messages don't overwrite the title. No-op without a DB."""
    pool = get_pool()
    if pool is None:
        return
    title = (title or "").strip()
    if not title:
        return
    with pool.connection() as conn:
        if only_if_empty:
            conn.execute(
                "UPDATE sessions SET title = %s WHERE session_id = %s AND (title IS NULL OR title = '')",
                (title, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET title = %s WHERE session_id = %s",
                (title, session_id),
            )


# ---------------------------------------------------------------------------
# Product chat messages, runs, events, summaries, and artifacts
# ---------------------------------------------------------------------------


def active_run_for_session(session_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM agent_runs
            WHERE session_id = %s AND status IN ('queued', 'running', 'waiting')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def create_chat_run(
    *,
    session_id: str,
    user_id: str,
    content: str,
    client_message_id: str,
    active_project_id: str | None,
    selected_slot: int | None,
    page_order_revision: int | None,
    artifact_refs: list[str],
) -> dict[str, Any]:
    pool = get_pool()
    if pool is None:
        raise RuntimeError("chat runs require a database")
    user_message_id = uuid.uuid4().hex
    assistant_message_id = uuid.uuid4().hex
    agent_run_id = f"run_{uuid.uuid4().hex[:12]}"
    attachments_json = json.dumps([{"artifact_ref": r} for r in artifact_refs], ensure_ascii=False)
    with pool.connection() as conn:
        existing = conn.execute(
            """
            SELECT r.*
            FROM chat_messages m
            JOIN agent_runs r ON r.user_message_id = m.message_id
            WHERE m.session_id = %s AND m.client_message_id = %s AND m.role = 'user'
            LIMIT 1
            """,
            (session_id, client_message_id),
        ).fetchone()
        if existing:
            return dict(existing)
        active = conn.execute(
            """
            SELECT agent_run_id
            FROM agent_runs
            WHERE session_id = %s AND status IN ('queued', 'running', 'waiting')
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if active:
            raise RuntimeError(f"session_has_active_run:{active['agent_run_id']}")
        conn.execute(
            """
            INSERT INTO chat_messages (
                message_id, session_id, user_id, role, content, status,
                client_message_id, active_project_id, selected_slot,
                page_order_revision, attachments
            )
            VALUES (%s, %s, %s, 'user', %s, 'complete', %s, %s, %s, %s, %s::jsonb)
            """,
            (
                user_message_id,
                session_id,
                user_id,
                content,
                client_message_id,
                active_project_id,
                selected_slot,
                page_order_revision,
                attachments_json,
            ),
        )
        conn.execute(
            """
            INSERT INTO chat_messages (
                message_id, session_id, user_id, role, content, status,
                active_project_id, selected_slot, page_order_revision
            )
            VALUES (%s, %s, %s, 'assistant', '', 'streaming', %s, %s, %s)
            """,
            (
                assistant_message_id,
                session_id,
                user_id,
                active_project_id,
                selected_slot,
                page_order_revision,
            ),
        )
        row = conn.execute(
            """
            INSERT INTO agent_runs (
                agent_run_id, session_id, user_id, user_message_id, assistant_message_id,
                active_project_id, selected_slot, page_order_revision, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'queued')
            RETURNING *
            """,
            (
                agent_run_id,
                session_id,
                user_id,
                user_message_id,
                assistant_message_id,
                active_project_id,
                selected_slot,
                page_order_revision,
            ),
        ).fetchone()
        if artifact_refs:
            conn.execute(
                """
                UPDATE artifacts
                SET message_id = %s, status = 'attached', updated_at = now()
                WHERE session_id = %s AND user_id = %s AND artifact_ref = ANY(%s)
                """,
                (user_message_id, session_id, user_id, artifact_refs),
            )
        conn.execute(
            "UPDATE sessions SET last_active_at = now() WHERE session_id = %s",
            (session_id,),
        )
    return dict(row)


def update_agent_run(
    agent_run_id: str,
    *,
    status: str | None = None,
    todos: list[dict[str, Any]] | None = None,
    gate: dict[str, Any] | None = None,
    clear_gate: bool = False,
    error: str | None = None,
    active_project_id: str | None = None,
) -> None:
    pool = get_pool()
    if pool is None:
        return
    sets = ["updated_at = now()"]
    params: list[Any] = []
    if status is not None:
        sets.append("status = %s")
        params.append(status)
        if status in {"complete", "failed", "cancelled"}:
            sets.append("completed_at = now()")
    if todos is not None:
        sets.append("todos = %s::jsonb")
        params.append(json.dumps(todos, ensure_ascii=False))
    if gate is not None:
        sets.append("gate = %s::jsonb")
        params.append(json.dumps(gate, ensure_ascii=False))
    elif clear_gate or status not in (None, "waiting"):
        sets.append("gate = NULL")
    if error is not None:
        sets.append("error = %s")
        params.append(error)
    if active_project_id is not None:
        sets.append("active_project_id = %s")
        params.append(active_project_id)
    params.append(agent_run_id)
    with pool.connection() as conn:
        conn.execute(
            f"UPDATE agent_runs SET {', '.join(sets)} WHERE agent_run_id = %s",
            params,
        )


def get_agent_run(agent_run_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute("SELECT * FROM agent_runs WHERE agent_run_id = %s", (agent_run_id,)).fetchone()
    return dict(row) if row else None


def append_agent_event(
    *,
    agent_run_id: str,
    session_id: str,
    event: dict[str, Any],
) -> Optional[int]:
    pool = get_pool()
    if pool is None:
        return None
    ev = dict(event)
    typ = str(ev.get("type") or "event")
    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO agent_run_events (agent_run_id, session_id, type, payload)
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING event_id
            """,
            (agent_run_id, session_id, typ, json.dumps(ev, ensure_ascii=False)),
        ).fetchone()
    return int(row["event_id"]) if row else None


def list_agent_events(agent_run_id: str, *, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT event_id, payload
            FROM agent_run_events
            WHERE agent_run_id = %s AND event_id > %s
            ORDER BY event_id ASC
            LIMIT %s
            """,
            (agent_run_id, int(after), int(limit)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row["payload"] or {})
        payload["cursor"] = int(row["event_id"])
        out.append(payload)
    return out


def append_assistant_message(agent_run_id: str, text: str, *, final: bool = False) -> None:
    pool = get_pool()
    if pool is None:
        return
    run = get_agent_run(agent_run_id)
    if not run:
        return
    with pool.connection() as conn:
        if final:
            conn.execute(
                """
                UPDATE chat_messages
                SET content = %s, status = 'complete', updated_at = now()
                WHERE message_id = %s
                """,
                (text, run["assistant_message_id"]),
            )
        else:
            conn.execute(
                """
                UPDATE chat_messages
                SET content = content || %s, updated_at = now()
                WHERE message_id = %s
                """,
                (text, run["assistant_message_id"]),
            )


def list_chat_messages(session_id: str) -> list[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT message_id, role, content, status, attachments, created_at, seq
            FROM chat_messages
            WHERE session_id = %s AND role IN ('user', 'assistant')
            ORDER BY seq ASC
            """,
            (session_id,),
        ).fetchall()
    return [
        {
            "message_id": r["message_id"],
            "role": r["role"],
            "content": r["content"] or "",
            "status": r["status"],
            "attachments": r["attachments"] or [],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "seq": int(r["seq"] or 0),
        }
        for r in rows
        if (r["content"] or r["attachments"] or r["status"] == "streaming")
    ]


def get_conversation_summary(session_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversation_summaries WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def get_session_state(session_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM session_states WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def upsert_conversation_summary(
    *,
    session_id: str,
    user_id: str,
    covered_seq: int,
    summary: str,
    version: int = 1,
) -> None:
    pool = get_pool()
    if pool is None:
        return
    ensure_user(user_id)
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO conversation_summaries (session_id, user_id, covered_seq, version, summary)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                covered_seq = EXCLUDED.covered_seq,
                version = EXCLUDED.version,
                summary = EXCLUDED.summary,
                updated_at = now()
            """,
            (session_id, user_id, int(covered_seq), int(version), summary),
        )


def _validate_state_entries(entries: list[dict[str, Any]], *, session_message_ids: set[str]) -> list[dict[str, Any]]:
    allowed_kind = {"goal", "decision", "fact", "constraint", "open_question", "reference"}
    allowed_basis = {"user_stated", "system_observed", "assistant_inferred"}
    allowed_status = {"active", "resolved", "superseded"}
    out: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or "").strip()[:96]
        if not key or not all(part.replace("_", "").isalnum() for part in key.split(".")):
            continue
        kind = str(raw.get("kind") or "").strip()
        basis = str(raw.get("basis") or "").strip()
        status = str(raw.get("status") or "").strip()
        if kind not in allowed_kind or basis not in allowed_basis or status not in allowed_status:
            continue
        src = [
            str(x)
            for x in (raw.get("source_message_ids") or [])
            if str(x) in session_message_ids
        ][:12]
        out.append(
            {
                "key": key,
                "kind": kind,
                "value": raw.get("value"),
                "basis": basis,
                "source_message_ids": src,
                "status": status,
            }
        )
    return out


def commit_context_compaction(
    *,
    session_id: str,
    user_id: str,
    expected_summary_version: int,
    expected_state_revision: int,
    expected_covered_seq: int,
    new_summary: str,
    state_delta: dict[str, Any],
    new_covered_seq: int,
) -> bool:
    """Atomically advance conversation summary and session state.

    Returns False when a concurrent compaction already committed newer data.
    """
    pool = get_pool()
    if pool is None:
        return False
    ensure_user(user_id)
    deletes = {str(x or "").strip() for x in (state_delta.get("deletes") or []) if str(x or "").strip()}
    upserts_raw = [x for x in (state_delta.get("upserts") or []) if isinstance(x, dict)]
    upsert_keys = {str(x.get("key") or "").strip() for x in upserts_raw}
    if deletes & upsert_keys:
        return False

    with pool.connection() as conn:
        with conn.transaction():
            conn.execute("SELECT session_id FROM sessions WHERE session_id = %s FOR UPDATE", (session_id,))
            msg_rows = conn.execute(
                "SELECT message_id FROM chat_messages WHERE session_id = %s",
                (session_id,),
            ).fetchall()
            session_message_ids = {str(r["message_id"]) for r in msg_rows}
            upserts = _validate_state_entries(upserts_raw, session_message_ids=session_message_ids)

            summary_row = conn.execute(
                "SELECT version, covered_seq FROM conversation_summaries WHERE session_id = %s FOR UPDATE",
                (session_id,),
            ).fetchone()
            state_row = conn.execute(
                "SELECT revision, updated_through_seq, entries FROM session_states WHERE session_id = %s FOR UPDATE",
                (session_id,),
            ).fetchone()
            cur_summary_version = int(summary_row["version"]) if summary_row else 0
            cur_summary_covered = int(summary_row["covered_seq"]) if summary_row else int(expected_covered_seq)
            cur_state_revision = int(state_row["revision"]) if state_row else 0
            cur_state_covered = int(state_row["updated_through_seq"]) if state_row else int(expected_covered_seq)
            if (
                cur_summary_version != int(expected_summary_version)
                or cur_state_revision != int(expected_state_revision)
                or cur_summary_covered != int(expected_covered_seq)
                or cur_state_covered != int(expected_covered_seq)
            ):
                return False

            existing_entries = state_row["entries"] if state_row and isinstance(state_row["entries"], list) else []
            by_key = {
                str(e.get("key")): dict(e)
                for e in existing_entries
                if isinstance(e, dict) and str(e.get("key") or "")
            }
            for key in deletes:
                by_key.pop(key, None)
            for entry in upserts:
                by_key[str(entry["key"])] = entry
            entries = list(by_key.values())
            if len(entries) > 64:
                entries.sort(
                    key=lambda e: (
                        0 if str(e.get("status") or "") in {"resolved", "superseded"} else 1,
                        str(e.get("key") or ""),
                    )
                )
                entries = entries[-64:]

            conn.execute(
                """
                INSERT INTO conversation_summaries (session_id, user_id, covered_seq, version, summary)
                VALUES (%s, %s, %s, 1, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    covered_seq = EXCLUDED.covered_seq,
                    version = conversation_summaries.version + 1,
                    summary = EXCLUDED.summary,
                    updated_at = now()
                """,
                (session_id, user_id, int(new_covered_seq), new_summary),
            )
            conn.execute(
                """
                INSERT INTO session_states (session_id, user_id, revision, updated_through_seq, entries)
                VALUES (%s, %s, 1, %s, %s::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    revision = session_states.revision + 1,
                    updated_through_seq = EXCLUDED.updated_through_seq,
                    entries = EXCLUDED.entries,
                    updated_at = now()
                """,
                (session_id, user_id, int(new_covered_seq), json.dumps(entries, ensure_ascii=False)),
            )
    return True


def create_artifact(
    *,
    artifact_ref: str,
    user_id: str,
    session_id: str,
    filename: str,
    mime: str,
    size_bytes: int,
    width: int | None,
    height: int | None,
    sha256: str,
    storage_path: str,
    thumbnail_path: str | None = None,
) -> dict[str, Any]:
    pool = get_pool()
    if pool is None:
        raise RuntimeError("artifacts require a database")
    ensure_user(user_id)
    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO artifacts (
                artifact_ref, user_id, session_id, filename, mime, size_bytes,
                width, height, sha256, storage_path, thumbnail_path, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft')
            RETURNING *
            """,
            (
                artifact_ref,
                user_id,
                session_id,
                filename,
                mime,
                int(size_bytes),
                width,
                height,
                sha256,
                storage_path,
                thumbnail_path,
            ),
        ).fetchone()
    return dict(row)


def get_artifact(*, artifact_ref: str, session_id: str | None = None, user_id: str | None = None) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    clauses = ["artifact_ref = %s"]
    params: list[Any] = [artifact_ref]
    if session_id is not None:
        clauses.append("session_id = %s")
        params.append(session_id)
    if user_id is not None:
        clauses.append("user_id = %s")
        params.append(user_id)
    with pool.connection() as conn:
        row = conn.execute(f"SELECT * FROM artifacts WHERE {' AND '.join(clauses)}", params).fetchone()
    return dict(row) if row else None


def list_artifacts_for_message(message_id: str) -> list[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE message_id = %s AND status <> 'deleted' ORDER BY created_at ASC",
            (message_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_artifact_deleted(*, artifact_ref: str, session_id: str, user_id: str) -> bool:
    pool = get_pool()
    if pool is None:
        return False
    with pool.connection() as conn:
        cur = conn.execute(
            """
            UPDATE artifacts SET status = 'deleted', updated_at = now()
            WHERE artifact_ref = %s AND session_id = %s AND user_id = %s AND status = 'draft'
            """,
            (artifact_ref, session_id, user_id),
        )
    return bool(cur.rowcount)


def cleanup_draft_artifacts(*, older_than_hours: int = 24) -> list[str]:
    pool = get_pool()
    if pool is None:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            """
            UPDATE artifacts
            SET status = 'deleted', updated_at = now()
            WHERE status = 'draft' AND created_at < now() - (%s || ' hours')::interval
            RETURNING storage_path, thumbnail_path
            """,
            (int(older_than_hours),),
        ).fetchall()
    paths: list[str] = []
    for r in rows:
        for key in ("storage_path", "thumbnail_path"):
            value = r.get(key)
            if value:
                paths.append(str(value))
    return paths


def mark_stale_running_runs_failed() -> None:
    pool = get_pool()
    if pool is None:
        return
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE agent_runs
            SET status = 'failed',
                error = COALESCE(error, 'server restarted while run was active'),
                updated_at = now(),
                completed_at = now()
            WHERE status IN ('queued', 'running')
            """
        )
# ---------------------------------------------------------------------------
# Turns (single-page edit history)
# ---------------------------------------------------------------------------


def record_turn(
    *,
    project_id: str,
    page_num: int,
    demand: str,
    ok: bool,
    session_id: str | None = None,
    error: str | None = None,
    turn_dir: str | None = None,
) -> Optional[int]:
    """Insert a turn row; returns its id, or ``None`` when DB disabled.

    Best-effort: history logging must never break an edit, so failures here are
    swallowed (the edit itself already succeeded/failed on disk).
    """
    pool = get_pool()
    if pool is None:
        return None
    try:
        with pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO turns (project_id, session_id, page_num, demand, ok, error, turn_dir)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING turn_id
                """,
                (project_id, session_id, int(page_num), demand, bool(ok), error, turn_dir),
            ).fetchone()
        return int(row["turn_id"]) if row else None
    except Exception:
        return None


def get_deck_style(project_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM deck_styles WHERE project_id = %s",
            (project_id,),
        ).fetchone()
    return dict(row) if row else None


def upsert_deck_style(
    *,
    project_id: str,
    status: str,
    style_json: dict[str, Any] | None = None,
    source: str | None = None,
    analysis_run_id: str | None = None,
    error: str | None = None,
    bump_revision: bool = False,
    expected_analysis_run_id: str | None = None,
    expected_status: str | None = None,
) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    style_s = json.dumps(style_json, ensure_ascii=False) if style_json is not None else None
    with pool.connection() as conn:
        if expected_analysis_run_id is not None or expected_status is not None:
            clauses = ["project_id = %s"]
            params: list[Any] = [project_id]
            if expected_analysis_run_id is not None:
                clauses.append("analysis_run_id = %s")
                params.append(expected_analysis_run_id)
            if expected_status is not None:
                clauses.append("status = %s")
                params.append(expected_status)
            row = conn.execute(
                f"""
                UPDATE deck_styles SET
                    status = %s,
                    style_json = %s::jsonb,
                    source = %s,
                    analysis_run_id = %s,
                    error = %s,
                    revision = CASE WHEN %s THEN revision + 1 ELSE revision END,
                    updated_at = now()
                WHERE {' AND '.join(clauses)}
                RETURNING *
                """,
                [status, style_s, source, analysis_run_id, error, bool(bump_revision), *params],
            ).fetchone()
            return dict(row) if row else None
        row = conn.execute(
            """
            INSERT INTO deck_styles (project_id, status, style_json, source, analysis_run_id, error)
            VALUES (%s, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (project_id) DO UPDATE SET
                status = EXCLUDED.status,
                style_json = EXCLUDED.style_json,
                source = EXCLUDED.source,
                analysis_run_id = EXCLUDED.analysis_run_id,
                error = EXCLUDED.error,
                revision = CASE WHEN %s THEN deck_styles.revision + 1 ELSE deck_styles.revision END,
                updated_at = now()
            RETURNING *
            """,
            (project_id, status, style_s, source, analysis_run_id, error, bool(bump_revision)),
        ).fetchone()
    return dict(row) if row else None


def mark_stale_deck_style_analyses_failed() -> None:
    pool = get_pool()
    if pool is None:
        return
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE deck_styles
            SET status = 'failed',
                error = COALESCE(error, 'server restarted while style analysis was running'),
                analysis_run_id = NULL,
                updated_at = now()
            WHERE status = 'analyzing'
            """
        )


def list_deck_style_presets(user_id: str) -> Optional[list[dict[str, Any]]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT preset_id, user_id, name, style_json, created_at, updated_at
            FROM deck_style_presets
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def find_deck_style_preset_by_name(
    *, user_id: str, name: str, exclude_preset_id: str | None = None
) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    clean = (name or "").strip()
    if not clean:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM deck_style_presets
            WHERE user_id = %s
              AND lower(btrim(name)) = lower(btrim(%s))
              AND (%s IS NULL OR preset_id <> %s)
            LIMIT 1
            """,
            (user_id, clean, exclude_preset_id, exclude_preset_id),
        ).fetchone()
    return dict(row) if row else None


def upsert_deck_style_preset(
    *, preset_id: str | None, user_id: str, name: str, style_json: dict[str, Any]
) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    ensure_user(user_id)
    pid = preset_id or uuid.uuid4().hex
    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO deck_style_presets (preset_id, user_id, name, style_json)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (preset_id) DO UPDATE SET
                name = EXCLUDED.name,
                style_json = EXCLUDED.style_json,
                updated_at = now()
            WHERE deck_style_presets.user_id = EXCLUDED.user_id
            RETURNING *
            """,
            (pid, user_id, name, json.dumps(style_json, ensure_ascii=False)),
        ).fetchone()
    return dict(row) if row else None


def get_deck_style_preset(*, user_id: str, preset_id: str) -> Optional[dict[str, Any]]:
    pool = get_pool()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM deck_style_presets WHERE user_id = %s AND preset_id = %s",
            (user_id, preset_id),
        ).fetchone()
    return dict(row) if row else None


def delete_deck_style_preset(*, user_id: str, preset_id: str) -> bool:
    pool = get_pool()
    if pool is None:
        return False
    with pool.connection() as conn:
        cur = conn.execute(
            "DELETE FROM deck_style_presets WHERE user_id = %s AND preset_id = %s",
            (user_id, preset_id),
        )
    return bool(cur.rowcount)


__all__ = [
    "ensure_user",
    "login_user",
    "upsert_deck",
    "list_decks_by_user",
    "get_deck",
    "list_sessions_by_user",
    "get_session_title",
    "set_session_title",
    "active_run_for_session",
    "create_chat_run",
    "update_agent_run",
    "get_agent_run",
    "append_agent_event",
    "list_agent_events",
    "append_assistant_message",
    "list_chat_messages",
    "get_conversation_summary",
    "get_session_state",
    "upsert_conversation_summary",
    "commit_context_compaction",
    "create_artifact",
    "get_artifact",
    "list_artifacts_for_message",
    "mark_artifact_deleted",
    "cleanup_draft_artifacts",
    "mark_stale_running_runs_failed",
    "record_turn",
    "get_deck_style",
    "upsert_deck_style",
    "mark_stale_deck_style_analyses_failed",
    "list_deck_style_presets",
    "find_deck_style_preset_by_name",
    "upsert_deck_style_preset",
    "get_deck_style_preset",
    "delete_deck_style_preset",
]
