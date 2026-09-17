"""Wipe all persistent agent state so the system can start clean.

The following state can survive a restart and is handled by this command:

1. Postgres (``PPT_DATABASE_URL``)
   * application tables: ``users`` / ``decks`` / ``sessions`` /
     ``session_decks`` / ``turns`` / ``deck_styles`` /
     ``deck_style_presets`` / ``chat_messages`` / ``agent_runs`` /
     ``conversation_summaries`` / ``session_states`` /
     ``session_resources`` / ``session_resource_files`` /
     ``session_resource_mentions`` (see ``workspace/schema.sql``).
   * LangGraph checkpoint tables owned by ``PostgresSaver``
     (``checkpoints`` / ``checkpoint_writes`` / ``checkpoint_blobs`` /
     ``checkpoint_migrations``) — these hold the multi-turn chat memory.

2. On-disk workspaces under ``agent_backend/result/<project_id>/`` — the
   uploaded PDF/PPTX plus every derived artifact (page PNGs, baseline +
   preview HTML, per-page state, turn history).
3. Python bytecode caches (``__pycache__/``) under ``agent_backend/``.

This script truncates the Postgres data and deletes the on-disk workspaces and
Python caches, giving a fresh system. It is destructive and irreversible, so
it prompts for confirmation unless ``--yes`` is passed.

Usage
-----

    # wipe DB rows + on-disk workspaces + Python caches (asks to confirm):
    python -m agent_backend.server.reset_state

    # skip the prompt:
    python -m agent_backend.server.reset_state --yes

    # only wipe the database, keep uploaded files on disk:
    python -m agent_backend.server.reset_state --db-only --yes

    # only delete on-disk workspaces + Python caches, keep the database:
    python -m agent_backend.server.reset_state --files-only --yes

    # preview what would happen, change nothing:
    python -m agent_backend.server.reset_state --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from agent_backend.workspace.db import database_url, get_pool
from agent_backend.workspace.env import load_dotenv
from agent_backend.workspace.paths import DEFAULT_RESULTS_ROOT, DEFAULT_SESSION_RESOURCES_ROOT

_AGENT_BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Load agent_backend/.env so PPT_DATABASE_URL etc. are available even when the
# process was started without exporting them by hand.
load_dotenv()


# Application tables, ordered so FK-dependent children are listed first. We use
# TRUNCATE ... CASCADE anyway, but keeping the order documents the graph.
_APP_TABLES = (
    "agent_run_events",
    "agent_runs",
    "session_resource_mentions",
    "session_resource_files",
    "session_resources",
    "session_states",
    "conversation_summaries",
    "chat_messages",
    "turns",
    "session_decks",
    "sessions",
    "deck_styles",
    "deck_style_presets",
    "decks",
    "users",
)

# LangGraph PostgresSaver's tables. It recreates them on the next .setup(), so
# truncating (or dropping) them just clears the chat memory.
_CHECKPOINT_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)


def _existing_tables(conn, candidates: tuple[str, ...]) -> list[str]:
    """Return the subset of ``candidates`` that actually exist in the DB."""
    rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = ANY(%s)
        """,
        (list(candidates),),
    ).fetchall()
    present = {(r["table_name"] if isinstance(r, dict) else r[0]) for r in rows}
    # Preserve the caller's order.
    return [t for t in candidates if t in present]


def reset_database(*, dry_run: bool) -> bool:
    """Truncate app + checkpoint tables. Returns True if a DB was touched."""
    url = database_url()
    if not url:
        print("[reset] no PPT_DATABASE_URL configured; skipping database.")
        return False

    pool = get_pool()
    if pool is None:  # pragma: no cover - defensive
        print("[reset] database URL set but pool unavailable; skipping database.")
        return False

    print(f"[reset] database: {url}")
    with pool.connection() as conn:
        tables = _existing_tables(conn, _APP_TABLES + _CHECKPOINT_TABLES)
        if not tables:
            print("[reset] no known tables present; nothing to truncate.")
            return True
        for t in tables:
            print(f"[reset]   - {t}")
        if dry_run:
            print("[reset] dry-run: no rows deleted.")
            return True
        # RESTART IDENTITY resets BIGSERIAL counters (e.g. turns.turn_id).
        # CASCADE also clears any table with a FK into these.
        idents = ", ".join(tables)
        conn.execute(f"TRUNCATE {idents} RESTART IDENTITY CASCADE")
        print(f"[reset] truncated {len(tables)} table(s).")
    return True


def reset_files(*, dry_run: bool) -> None:
    """Delete workspaces, session resources, and Python bytecode caches."""
    root = DEFAULT_RESULTS_ROOT
    if root.exists():
        projects = [p for p in root.iterdir() if p.is_dir()]
        print(f"[reset] workspaces under {root}: {len(projects)} project(s)")
        for p in projects:
            print(f"[reset]   - {p.name}")
        if dry_run:
            print("[reset] dry-run: no files deleted.")
        else:
            for p in projects:
                shutil.rmtree(p, ignore_errors=True)
            print(f"[reset] deleted {len(projects)} workspace(s).")
    else:
        print(f"[reset] no workspace dir at {root}; skipping files.")

    resource_root = DEFAULT_SESSION_RESOURCES_ROOT
    if resource_root.exists():
        print(f"[reset] session resources under {resource_root}")
        if dry_run:
            print("[reset] dry-run: session resources preserved.")
        else:
            shutil.rmtree(resource_root, ignore_errors=True)
            print("[reset] deleted session resource root.")

    cache_dirs = [
        p for p in _AGENT_BACKEND_ROOT.rglob("__pycache__") if p.is_dir()
    ]
    print(
        f"[reset] Python caches under {_AGENT_BACKEND_ROOT}: "
        f"{len(cache_dirs)} director(y/ies)"
    )
    if dry_run:
        for p in cache_dirs:
            print(f"[reset]   - {p.relative_to(_AGENT_BACKEND_ROOT)}")
        print("[reset] dry-run: Python caches preserved.")
    else:
        for p in cache_dirs:
            shutil.rmtree(p, ignore_errors=True)
        print(f"[reset] deleted {len(cache_dirs)} Python cache director(y/ies).")


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agent_backend.server.reset_state",
        description="Wipe persistent agent state (Postgres rows + on-disk workspaces).",
    )
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--db-only", action="store_true", help="Only wipe the database.")
    scope.add_argument("--files-only", action="store_true", help="Only delete on-disk workspaces.")
    ap.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen, change nothing.")
    args = ap.parse_args(argv)

    do_db = not args.files_only
    do_files = not args.db_only

    targets = []
    if do_db:
        targets.append("Postgres data (users/decks/sessions/turns/deck styles/style templates + chat checkpoints)")
    if do_files:
        targets.append(
            f"on-disk workspaces under {DEFAULT_RESULTS_ROOT}, session resources under "
            f"{DEFAULT_SESSION_RESOURCES_ROOT}, and Python caches under {_AGENT_BACKEND_ROOT}"
        )
    print("[reset] this will permanently delete:")
    for t in targets:
        print(f"[reset]   * {t}")

    if not args.dry_run and not args.yes:
        if not _confirm("[reset] proceed?"):
            print("[reset] aborted.")
            return 1

    try:
        if do_db:
            reset_database(dry_run=args.dry_run)
        if do_files:
            reset_files(dry_run=args.dry_run)
    finally:
        # Close the shared pool so its worker threads shut down cleanly on exit.
        pool = get_pool()
        if pool is not None:
            pool.close()

    print("[reset] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
