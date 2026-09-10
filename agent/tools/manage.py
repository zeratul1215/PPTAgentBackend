"""Deck-management tools: list the user's decks and switch the active one.

These make the agent multi-deck aware. A session is a chat thread that can move
between decks; ``set_active_deck`` writes the shared session store, so the very
next tool call in the same turn operates on the newly selected deck.
"""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from agent_backend.agent.sessions import STORE
from agent_backend.agent.tools.context import (
    outline,
    require_session_id,
    require_user_id,
    title_of,
    load_page_state,
)
from agent_backend.workspace.paths import list_projects, read_json, workspace_for
from agent_backend.workspace.repo import list_decks_by_user


def deck_summaries(user_id: str) -> list[dict[str, Any]]:
    """Metadata for every deck owned by ``user_id`` (newest first).

    Reads from the database when one is configured; otherwise falls back to
    scanning ``result/*/project.json`` on disk.
    """
    db_rows = list_decks_by_user(user_id)
    if db_rows is not None:
        return db_rows

    out: list[dict[str, Any]] = []
    for paths in list_projects():
        mf = paths.project_manifest_json()
        if not mf.exists():
            continue
        try:
            manifest = read_json(mf)
        except Exception:
            continue
        if str(manifest.get("user_id") or "") != str(user_id):
            continue
        out.append(
            {
                "project_id": paths.project_id,
                "title": manifest.get("title") or paths.project_id,
                "page_count": int(manifest.get("page_count") or 0),
            }
        )
    return out


def _resolve_deck_ref(*, decks: list[dict[str, Any]], deck_ref: str) -> dict[str, Any] | None:
    """Resolve a deck reference (id, title, or fuzzy description) to one deck."""
    ref = (deck_ref or "").strip()
    if not ref:
        return None
    # 1) Exact project_id.
    for d in decks:
        if d["project_id"] == ref:
            return d
    low = ref.lower()
    # 2) Exact / substring title match.
    for d in decks:
        if (d.get("title") or "").lower() == low:
            return d
    subs = [d for d in decks if low in (d.get("title") or "").lower()]
    if len(subs) == 1:
        return subs[0]
    # 3) Fuzzy: token overlap against title + first-page content.
    tokens = [t for t in low.replace(",", " ").split() if t]
    if not tokens:
        return None
    best: tuple[int, dict[str, Any]] | None = None
    for d in decks:
        paths = workspace_for(d["project_id"])
        hay = (d.get("title") or "").lower()
        state = load_page_state(paths, 1)
        if state:
            hay += " " + title_of(state).lower()
        score = sum(hay.count(t) for t in tokens)
        if score > 0 and (best is None or score > best[0]):
            best = (score, d)
    return best[1] if best else None


@tool
def list_decks(runtime: ToolRuntime) -> dict[str, Any]:
    """List every deck the current user has, with the active one flagged.

    Use this to see what the user can work on, to resolve which deck a request
    refers to, or before switching decks. ``project_id`` is an opaque handle;
    refer to decks by title when talking to the user.
    """
    uid = require_user_id(runtime)
    sid = require_session_id(runtime)
    active = STORE.active_project_id(sid)
    decks = deck_summaries(uid)
    for d in decks:
        d["active"] = d["project_id"] == active
    return {"user_id": uid, "active_project_id": active, "decks": decks}


@tool
def set_active_deck(deck_ref: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Switch the deck this session is editing.

    `deck_ref` may be a deck's exact id, its title, or a description ("the
    revenue deck", "the one about onboarding"). After switching, subsequent
    tools (get_deck_outline, edit_pages, ...) operate on the new deck. If the
    reference is ambiguous or matches nothing, this returns the candidate list
    so you can ask the user which one they mean.
    """
    uid = require_user_id(runtime)
    sid = require_session_id(runtime)
    decks = deck_summaries(uid)
    if not decks:
        return {"ok": False, "error": "this user has no decks yet", "decks": []}

    match = _resolve_deck_ref(decks=decks, deck_ref=deck_ref)
    if match is None:
        return {
            "ok": False,
            "error": f"could not resolve deck reference: {deck_ref!r}",
            "decks": decks,
        }

    STORE.set_active(session_id=sid, project_id=match["project_id"])
    outline_pages = outline(workspace_for(match["project_id"]))
    return {
        "ok": True,
        "active_project_id": match["project_id"],
        "title": match.get("title"),
        "page_count": len(outline_pages),
    }


__all__ = ["list_decks", "set_active_deck", "deck_summaries"]
