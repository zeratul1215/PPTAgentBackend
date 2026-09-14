"""Structural page tools: delete / add / move pages on the active deck.

These mutate the deck's *ordered page list* (pages.json) — they never renumber
on-disk slots, so a page's artifacts stay valid and a delete is just a drop from
the order (files become harmless orphans).

Page-number semantics for the agent (see the system prompt): the user names
pages by their current 1-based display position. This module snapshot-resolves
those positions to stable slots against the order *at call time*, then mutates,
so a batch like "delete page 5, then recolor page 6" acts on the pages the user
saw, regardless of execution order. When the user explicitly means the
post-change state ("the new page 5 after deleting"), the agent sequences the
tool calls accordingly and each call re-reads the current order.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from agent_backend.agent.tools.context import (
    create_blank_page_artifacts,
    emit,
    page_lock,
    page_order_lock,
    require_project_id,
    sync_deck_page_count,
    workspace_for,
)
from agent_backend.agent.tools.deck_style import require_ready_style
from agent_backend.workspace import pageorder


def _positions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent-facing order view with stable refs and current positions."""
    return [{"page": int(e["position"]), "page_ref": pageorder.page_ref_for_slot(int(e["slot"]))} for e in entries]


@tool
def delete_pages(page_refs: list[str], runtime: ToolRuntime) -> dict[str, Any]:
    """Delete one or more pages from the active deck. THIS IS PERMANENT.

    `page_refs` is a list of stable page refs returned by get_deck_outline or
    locate_pages. Returns the new page count and remaining user-visible pages.
    """
    pid = require_project_id(runtime)
    paths = workspace_for(pid)
    slots = [pageorder.slot_for_page_ref(paths, str(ref)) for ref in page_refs or []]
    missing = [ref for ref, slot in zip(page_refs or [], slots) if slot is None]
    slots = [int(s) for s in slots if s is not None]
    if not slots:
        return {"project_id": pid, "ok": False, "error": "no matching pages to delete", "missing": missing}
    with ExitStack() as stack:
        for slot in sorted(set(slots)):
            stack.enter_context(page_lock(pid, slot))
        with page_order_lock(pid):
            slots = [pageorder.slot_for_page_ref(paths, str(ref)) for ref in page_refs or []]
            missing = [ref for ref, slot in zip(page_refs or [], slots) if slot is None]
            slots = [int(s) for s in slots if s is not None]
            if not slots:
                return {"project_id": pid, "ok": False, "error": "no matching pages to delete", "missing": missing}
            res = pageorder.delete_slots(paths, slots)

    sync_deck_page_count(pid)
    emit(pid, {"type": "pages_changed", "project_id": pid, "op": "delete"})
    return {
        "project_id": pid,
        "ok": True,
        "deleted": len(res["removed"]),
        "missing": missing,
        "page_count": len(res["order"]),
        "revision": res.get("revision"),
        "pages": _positions(res["order"]),
    }


@tool
def add_page(runtime: ToolRuntime, at_position: int | None = None) -> dict[str, Any]:
    """Insert a new blank page into the active deck.

    `at_position` is the 1-based display position the new page should occupy
    (e.g. 3 inserts it as the new page 3, pushing others down). Omit it to append
    at the end. The new page starts blank and editable; describe its content with
    a follow-up `fill_empty_pages` call targeting the returned page_ref.
    """
    pid = require_project_id(runtime)
    paths = workspace_for(pid)
    # Agent-created blank pages are normally followed by style-guided full
    # pipeline content generation, so preflight the project style before
    # mutating page order. Manual frontend blank-page insertion uses the HTTP
    # endpoint and remains available without this gate.
    style_row = require_ready_style(pid, interrupt_when_unready=True)

    from agent_backend.workspace.paths import read_json

    title = (read_json(paths.project_manifest_json()) or {}).get("title") or "PPTAgent"

    with page_order_lock(pid):
        res = pageorder.add_page(
            paths,
            at_position=int(at_position) if at_position is not None else None,
            origin="scratch",
        )
        create_blank_page_artifacts(paths, int(res["slot"]), title=title)

    sync_deck_page_count(pid)
    emit(pid, {"type": "pages_changed", "project_id": pid, "op": "add", "position": res["position"]})
    return {
        "project_id": pid,
        "ok": True,
        "page": res["position"],
        "page_ref": res["page_ref"],
        "revision": res.get("revision"),
        "deck_style_revision": int(style_row.get("revision") or 0),
        "page_count": len(res["order"]),
        "pages": _positions(res["order"]),
    }


@tool
def move_page(page_ref: str, to_page: int, runtime: ToolRuntime) -> dict[str, Any]:
    """Reorder the active deck: move a page to a new position.

    `page_ref` is the stable page to move. `to_page` is the 1-based display
    position it should occupy after the move. Returns the new page order.
    """
    pid = require_project_id(runtime)
    paths = workspace_for(pid)
    with page_order_lock(pid):
        slot = pageorder.slot_for_page_ref(paths, str(page_ref))
        if slot is None:
            return {"project_id": pid, "ok": False, "error": "page_not_found"}
        res = pageorder.move_slot_to_position(paths, slot=int(slot), to_position=int(to_page))
        if not res.get("ok"):
            return {"project_id": pid, "ok": False, "error": res.get("error")}

    emit(pid, {"type": "pages_changed", "project_id": pid, "op": "move"})
    return {"project_id": pid, "ok": True, "revision": res.get("revision"), "pages": _positions(res["order"])}


__all__ = ["delete_pages", "add_page", "move_page"]
