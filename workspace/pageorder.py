"""Ordered page list: decouple a deck's *display order* from *disk layout*.

Why this exists
---------------
Historically a "page number" meant everything at once: the 1-based display
position, the on-disk directory name (``page_003``), the turn-local HTML chunk
name, and the baseline PNG index. That made add / delete / reorder impossible
without renaming a pile of files.

This module introduces three separate concepts:

* ``page_id`` — a stable, never-reused identity for a page (audit / references).
* ``slot``    — the stable on-disk key. ``WorkspacePaths.*(slot)`` addresses the
  page's directory and baseline PNG. **Slots are never renumbered**, so a page's
  files stay valid for the life of the deck. Deleting a page just drops it from
  the order and leaves its files as harmless orphans.
* ``position`` — the 1-based number the user and agent see. It is *derived*:
  ``position = index in the order list + 1``. Nothing stores it.

The ordered list lives in ``pages.json`` at the workspace root and is the single
source of truth for "what pages this deck has and in what order". Everything
that used to iterate ``range(1, n+1)`` or ``sorted(glob(chunk_*))`` should read
the order here instead.

``source_pdf_index`` is a legacy field name kept for frontend compatibility. It
now records the 0-based baseline PNG source index, not a persisted source PDF.
Scratch pages (created from zero) have ``source_pdf_index = None``.
"""

from __future__ import annotations

from typing import Any, Optional

from .paths import WorkspacePaths, read_json, write_json


SCHEMA_VERSION = "pageorder_v2"


# ---------------------------------------------------------------------------
# Load / synthesize / save
# ---------------------------------------------------------------------------


def _pages_json(paths: WorkspacePaths):
    return paths.root / "pages.json"


def _manifest_page_count(paths: WorkspacePaths) -> int:
    """Read the scalar page_count from whichever manifest has it.

    Kept local (rather than importing agent.tools.context.page_count) so this
    module has no dependency on the agent layer and cannot recurse.
    """
    for mf in (paths.project_manifest_json(), paths.baseline_manifest_json):
        try:
            if mf.exists():
                obj = read_json(mf)
                if isinstance(obj, dict) and obj.get("page_count") is not None:
                    return int(obj.get("page_count") or 0)
        except Exception:
            continue
    return 0


def _synthesize_from_manifest(paths: WorkspacePaths) -> dict[str, Any]:
    """Build a default order for decks bootstrapped before pages.json existed.

    Old workspaces are 1:1 positional: slot i == position i == source page i-1.
    """
    n = _manifest_page_count(paths)
    order = [
        {"page_id": i, "slot": i, "source_pdf_index": i - 1, "origin": "pdf"}
        for i in range(1, n + 1)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 1,
        "next_page_id": n + 1,
        "next_slot": n + 1,
        "order": order,
    }


def load_pages(paths: WorkspacePaths) -> dict[str, Any]:
    """Return the deck's page-order document, synthesizing it if absent.

    Synthesized docs are persisted so the deck has a stable order from then on.
    """
    p = _pages_json(paths)
    if p.exists():
        try:
            obj = read_json(p)
            if isinstance(obj, dict) and isinstance(obj.get("order"), list):
                return _normalize(obj)
        except Exception:
            pass
    obj = _synthesize_from_manifest(paths)
    try:
        write_json(p, obj)
    except Exception:
        pass
    return obj


def _normalize(obj: dict[str, Any]) -> dict[str, Any]:
    order = []
    for e in obj.get("order") or []:
        if not isinstance(e, dict):
            continue
        try:
            slot = int(e["slot"])
            page_id = int(e["page_id"])
        except Exception:
            continue
        spi = e.get("source_pdf_index")
        order.append(
            {
                "page_id": page_id,
                "slot": slot,
                "source_pdf_index": int(spi) if spi is not None else None,
                "origin": str(e.get("origin") or "pdf"),
            }
        )
    max_id = max((e["page_id"] for e in order), default=0)
    max_slot = max((e["slot"] for e in order), default=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": int(obj.get("revision") or 1),
        "next_page_id": int(obj.get("next_page_id") or (max_id + 1)),
        "next_slot": int(obj.get("next_slot") or (max_slot + 1)),
        "order": order,
    }


def save_pages(paths: WorkspacePaths, obj: dict[str, Any]) -> None:
    write_json(_pages_json(paths), _normalize(obj))


def _bump_revision(doc: dict[str, Any]) -> None:
    doc["revision"] = int(doc.get("revision") or 1) + 1


def revision(paths: WorkspacePaths) -> int:
    return int(load_pages(paths).get("revision") or 1)


def write_initial_order(paths: WorkspacePaths, *, page_count: int) -> None:
    """Called by bootstrap: the initial order is 1:1 with baseline PNGs."""
    order = [
        {"page_id": i, "slot": i, "source_pdf_index": i - 1, "origin": "pdf"}
        for i in range(1, int(page_count) + 1)
    ]
    save_pages(
        paths,
        {
            "schema_version": SCHEMA_VERSION,
            "revision": 1,
            "next_page_id": int(page_count) + 1,
            "next_slot": int(page_count) + 1,
            "order": order,
        },
    )


# ---------------------------------------------------------------------------
# Read helpers (position <-> slot <-> page_id)
# ---------------------------------------------------------------------------


def ordered_entries(paths: WorkspacePaths) -> list[dict[str, Any]]:
    """The pages in display order. Each entry gains a derived 1-based ``position``."""
    order = load_pages(paths)["order"]
    return [{**e, "position": i + 1} for i, e in enumerate(order)]


def page_ref_for_slot(slot: int) -> str:
    return f"page@{int(slot)}"


def slot_for_page_ref(paths: WorkspacePaths, page_ref: str) -> Optional[int]:
    text = str(page_ref or "").strip()
    if not text.startswith("page@"):
        return None
    try:
        slot = int(text.split("@", 1)[1])
    except Exception:
        return None
    return slot if entry_for_slot(paths, slot) is not None else None


def page_count(paths: WorkspacePaths) -> int:
    return len(load_pages(paths)["order"])


def entry_at_position(paths: WorkspacePaths, position: int) -> Optional[dict[str, Any]]:
    entries = ordered_entries(paths)
    if 1 <= int(position) <= len(entries):
        return entries[int(position) - 1]
    return None


def slot_for_position(paths: WorkspacePaths, position: int) -> Optional[int]:
    """Map a user-facing display position to its stable on-disk slot."""
    e = entry_at_position(paths, position)
    return int(e["slot"]) if e else None


def position_for_slot(paths: WorkspacePaths, slot: int) -> Optional[int]:
    for e in ordered_entries(paths):
        if int(e["slot"]) == int(slot):
            return int(e["position"])
    return None


def entry_for_slot(paths: WorkspacePaths, slot: int) -> Optional[dict[str, Any]]:
    for e in ordered_entries(paths):
        if int(e["slot"]) == int(slot):
            return e
    return None


def resolve_positions_to_slots(
    paths: WorkspacePaths, positions: list[int]
) -> list[tuple[int, Optional[int]]]:
    """Snapshot-resolve a batch of positions to slots against the CURRENT order.

    Returns ``[(position, slot_or_None), ...]``. This is the primitive that makes
    a multi-target request order-independent: resolve every position the user
    named to a stable slot *before* any structural change executes, so later
    edits/deletes act on the pages the user meant, not on shifted positions.
    """
    entries = ordered_entries(paths)
    by_pos = {e["position"]: int(e["slot"]) for e in entries}
    return [(int(p), by_pos.get(int(p))) for p in positions]


# ---------------------------------------------------------------------------
# Structural mutations. Callers hold the per-project lock. All return the new
# ordered_entries() so callers can report the fresh layout.
# ---------------------------------------------------------------------------


def delete_slots(paths: WorkspacePaths, slots: list[int]) -> dict[str, Any]:
    """Remove pages (by slot) from the order. Files are left as orphans on disk.

    Returns ``{"removed": [slots...], "order": <new entries>}``.
    """
    doc = load_pages(paths)
    want = {int(s) for s in slots}
    removed = [e["slot"] for e in doc["order"] if e["slot"] in want]
    doc["order"] = [e for e in doc["order"] if e["slot"] not in want]
    if removed:
        _bump_revision(doc)
    save_pages(paths, doc)
    return {"removed": removed, "order": ordered_entries(paths), "revision": revision(paths)}


def move_slot_to_position(
    paths: WorkspacePaths, *, slot: int, to_position: int
) -> dict[str, Any]:
    """Move one page (by slot) so it lands at 1-based ``to_position``."""
    doc = load_pages(paths)
    order = doc["order"]
    idx = next((i for i, e in enumerate(order) if int(e["slot"]) == int(slot)), None)
    if idx is None:
        return {"ok": False, "error": f"unknown slot {slot}", "order": ordered_entries(paths)}
    entry = order.pop(idx)
    dest = max(1, min(int(to_position), len(order) + 1)) - 1
    order.insert(dest, entry)
    doc["order"] = order
    _bump_revision(doc)
    save_pages(paths, doc)
    return {"ok": True, "order": ordered_entries(paths), "revision": revision(paths)}


def reorder_by_slots(paths: WorkspacePaths, slots: list[int]) -> dict[str, Any]:
    """Set the whole order from a full permutation of the deck's slots.

    Any slots omitted from ``slots`` keep their relative order at the end (so a
    partial list can't accidentally drop pages).
    """
    doc = load_pages(paths)
    by_slot = {int(e["slot"]): e for e in doc["order"]}
    seen: set[int] = set()
    new_order: list[dict[str, Any]] = []
    for s in slots:
        s = int(s)
        if s in by_slot and s not in seen:
            new_order.append(by_slot[s])
            seen.add(s)
    for e in doc["order"]:
        if int(e["slot"]) not in seen:
            new_order.append(e)
    changed = [int(e["slot"]) for e in doc["order"]] != [int(e["slot"]) for e in new_order]
    doc["order"] = new_order
    if changed:
        _bump_revision(doc)
    save_pages(paths, doc)
    return {"ok": True, "order": ordered_entries(paths), "revision": revision(paths)}


def reorder_by_slots_with_revision(
    paths: WorkspacePaths, *, slots: list[int], base_revision: int | None = None
) -> dict[str, Any]:
    doc = load_pages(paths)
    current = int(doc.get("revision") or 1)
    if base_revision is not None and int(base_revision) != current:
        return {
            "ok": False,
            "error": "revision_conflict",
            "expected_revision": current,
            "order": ordered_entries(paths),
            "revision": current,
        }
    return reorder_by_slots(paths, slots)


def add_page(
    paths: WorkspacePaths,
    *,
    at_position: Optional[int] = None,
    origin: str = "scratch",
    source_pdf_index: Optional[int] = None,
) -> dict[str, Any]:
    """Allocate a fresh page_id + slot and insert it at ``at_position``.

    ``at_position`` is 1-based; ``None`` (or out of range) appends to the end.
    The new page has no disk artifacts yet — the caller (create-page pipeline)
    is responsible for writing ``pptist_slide.json`` / ``current_page_state.json``.

    Returns ``{"page_id", "slot", "position", "order"}``.
    """
    doc = load_pages(paths)
    page_id = int(doc["next_page_id"])
    slot = int(doc["next_slot"])
    doc["next_page_id"] = page_id + 1
    doc["next_slot"] = slot + 1
    entry = {
        "page_id": page_id,
        "slot": slot,
        "source_pdf_index": int(source_pdf_index) if source_pdf_index is not None else None,
        "origin": str(origin or "scratch"),
    }
    order = doc["order"]
    if at_position is None or int(at_position) < 1 or int(at_position) > len(order) + 1:
        order.append(entry)
    else:
        order.insert(int(at_position) - 1, entry)
    doc["order"] = order
    _bump_revision(doc)
    save_pages(paths, doc)
    return {
        "page_id": page_id,
        "slot": slot,
        "page_ref": page_ref_for_slot(slot),
        "position": next(
            (i + 1 for i, e in enumerate(order) if e["slot"] == slot), len(order)
        ),
        "order": ordered_entries(paths),
        "revision": revision(paths),
    }


__all__ = [
    "SCHEMA_VERSION",
    "load_pages",
    "save_pages",
    "revision",
    "write_initial_order",
    "ordered_entries",
    "page_count",
    "entry_at_position",
    "slot_for_position",
    "position_for_slot",
    "entry_for_slot",
    "resolve_positions_to_slots",
    "page_ref_for_slot",
    "slot_for_page_ref",
    "delete_slots",
    "move_slot_to_position",
    "reorder_by_slots",
    "reorder_by_slots_with_revision",
    "add_page",
]
