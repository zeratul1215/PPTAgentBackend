"""Shared plumbing for the agent's tools.

This module holds everything the individual tool modules need but that is not
itself a tool: the run-scoped ``AgentContext``, the ``project_id`` accessor, the
SSE progress-publisher hook, per-project locking, and small workspace readers.

Keeping this separate lets the light-weight inspection tools and the heavy
edit/ingest tools import a common core without a circular dependency.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from langchain.tools import ToolRuntime
from typing_extensions import TypedDict

from agent_backend.agent.sessions import STORE
from agent_backend.workspace.paths import read_json, workspace_for


class AgentContext(TypedDict, total=False):
    """Run-scoped context injected at invoke time.

    A session is a chat thread, not a deck. The active deck is resolved *live*
    from the session store (see ``require_project_id``) so a ``set_active_deck``
    call mid-turn takes effect for the rest of that turn.
    """

    session_id: str
    user_id: str
    agent_run_id: str
    current_message_id: str


# ---------------------------------------------------------------------------
# Progress publishing. The server wires this to its SSE _EventBus so chat-driven
# edits surface on the same live-preview channel as before. Kept as a module
# hook to avoid a tools -> server import cycle.
# ---------------------------------------------------------------------------

_progress_publisher: Optional[Callable[[str, dict[str, Any]], None]] = None


def set_progress_publisher(fn: Callable[[str, dict[str, Any]], None] | None) -> None:
    global _progress_publisher
    _progress_publisher = fn


def emit(project_id: str, event: dict[str, Any]) -> None:
    fn = _progress_publisher
    if fn is None:
        return
    try:
        fn(project_id, event)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-project serialization. A pipeline turn mutates shared per-project disk
# state and uses next_turn_dir() with mkdir(exist_ok=False); concurrent turns
# for the same project would race. Guard with a threading.Lock because tools
# may execute off the event loop.
# ---------------------------------------------------------------------------

_project_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def project_lock(project_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _project_locks.get(project_id)
        if lock is None:
            lock = threading.Lock()
            _project_locks[project_id] = lock
        return lock


# ---------------------------------------------------------------------------
# Preview index rebuild lock. `preview/index.html` is rebuilt from ALL of a
# deck's chunk_*.html files (step3 + step4). When several pages of one batch run
# in PARALLEL, each rebuild does read-all-chunks -> render -> atomic write; two
# overlapping rebuilds can lose an update (A reads, B writes, A writes its stale
# copy). Per-page files never collide (different folders), so this is the only
# extra guard parallel editing needs. Keyed per project; held only for the
# in-memory rebuild, so it never blocks the expensive LLM/render work.
# ---------------------------------------------------------------------------

_index_locks: dict[str, threading.Lock] = {}
_index_locks_guard = threading.Lock()


def index_rebuild_lock(project_id: str) -> threading.Lock:
    with _index_locks_guard:
        lock = _index_locks.get(project_id)
        if lock is None:
            lock = threading.Lock()
            _index_locks[project_id] = lock
        return lock


def reread_if_needed(*, pid: str, page: int, slot: int, model: str, paths) -> bool:
    """Backward-compatible wrapper around shared page understanding."""
    from agent_backend.workspace.dirty import is_pending_reread

    if not is_pending_reread(paths, slot):
        return False

    from agent_backend.agent.tools.page_understanding import ensure_page_understanding

    emit(pid, {"type": "reread_started", "page": page, "slot": slot})
    ok = True
    error: str | None = None
    try:
        ensure_page_understanding(
            paths=paths,
            project_id=pid,
            slot=int(slot),
            display_page=int(page),
            model=model,
            focus=[],
            agent_run_id="",
        )
    except Exception as exc:  # noqa: BLE001 - retain marker and report stale input
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    emit(pid, {"type": "reread_finished", "page": page, "slot": slot, "ok": ok, "error": error})
    if not ok:
        raise RuntimeError(error or "reread failed")
    return True


# ---------------------------------------------------------------------------
# Context + workspace helpers shared across tools.
# ---------------------------------------------------------------------------


def require_session_id(runtime: ToolRuntime) -> str:
    ctx = runtime.context or {}
    sid = ctx.get("session_id") if isinstance(ctx, dict) else getattr(ctx, "session_id", None)
    if not sid:
        raise ValueError("no session_id in runtime context")
    return str(sid)


def require_user_id(runtime: ToolRuntime) -> str:
    """User owning the session. Prefer the session store; fall back to context."""
    sid = require_session_id(runtime)
    uid = STORE.user_id(sid)
    if uid:
        return uid
    ctx = runtime.context or {}
    uid = ctx.get("user_id") if isinstance(ctx, dict) else getattr(ctx, "user_id", None)
    if not uid:
        raise ValueError("no user_id for session")
    return str(uid)


def require_agent_run_id(runtime: ToolRuntime) -> str:
    ctx = runtime.context or {}
    rid = ctx.get("agent_run_id") if isinstance(ctx, dict) else getattr(ctx, "agent_run_id", None)
    return str(rid or "run_unknown")


def require_project_id(runtime: ToolRuntime) -> str:
    """Live active deck for the current session.

    Resolved from the session store on every call, so switching decks mid-turn
    (via ``set_active_deck``) is visible to the next tool immediately.
    """
    sid = require_session_id(runtime)
    pid = STORE.active_project_id(sid)
    if not pid:
        raise ValueError(
            "no active deck for this session. Ask the user which deck to work on, "
            "or use list_decks + set_active_deck first."
        )
    return str(pid)


def page_count(paths) -> int:
    """Number of pages in the deck's current display order.

    Reads the ordered page list (``pages.json``), which is the source of truth
    once add/delete/reorder are in play. Falls back to the manifest scalar for
    freshly bootstrapped decks (pageorder synthesizes + persists from it).
    """
    from agent_backend.workspace.pageorder import page_count as _pc

    return int(_pc(paths))


def load_page_state(paths, page_num: int) -> dict[str, Any] | None:
    """Return a lightweight page view without invoking Step1.

    Read-only tools always project the authoritative PPTist JSON. They do not
    consume Full-Pipeline ``current_page_state.json`` because that state may be
    pending refresh after a Patch or manual edit.
    """
    slide_path = paths.pptist_slide_json(int(page_num))
    if not slide_path.exists():
        return None
    try:
        slide = read_json(slide_path)
        if not (isinstance(slide, dict) and isinstance(slide.get("elements"), list)):
            return None
        from agent_backend.agent.tools.html_to_pptist import slide_to_plan_page

        plan_page = slide_to_plan_page(slide, page_id=f"page{int(page_num) - 1}")
        return _normalize_step1_input(
            {
                "page_num": int(page_num),
                "page_size_pt": None,
                "plan_page": plan_page,
            }
        )
    except Exception:
        return None


def _normalize_step1_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a bootstrap step1-input object onto the page-state shape.

    Mirrors ``_iter_text_items_from_plan_page`` in the step1 module so an
    un-edited page reads the same way an edited one does. This is a lossy,
    read-only view (no segments / palette enrichment); it exists purely so the
    agent can see titles and text previews before a page is edited.
    """
    if not isinstance(raw, dict):
        return {"texts": [], "images": []}
    plan_page = raw.get("plan_page") if isinstance(raw.get("plan_page"), dict) else {}

    texts: list[dict[str, Any]] = []
    next_id = 0

    def _emit(kind: str, text: str) -> None:
        nonlocal next_id
        t = (text or "").strip()
        if not t:
            return
        texts.append(
            {
                "id": f"t{next_id}",
                "kind": str(kind or "body"),
                "text": t,
                "segments": [t],
            }
        )
        next_id += 1

    def _blocks(blk: dict[str, Any]):
        kind = str(blk.get("kind") or "body")
        if kind == "bullets":
            for it in blk.get("items") or []:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    yield ("bullet_item", str(it.get("text") or ""))
        else:
            yield (kind, str(blk.get("text") or ""))

    for blk in plan_page.get("blocks") or []:
        if not isinstance(blk, dict):
            continue
        for kind, txt in _blocks(blk):
            _emit(kind, txt)

    images: list[dict[str, Any]] = []
    for im in plan_page.get("images") or []:
        if not isinstance(im, dict):
            continue
        src = im.get("src")
        safe_ref = "embedded_image" if isinstance(src, str) and src.startswith("data:image/") else src
        images.append(
            {
                "id": im.get("id"),
                "ref": safe_ref,
                "src": safe_ref,
                "description_en": im.get("description_en"),
            }
        )

    return {
        "schema_version": "step1_input_normalized_v1",
        "page_num": raw.get("page_num"),
        "page_size_pt": raw.get("page_size_pt"),
        "texts": texts,
        "images": images,
        "original_layout_description_en": plan_page.get("layout_notes") or "",
    }


def create_blank_page_artifacts(paths, slot: int, *, title: str = "PPTAgent") -> None:
    import base64

    from agent_backend.workspace.paths import write_json as _wj
    from agent_backend.workspace.assets import materialize_pptist_slide_assets

    slide = {
        "elements": [],
        "background": {"type": "solid", "color": "#ffffff"},
    }
    _wj(paths.pptist_slide_json(int(slot)), slide)
    materialize_pptist_slide_assets(paths, int(slot), slide)
    # A minimal white PNG is enough until the frontend renders and uploads the
    # real page image after the user edits the new slide.
    blank_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mP4"
        "/58BAAT/Af9jgNErAAAAAElFTkSuQmCC"
    )
    png_path = paths.reread_page_png(int(slot))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not png_path.exists():
        png_path.write_bytes(blank_png)

def sync_deck_page_count(project_id: str) -> None:
    """Refresh the deck's cached page_count (project.json + DB row) from the
    ordered page list after a structural change (add/delete), so deck listings
    stay accurate. Best-effort; never raises."""
    from agent_backend.workspace.paths import write_json as _wj
    from agent_backend.workspace.pageorder import page_count as _pc

    try:
        paths = workspace_for(project_id)
        n = int(_pc(paths))
        mf = paths.project_manifest_json()
        if not mf.exists():
            return
        obj = read_json(mf)
        if not isinstance(obj, dict):
            return
        obj["page_count"] = n
        _wj(mf, obj)
        uid = obj.get("user_id")
        if uid:
            from agent_backend.workspace.repo import upsert_deck

            upsert_deck(
                project_id=project_id,
                user_id=str(uid),
                title=obj.get("title") or project_id,
                page_count=n,
                page_size_pt=obj.get("page_size_pt"),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[pages] page_count sync failed: {type(exc).__name__}: {exc}")


def is_page_processed(paths, page_num: int) -> bool:
    """True once a page has authoritative PPTist JSON."""
    return paths.pptist_slide_json(page_num).exists()


def title_of(state: dict[str, Any]) -> str:
    for t in state.get("texts") or []:
        if t.get("kind") in ("title", "subheading"):
            txt = (t.get("text") or "").strip()
            if txt:
                return txt
    for t in state.get("texts") or []:
        txt = (t.get("text") or "").strip()
        if txt:
            return txt
    return ""


def preview_of(state: dict[str, Any], limit: int = 240) -> str:
    parts: list[str] = []
    for t in state.get("texts") or []:
        txt = (t.get("text") or "").strip()
        if txt:
            parts.append(txt)
    blob = " \u2022 ".join(parts)
    return blob[:limit] + ("\u2026" if len(blob) > limit else "")


def outline(paths) -> list[dict[str, Any]]:
    """Per-page outline in current display order.

    ``page`` is the 1-based display *position* the user/agent sees; ``slot`` is
    the stable on-disk key used to fetch chunk/state/png (never renumbered).
    Disk readers take a slot; the position is derived from the ordered list.
    """
    from agent_backend.workspace.pageorder import ordered_entries

    out: list[dict[str, Any]] = []
    for entry in ordered_entries(paths):
        position = int(entry["position"])
        slot = int(entry["slot"])
        origin = str(entry.get("origin") or "pdf")
        processed = is_page_processed(paths, slot)
        state = load_page_state(paths, slot)
        base = {
            "page": position,
            "slot": slot,
            "page_ref": f"page@{slot}",
            "origin": origin,
            "source_pdf_index": entry.get("source_pdf_index"),
            "processed": processed,
        }
        if state is None:
            out.append({**base, "title": "", "preview": "", "loaded": False})
            continue
        out.append(
            {
                **base,
                "title": title_of(state),
                "preview": preview_of(state),
                "num_texts": len(state.get("texts") or []),
                "num_images": len(state.get("images") or []),
                "loaded": True,
            }
        )
    return out


__all__ = [
    "AgentContext",
    "set_progress_publisher",
    "emit",
    "project_lock",
    "index_rebuild_lock",
    "reread_if_needed",
    "require_session_id",
    "require_user_id",
    "require_agent_run_id",
    "require_project_id",
    "page_count",
    "load_page_state",
    "title_of",
    "preview_of",
    "outline",
    "is_page_processed",
    "workspace_for",
]
