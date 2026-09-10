"""Heavy edit tool: run the single-page pipeline for one or more pages.

This is the agent's workhorse. Each edit runs the vendored LangGraph pipeline
(`run_pipeline_once`) which mutates shared per-project disk state, so turns are
serialized per project and progress is published onto the SSE channel.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from typing_extensions import NotRequired, TypedDict

from agent_backend.agent.models import agent_model_name
from agent_backend.agent.tools.context import (
    emit,
    project_lock,
    require_project_id,
    require_session_id,
    workspace_for,
)
from agent_backend.agent.tools.page_understanding import ensure_page_understanding
from agent_backend.agent.tools.deck_style import public_style_row, require_ready_style
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.graph import (
    run_pipeline_once,
)
from agent_backend.workspace import pageorder
from agent_backend.workspace.repo import record_turn


# Cap concurrent page pipelines per batch. Each page runs its own step1-4 (LLM
# calls + a Playwright QA render), so this bounds peak memory / browser count and
# upstream API concurrency. Override via env for tuning.
try:
    MAX_PARALLEL_PAGES = max(1, int(os.environ.get("PPT_MAX_PARALLEL_PAGES", "3")))
except ValueError:
    MAX_PARALLEL_PAGES = 3


class PageEdit(TypedDict):
    page_ref: str
    demand: str
    use_deck_style: NotRequired[bool]


@tool
def edit_pages(edits: list[PageEdit], runtime: ToolRuntime) -> dict[str, Any]:
    """Apply edits to one or more pages by running the single-page pipeline.

    `edits` is a list of `{"page_ref": "page@<id>", "demand": "<natural-language
    instruction for that page>"}`. Each demand should be a complete, standalone
    instruction (e.g. "translate the body to bilingual, English left / Chinese
    right" or "make the title font larger and use a blue background"). Edits run
    sequentially and are applied directly (no confirmation). Returns per-page
    success, the turn directory, and any errors.
    """
    pid = require_project_id(runtime)
    try:
        sid = require_session_id(runtime)
    except Exception:
        sid = None
    model = agent_model_name()
    lock = project_lock(pid)
    paths = workspace_for(pid)

    parsed: list[dict[str, Any]] = []
    for e in edits or []:
        slot = pageorder.slot_for_page_ref(paths, str(e.get("page_ref") or ""))
        if slot is None:
            continue
        demand = str(e.get("demand") or "").strip()
        if not demand:
            continue
        parsed.append({
            "page": int(pageorder.position_for_slot(paths, slot) or 0),
            "slot": int(slot),
            "demand": demand,
            "use_deck_style": bool(e.get("use_deck_style")),
        })

    if not parsed:
        return {"project_id": pid, "ok": False, "error": "no valid edits (each needs page + non-empty demand)"}

    style_needed = any(bool(item.get("use_deck_style")) for item in parsed)
    deck_style_row = require_ready_style(pid, interrupt_when_unready=True) if style_needed else None

    emit(pid, {"type": "batch_started", "project_id": pid, "count": len(parsed)})

    def _run_one(i: int, page: int, demand: str, slot: int | None, use_deck_style: bool) -> dict[str, Any]:
        """Run one page's pipeline end-to-end. Safe to call from a worker thread:
        each page writes only its own per-page files; the sole shared write
        (preview/index.html) is serialized by index_rebuild_lock inside step3/4.
        """
        if slot is None:
            err = f"page {page} does not exist in this deck"
            emit(pid, {"type": "task_failed", "page": page, "slot": None, "demand": demand, "index": i, "error": err})
            return {"page": page, "ok": False, "error": err}

        emit(pid, {"type": "task_started", "page": page, "slot": slot, "demand": demand, "index": i})
        try:
            understanding = ensure_page_understanding(
                paths=paths,
                project_id=pid,
                slot=int(slot),
                display_page=page,
                model=model,
                focus=[],
                agent_run_id="",
            )
            core = understanding.get("core")
            understanding_status = str(understanding.get("_status") or "cached")
            if not isinstance(core, dict):
                raise RuntimeError("page understanding did not produce core state")

            final_state = run_pipeline_once(
                project_id=pid,
                page_num=slot,
                demand=demand,
                model=model,
                display_page=page,
                batch_index=i,
                current_page_state=core,
                understanding_status=understanding_status,
                deck_style=deck_style_row.get("style_json") if use_deck_style and isinstance(deck_style_row, dict) else None,
                deck_style_revision=int(deck_style_row.get("revision") or 0) if use_deck_style and isinstance(deck_style_row, dict) else 0,
            )
            turn_dir = final_state.get("turn_dir")
            # Turn success is decided SOLELY by whether step3 (reassemble)
            # produced a result. Step3 is the step that writes the chunk the user
            # sees; every later step (step4 QA/autoshrink, commit) is polish over
            # an already-committed preview, and the pre-step3 beautify_image is a
            # non-critical enhancement that degrades gracefully. If step3 raised,
            # the pipeline would have thrown before returning here (caught below),
            # so reaching this point with a step3_result means we display it and
            # never ask the agent to re-run. This deliberately ignores any
            # non-fatal `errors` the pipeline may have accumulated, so a transient
            # beautify/QA hiccup never provokes a destructive full re-plan (which
            # can lose one-shot state such as a consumed pending-uploads manifest).
            ok = bool(final_state.get("step3_result"))
            record_turn(
                project_id=pid,
                session_id=sid,
                page_num=slot,
                demand=demand,
                ok=ok,
                error=None,
                turn_dir=str(turn_dir) if turn_dir else None,
            )
            emit(
                pid,
                {
                    "type": "task_finished",
                    "page": page,
                    "slot": slot,
                    "demand": demand,
                    "index": i,
                    "turn_dir": turn_dir,
                    "errors": [],
                    "prepare_reread_png": False,
                },
            )
            if ok:
                from agent_backend.workspace.dirty import mark_pending_reread

                mark_pending_reread(paths, slot)
            return {"page": page, "ok": ok, "turn_dir": turn_dir, "errors": []}
        except Exception as exc:  # noqa: BLE001 - surface to the agent
            err = f"{type(exc).__name__}: {exc}"
            record_turn(
                project_id=pid,
                session_id=sid,
                page_num=slot,
                demand=demand,
                ok=False,
                error=err,
            )
            emit(pid, {"type": "task_failed", "page": page, "slot": slot, "demand": demand, "index": i, "error": err})
            return {"page": page, "ok": False, "error": err}

    # Pages of one batch target DIFFERENT slots (different folders), so they can
    # run concurrently. We hold the project lock for the WHOLE batch so structural
    # ops (add/delete/reorder/sync) still can't interleave a running edit, then
    # fan the pages out onto a small pool. Results are reassembled in input order.
    results_by_index: dict[int, dict[str, Any]] = {}
    max_workers = min(MAX_PARALLEL_PAGES, len(parsed))
    with lock:
        if style_needed:
            latest = public_style_row(pid)
            if latest.get("status") != "ready" or not isinstance(latest.get("style_json"), dict):
                raise RuntimeError("deck style became unavailable before editing")
            deck_style_row = latest
        if max_workers <= 1:
            for i, item in enumerate(parsed):
                results_by_index[i] = _run_one(i, item["page"], item["demand"], item["slot"], bool(item.get("use_deck_style")))
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="edit") as pool:
                futures = {
                    pool.submit(_run_one, i, item["page"], item["demand"], item["slot"], bool(item.get("use_deck_style"))): i
                    for i, item in enumerate(parsed)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results_by_index[idx] = fut.result()

    results = [results_by_index[i] for i in range(len(parsed))]
    emit(pid, {"type": "batch_finished", "project_id": pid})

    ok = all(r.get("ok") for r in results)
    return {"project_id": pid, "ok": ok, "results": results}


__all__ = ["edit_pages", "PageEdit"]
