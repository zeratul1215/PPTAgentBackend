"""Shared page understanding service and Agent tool.

This replaces the old split between Full Pipeline Step1 and reread. The
persisted file is a small wrapper whose ``core`` field is exactly the historical
``understand_output_v1`` object consumed by Step2.
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from typing_extensions import NotRequired, TypedDict

from agent_backend.agent.models import agent_model_name
from agent_backend.agent.tools.context import (
    emit,
    project_lock,
    require_agent_run_id,
    require_project_id,
    workspace_for,
)
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.reread import (
    _load_slide_for_reread,
    _materialize_images,
)
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.step_1_understand.step1_understand_recompose_mvp import (
    _call_claude_json,
    _resolve_backend,
)
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.steps import run_step1
from agent_backend.workspace import pageorder
from agent_backend.workspace.assets import materialize_pptist_slide_assets, page_asset_source_dir
from agent_backend.workspace.dirty import clear_pending_reread, is_pending_reread
from agent_backend.workspace.paths import WorkspacePaths, read_json, write_json


try:
    MAX_PARALLEL_UNDERSTAND = max(1, int(os.environ.get("PPT_MAX_PARALLEL_UNDERSTAND", "3")))
except ValueError:
    MAX_PARALLEL_UNDERSTAND = 3


class PageUnderstandRequest(TypedDict, total=False):
    page_ref: str
    focus: list[str]
    reuse_focus_ids: list[str]
    force_new_focus: NotRequired[bool]


FOCUS_CACHE_LIMIT = 16
FOCUS_TEXT_LIMIT = 300
FOCUS_ANSWER_LIMIT = 4000
FOCUS_TOTAL_TEXT_LIMIT = 24000


_COMMON_FOCUS_PROMPT = """You summarize ONE slide for a deck-level PPT agent.

Return STRICT JSON only:
{
  "common": {
    "title": "",
    "page_type": "",
    "topic": "",
    "summary": "",
    "key_points": [],
    "language": ""
  },
  "focused_results": [
    {"focus": "...", "answer": "..."}
  ]
}

Rules:
- Use the rendered slide image and the provided structured understanding.
- common should be concise and useful for selecting pages and planning edits.
- If focus_requests is empty, focused_results must be [].
- For each focus request, answer only objective facts already present on the slide. Be specific.
- Never write edit suggestions, generated copy, execution plans, routing decisions, or tool advice in focused_results.
"""


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _normalize_focus_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _dedupe_focus_items(items: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item or "").strip()[:FOCUS_TEXT_LIMIT]
        key = _normalize_focus_text(text)
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _new_focus_id() -> str:
    return f"focus_{uuid.uuid4().hex[:12]}"


def _focused_entries(wrapper: dict[str, Any] | None) -> list[dict[str, Any]]:
    focused = wrapper.get("focused") if isinstance(wrapper, dict) else None
    if not isinstance(focused, dict):
        return []
    entries = focused.get("entries")
    if not isinstance(entries, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fid = str(entry.get("focus_id") or "").strip()
        focus = str(entry.get("focus") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        if fid and focus and answer:
            out.append(
                {
                    "focus_id": fid,
                    "focus": focus[:FOCUS_TEXT_LIMIT],
                    "answer": answer[:FOCUS_ANSWER_LIMIT],
                    "source_run_id": str(entry.get("source_run_id") or ""),
                    "created_at": str(entry.get("created_at") or ""),
                    "last_used_at": str(entry.get("last_used_at") or entry.get("created_at") or ""),
                }
            )
    return out


def _append_focus_entries(
    existing_entries: list[dict[str, Any]],
    new_results: list[dict[str, str]],
    *,
    source_run_id: str,
    preserve_focus_texts: set[str] | None = None,
    preserve_focus_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    now = _now_iso()
    preserve_focus_texts = preserve_focus_texts or set()
    preserve_focus_ids = preserve_focus_ids or set()
    by_norm: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        by_norm[_normalize_focus_text(str(entry.get("focus") or ""))] = dict(entry)
    for item in new_results:
        focus = str(item.get("focus") or "").strip()[:FOCUS_TEXT_LIMIT]
        answer = str(item.get("answer") or "").strip()[:FOCUS_ANSWER_LIMIT]
        key = _normalize_focus_text(focus)
        if not focus or not answer:
            continue
        existing = by_norm.get(key)
        by_norm[key] = {
            "focus_id": str(existing.get("focus_id") if existing else "") or _new_focus_id(),
            "focus": focus,
            "answer": answer,
            "source_run_id": str(source_run_id or ""),
            "created_at": str(existing.get("created_at") if existing else "") or now,
            "last_used_at": now,
        }

    entries = list(by_norm.values())
    priority: dict[str, int] = {}
    for entry in entries:
        fid = str(entry.get("focus_id") or "")
        key = _normalize_focus_text(str(entry.get("focus") or ""))
        if fid in preserve_focus_ids or key in preserve_focus_texts:
            priority[fid] = 0
        else:
            priority[fid] = 1
    entries.sort(
        key=lambda e: (
            priority.get(str(e.get("focus_id") or ""), 1),
            str(e.get("last_used_at") or ""),
        ),
        reverse=True,
    )
    entries.sort(key=lambda e: priority.get(str(e.get("focus_id") or ""), 1))

    kept: list[dict[str, Any]] = []
    total = 0
    for entry in entries:
        if len(kept) >= FOCUS_CACHE_LIMIT:
            continue
        cost = len(str(entry.get("focus") or "")) + len(str(entry.get("answer") or ""))
        if kept and total + cost > FOCUS_TOTAL_TEXT_LIMIT:
            continue
        kept.append(entry)
        total += cost
    return kept


def _page_size(paths: WorkspacePaths, slot: int) -> dict[str, Any]:
    existing = paths.page_understanding_json(slot)
    if existing.exists():
        try:
            obj = read_json(existing)
            core = obj.get("core") if isinstance(obj, dict) else None
            ps = core.get("page_size_pt") if isinstance(core, dict) else None
            if isinstance(ps, dict):
                return {"w": ps.get("w"), "h": ps.get("h")}
        except Exception:
            pass
    mf = paths.project_manifest_json()
    if mf.exists():
        try:
            manifest = read_json(mf)
            ps = manifest.get("page_size_pt") if isinstance(manifest, dict) else None
            if isinstance(ps, dict):
                return ps
        except Exception:
            pass
    return {"w": None, "h": None}


def _build_understand_input(paths: WorkspacePaths, slot: int) -> dict[str, Any]:
    from agent_backend.agent.tools.html_to_pptist import slide_to_plan_page

    slide = _load_slide_for_reread(paths, slot)
    page_idx0 = int(slot) - 1
    plan_page = slide_to_plan_page(slide, page_id=f"page{page_idx0}")
    materialize_pptist_slide_assets(paths, slot, slide)
    asset_dir = page_asset_source_dir(paths, slot)
    _materialize_images(plan_page, asset_dir)

    png = paths.reread_page_png(slot)
    if not png.exists():
        png = paths.page_png(slot)
    if not png.exists():
        raise RuntimeError(f"page understanding: no current or baseline PNG for slot {slot}")

    return {
        "schema_version": "understand_input_v1",
        "page_num": int(slot),
        "page_size_pt": _page_size(paths, slot),
        "bundle_dir": str(asset_dir),
        "plan_page": plan_page,
        "page_png_path": str(png),
        "options": {
            "need_image_descriptions": True,
            "need_original_layout_description": True,
        },
    }


def _default_common(core: dict[str, Any]) -> dict[str, Any]:
    texts = [str(t.get("text") or "").strip() for t in core.get("texts") or [] if isinstance(t, dict)]
    title = ""
    for t in core.get("texts") or []:
        if isinstance(t, dict) and str(t.get("kind") or "") in {"title", "subheading"}:
            title = str(t.get("text") or "").strip()
            if title:
                break
    if not title and texts:
        title = texts[0][:80]
    blob = " ".join(texts)
    return {
        "title": title,
        "page_type": "",
        "topic": title,
        "summary": blob[:360],
        "key_points": [t[:160] for t in texts[:6]],
        "language": "",
    }


def _call_common_focus(
    *,
    core: dict[str, Any],
    png_path: Path | None,
    focus: list[str],
    model: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    common = _default_common(core)
    focused_results: list[dict[str, str]] = []
    try:
        base_url, key = _resolve_backend(None)
        payload = {
            "core": core,
            "focus_requests": focus,
        }
        images = [png_path.read_bytes()] if png_path and png_path.exists() else []
        _raw, obj, err = _call_claude_json(
            base_url=base_url,
            api_key=key,
            model=model,
            system_prompt=_COMMON_FOCUS_PROMPT,
            user_text=json.dumps(payload, ensure_ascii=False, indent=2),
            images=images,
            max_tokens=4096,
        )
        if err or not isinstance(obj, dict):
            warnings.append(f"common_focus_call_error: {err or 'invalid_json'}")
            return common, focused_results, warnings
        got_common = obj.get("common")
        if isinstance(got_common, dict):
            common = {
                "title": str(got_common.get("title") or common["title"]),
                "page_type": str(got_common.get("page_type") or ""),
                "topic": str(got_common.get("topic") or ""),
                "summary": str(got_common.get("summary") or common["summary"]),
                "key_points": [
                    str(x) for x in (got_common.get("key_points") or []) if isinstance(x, (str, int, float))
                ][:8],
                "language": str(got_common.get("language") or ""),
            }
        raw_results = obj.get("focused_results")
        if isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                f = str(item.get("focus") or "").strip()
                ans = str(item.get("answer") or "").strip()
                if f and ans:
                    focused_results.append({"focus": f, "answer": ans})
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"common_focus_exception: {type(exc).__name__}: {exc}")
    return common, focused_results, warnings


def _load_wrapper(paths: WorkspacePaths, slot: int) -> dict[str, Any] | None:
    p = paths.page_understanding_json(slot)
    if not p.exists():
        return None
    try:
        obj = read_json(p)
    except Exception:
        return None
    if not (isinstance(obj, dict) and isinstance(obj.get("core"), dict)):
        return None
    # Product is not live yet; old files are intentionally treated as missing so
    # v2 freshness/focus semantics stay simple.
    if obj.get("schema_version") != "page_understanding_v2":
        return None
    return obj


def ensure_page_understanding(
    *,
    paths: WorkspacePaths,
    project_id: str,
    slot: int,
    display_page: int | None,
    model: str,
    focus: list[str] | None = None,
    reuse_focus_ids: list[str] | None = None,
    agent_run_id: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Return the shared understanding wrapper, refreshing only when needed."""
    focus_items = _dedupe_focus_items(focus)
    reuse_ids = [str(x or "").strip() for x in reuse_focus_ids or [] if str(x or "").strip()]

    pending = is_pending_reread(paths, slot)
    existing = _load_wrapper(paths, slot)
    needs_full = pending or existing is None
    existing_entries = [] if needs_full else _focused_entries(existing)
    existing_by_id = {str(e.get("focus_id")): e for e in existing_entries}
    now = _now_iso()
    reused_entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    if needs_full and reuse_ids:
        warnings.extend([f"invalid_or_stale_focus_id:{fid}" for fid in reuse_ids])
    if not needs_full and reuse_ids:
        for fid in reuse_ids:
            entry = existing_by_id.get(fid)
            if entry:
                entry = dict(entry)
                entry["last_used_at"] = now
                reused_entries.append(entry)
            else:
                warnings.append(f"invalid_or_stale_focus_id:{fid}")
    existing_focus_norms = {_normalize_focus_text(str(e.get("focus") or "")) for e in existing_entries}
    new_focus_items = [f for f in focus_items if _normalize_focus_text(f) not in existing_focus_norms]
    duplicate_reused = [
        dict(e, last_used_at=now)
        for e in existing_entries
        if _normalize_focus_text(str(e.get("focus") or "")) in {_normalize_focus_text(f) for f in focus_items}
    ]
    reused_entries.extend([e for e in duplicate_reused if str(e.get("focus_id") or "") not in {str(x.get("focus_id") or "") for x in reused_entries}])
    reused_ids = {str(e.get("focus_id") or "") for e in reused_entries}
    if reused_ids:
        existing_entries = [
            {**e, "last_used_at": now} if str(e.get("focus_id") or "") in reused_ids else e
            for e in existing_entries
        ]
    needs_focus = bool(new_focus_items)

    if not needs_full and not needs_focus and existing is not None:
        if reused_entries:
            existing["focused"] = {
                "entries": _append_focus_entries(
                    [*existing_entries],
                    [],
                    source_run_id=agent_run_id,
                    preserve_focus_ids=reused_ids,
                )
            }
            write_json(paths.page_understanding_json(slot), existing)
        return {**existing, "_status": "cached", "_reused_focus": reused_entries, "_focus_warnings": warnings}

    page = int(display_page) if display_page is not None else pageorder.position_for_slot(paths, slot)
    emit(project_id, {"type": "understanding_started", "page": page, "slot": slot})

    wrapper: dict[str, Any]
    try:
        if needs_full:
            understand_input = _build_understand_input(paths, slot)
            core = run_step1(understand_input=understand_input, model=model, dry_run=dry_run)
            core["page_num"] = int(slot)
            core["page_id"] = f"page{int(slot) - 1}"
            png_path = Path(str(understand_input.get("page_png_path") or ""))
            common, focused_results, extra_warnings = _call_common_focus(
                core=core,
                png_path=png_path,
                focus=focus_items,
                model=model,
            )
            if extra_warnings:
                core.setdefault("warnings", []).extend(extra_warnings)
            focused = {
                "entries": _append_focus_entries(
                    [],
                    focused_results,
                    source_run_id=agent_run_id,
                    preserve_focus_texts={_normalize_focus_text(f) for f in focus_items},
                )
            }
            wrapper = {
                "schema_version": "page_understanding_v2",
                "core": core,
                "common": common,
                "focused": focused,
            }
            write_json(paths.page_understanding_json(slot), wrapper)
            clear_pending_reread(paths, slot)
            wrapper["_requested_focus"] = focus_items
            wrapper["_reused_focus"] = []
            wrapper["_focus_warnings"] = warnings
            wrapper["_status"] = "refreshed" if pending else "generated"
            return wrapper

        assert existing is not None
        core = existing["core"]
        png_path = paths.reread_page_png(slot) if paths.reread_page_png(slot).exists() else paths.page_png(slot)
        _common, focused_results, call_warnings = _call_common_focus(
            core=core,
            png_path=png_path,
            focus=new_focus_items,
            model=model,
        )
        all_warnings = [*warnings, *call_warnings]
        if call_warnings:
            core.setdefault("warnings", []).extend(call_warnings)
        existing["focused"] = {
            "entries": _append_focus_entries(
                existing_entries,
                focused_results,
                source_run_id=agent_run_id,
                preserve_focus_texts={_normalize_focus_text(f) for f in new_focus_items},
                preserve_focus_ids=reused_ids,
            )
        }
        write_json(paths.page_understanding_json(slot), existing)
        existing["_status"] = "focused"
        existing["_requested_focus"] = new_focus_items
        existing["_reused_focus"] = reused_entries
        existing["_focus_warnings"] = all_warnings
        return existing
    finally:
        emit(project_id, {"type": "understanding_finished", "page": page, "slot": slot})


def _inspect_view(paths: WorkspacePaths, slot: int) -> dict[str, Any]:
    entry = pageorder.entry_for_slot(paths, slot) or {}
    if is_pending_reread(paths, slot):
        status = "stale"
        wrapper = None
    else:
        wrapper = _load_wrapper(paths, slot)
        status = "current" if wrapper else "missing"
    entries = _focused_entries(wrapper) if wrapper and status == "current" else []
    return {
        "page": entry.get("position"),
        "page_ref": pageorder.page_ref_for_slot(slot),
        "understanding_status": status,
        "common": (wrapper.get("common") if isinstance(wrapper, dict) else {}) or {},
        "available_focus": [
            {"focus_id": str(e.get("focus_id") or ""), "focus": str(e.get("focus") or "")}
            for e in entries
        ],
    }


def _tool_view(paths: WorkspacePaths, slot: int, wrapper: dict[str, Any], agent_run_id: str) -> dict[str, Any]:
    entry = pageorder.entry_for_slot(paths, slot) or {}
    entries = _focused_entries(wrapper)
    requested_norms = {_normalize_focus_text(f) for f in wrapper.get("_requested_focus") or []}
    returned_ids = {str(e.get("focus_id") or "") for e in wrapper.get("_reused_focus") or []}
    focused_answers: list[dict[str, Any]] = []
    for entry_focus in entries:
        fid = str(entry_focus.get("focus_id") or "")
        if fid in returned_ids or _normalize_focus_text(str(entry_focus.get("focus") or "")) in requested_norms:
            focused_answers.append(entry_focus)
    return {
        "page": entry.get("position"),
        "page_ref": pageorder.page_ref_for_slot(slot),
        "status": wrapper.get("_status") or "cached",
        "common": wrapper.get("common") or {},
        "focused": focused_answers,
        "warnings": wrapper.get("_focus_warnings") or [],
    }


@tool
def understand_pages(pages: list[PageUnderstandRequest], runtime: ToolRuntime, inspect_only: bool = False) -> dict[str, Any]:
    """Understand one or more pages and optionally extract focused information.

    Use this when page content affects planning, page selection, cross-page
    summaries, or deterministic text you must embed into later edit demands. Use
    inspect_only=true first to see whether cached focused facts can be reused.
    Each item is {"page_ref": "page@<stable id>", "focus": ["..."],
    "reuse_focus_ids": ["..."], "force_new_focus": true}. Use
    force_new_focus only after inspecting cached focus and deciding the new
    objective facts are genuinely missing. Put all focus needs for the same
    page into one item whenever possible.
    """
    pid = require_project_id(runtime)
    agent_run_id = require_agent_run_id(runtime)
    paths = workspace_for(pid)
    model = agent_model_name()

    merged: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for item in pages or []:
        ref = str(item.get("page_ref") or "")
        slot = pageorder.slot_for_page_ref(paths, ref)
        if slot is None:
            continue
        if slot not in merged:
            merged[slot] = {"focus": [], "reuse_focus_ids": [], "force_new_focus": False}
            order.append(slot)
        for focus in item.get("focus") or []:
            f = str(focus or "").strip()
            if f and f not in merged[slot]["focus"]:
                merged[slot]["focus"].append(f)
        for fid in item.get("reuse_focus_ids") or []:
            val = str(fid or "").strip()
            if val and val not in merged[slot]["reuse_focus_ids"]:
                merged[slot]["reuse_focus_ids"].append(val)
        if bool(item.get("force_new_focus")):
            merged[slot]["force_new_focus"] = True

    if not order:
        return {"project_id": pid, "ok": False, "error": "no valid page_ref values"}

    results_by_slot: dict[int, dict[str, Any]] = {}
    with project_lock(pid):
        if inspect_only:
            return {
                "project_id": pid,
                "ok": True,
                "revision": pageorder.revision(paths),
                "inspect_only": True,
                "pages": [_inspect_view(paths, slot) for slot in order],
            }
        preflight_pages: list[dict[str, Any]] = []
        for slot in order:
            view = _inspect_view(paths, slot)
            has_current_focus = (
                view.get("understanding_status") == "current"
                and bool(view.get("available_focus"))
            )
            asks_new_focus = bool(merged[slot]["focus"])
            confirms_new_focus = bool(merged[slot].get("force_new_focus"))
            reuses_focus = bool(merged[slot]["reuse_focus_ids"])
            if has_current_focus and asks_new_focus and not confirms_new_focus and not reuses_focus:
                preflight_pages.append(view)
        if preflight_pages:
            return {
                "project_id": pid,
                "ok": False,
                "error": "focus_preflight_required",
                "revision": pageorder.revision(paths),
                "inspect_only": True,
                "pages": preflight_pages,
                "hint": (
                    "Review available_focus semantically first. Then call understand_pages "
                    "with reuse_focus_ids for covered facts, and set force_new_focus=true "
                    "only for genuinely missing focus directions."
                ),
            }
        max_workers = min(MAX_PARALLEL_UNDERSTAND, len(order))
        if max_workers <= 1:
            for slot in order:
                wrapper = ensure_page_understanding(
                    paths=paths,
                    project_id=pid,
                    slot=slot,
                    display_page=pageorder.position_for_slot(paths, slot),
                    model=model,
                    focus=merged[slot]["focus"],
                    reuse_focus_ids=merged[slot]["reuse_focus_ids"],
                    agent_run_id=agent_run_id,
                )
                results_by_slot[slot] = _tool_view(paths, slot, wrapper, agent_run_id)
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="understand") as pool:
                futures = {
                    pool.submit(
                        ensure_page_understanding,
                        paths=paths,
                        project_id=pid,
                        slot=slot,
                        display_page=pageorder.position_for_slot(paths, slot),
                        model=model,
                        focus=merged[slot]["focus"],
                        reuse_focus_ids=merged[slot]["reuse_focus_ids"],
                        agent_run_id=agent_run_id,
                    ): slot
                    for slot in order
                }
                for fut in as_completed(futures):
                    slot = futures[fut]
                    results_by_slot[slot] = _tool_view(paths, slot, fut.result(), agent_run_id)

    return {
        "project_id": pid,
        "ok": True,
        "revision": pageorder.revision(paths),
        "pages": [results_by_slot[s] for s in order if s in results_by_slot],
    }


__all__ = ["understand_pages", "ensure_page_understanding", "PageUnderstandRequest"]
