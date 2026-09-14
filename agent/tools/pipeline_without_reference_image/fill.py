"""Tool: fill_empty_pages — create full content on already-inserted blank pages."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from typing_extensions import TypedDict

from agent_backend.agent.models import agent_model_name
from agent_backend.agent.tools.context import (
    assert_unique_page_jobs,
    emit,
    execute_page_jobs,
    page_job,
    require_agent_run_id,
    require_project_id,
    require_session_id,
    workspace_for,
)
from agent_backend.agent.tools.deck_style import public_style_row, require_ready_style
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.graph import (
    commit_turn_html_to_pptist,
)
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.steps import (
    run_step4_qa,
)
from agent_backend.agent.tools.pipeline_without_reference_image.no_reference_step3 import (
    run_step3_without_reference,
)
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.step_2_plan.skills.base import (
    _call_claude_json,
)
from agent_backend.agent.tools.table_spec import table_from_composer
from agent_backend.workspace import pageorder
from agent_backend.workspace.assets import page_asset_source_dir
from agent_backend.workspace.dirty import mark_pending_reread
from agent_backend.workspace.paths import next_turn_dir, read_json, write_json, write_text
from agent_backend.workspace.repo import record_turn


class PageFill(TypedDict):
    page_ref: str
    demand: str


_COMPOSER_SYSTEM_PROMPT = """You are the Page Spec Composer for an AI PPT editor.

Convert ONE natural-language request for a blank slide into strict JSON.
Return JSON only. Do not include markdown fences.

Output schema:
{
  "schema_version": "page_spec_composer_v2",
  "texts": [
    {
      "kind": "title | subheading | body | bullet_item | caption | label | date",
      "text": "final slide copy",
      "segments": ["only for a real list, timeline, or staged sequence"]
    }
  ],
  "images": [
    {
      "asset_key": "asset_1",
      "description_en": "Concise factual description",
      "usage": "where/how this real image should appear"
    }
  ],
  "tables": [
    {
      "header_rows": 1,
      "rows": [["Header A", "Header B"], ["Cell A", "Cell B"]]
    }
  ],
  "visual_intent": {"requirements_text": "full-slide layout and visual organization"}
}

Rules:
- Preserve exact final copy the user explicitly provided.
- If the user gives only a topic, write concise slide-ready copy, but do not invent facts not supported by the request or attached images.
- Use texts for paragraphs and standalone labels. Use segments only for true lists, timelines, or stage sequences; joined segments must match text.
- Every staged image listed in the input must appear exactly once in images, using its asset_key. Do not invent image files.
- Tables are rectangular matrices only, with optional one header row. No merged cells, charts, formulas, audio, video, or nested structures.
- visual_intent.requirements_text should describe the whole page's composition, hierarchy, placement, and style needs for a designer. Keep it complete but concise.
"""


def _project_page_size(paths) -> dict[str, Any]:
    try:
        manifest = read_json(paths.project_manifest_json())
    except Exception:
        manifest = {}
    size = manifest.get("page_size_pt") if isinstance(manifest, dict) else {}
    if not isinstance(size, dict):
        size = {}
    w = size.get("w")
    h = size.get("h")
    return {
        "w": float(w) if isinstance(w, (int, float)) and w > 0 else 1000.0,
        "h": float(h) if isinstance(h, (int, float)) and h > 0 else 562.5,
    }


def _is_blank_slide(paths, slot: int) -> bool:
    p = paths.pptist_slide_json(int(slot))
    if not p.exists():
        return False
    try:
        slide = read_json(p)
    except Exception:
        return False
    elements = slide.get("elements") if isinstance(slide, dict) else None
    return isinstance(elements, list) and len(elements) == 0


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _pending_assets_for_run(paths, slot: int, run_id: str) -> tuple[list[dict[str, Any]], Path]:
    manifest = paths.pending_uploads_json(int(slot))
    uploads: list[dict[str, Any]] = []
    if manifest.exists():
        try:
            payload = read_json(manifest)
            raw = payload.get("uploads") if isinstance(payload, dict) else []
            uploads = [it for it in raw if isinstance(it, dict)] if isinstance(raw, list) else []
        except Exception:
            uploads = []

    out: list[dict[str, Any]] = []
    for item in uploads:
        if str(item.get("run_id") or "") != str(run_id):
            continue
        filename = str(item.get("filename") or "").strip()
        if not filename:
            continue
        path = paths.page_assets_dir(int(slot)) / "uploads" / filename
        if not path.is_file():
            continue
        w, h = _image_size(path)
        key = f"asset_{len(out) + 1}"
        out.append(
            {
                "asset_key": key,
                "filename": filename,
                "path": str(path),
                "original_filename": str(item.get("original_filename") or Path(filename).name),
                "user_note": str(item.get("user_note") or ""),
                "artifact_ref": str(item.get("artifact_ref") or ""),
                "width": w,
                "height": h,
            }
        )
    return out, manifest


def _asset_prompt_view(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "asset_key": a["asset_key"],
            "original_filename": a.get("original_filename") or "",
            "user_note": a.get("user_note") or "",
            "width": a.get("width"),
            "height": a.get("height"),
        }
        for a in assets
    ]


def _compose_page_spec(
    *,
    demand: str,
    page: int,
    page_ref: str,
    page_size_pt: dict[str, Any],
    deck_style: dict[str, Any],
    assets: list[dict[str, Any]],
    model: str,
    turn_dir: Path,
) -> dict[str, Any]:
    request_obj = {
        "schema_version": "page_spec_composer_request_v1",
        "page": int(page),
        "page_ref": page_ref,
        "page_size_pt": page_size_pt,
        "user_demand": demand,
        "deck_style": deck_style,
        "staged_images": _asset_prompt_view(assets),
    }
    write_json(turn_dir / "composer_request.json", request_obj)
    labeled_images: list[tuple[str, bytes]] = []
    for asset in assets[:6]:
        p = Path(str(asset.get("path") or ""))
        if p.is_file():
            label = f"{asset['asset_key']} / {asset.get('original_filename') or p.name}: {asset.get('user_note') or ''}"
            labeled_images.append((label, p.read_bytes()))
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_COMPOSER_SYSTEM_PROMPT,
        user_text=json.dumps(request_obj, ensure_ascii=False, indent=2),
        labeled_images=labeled_images,
        max_output_tokens=4096,
        retries=1,
        tag="fill.composer",
        reasoning_effort="minimal",
    )
    write_text(turn_dir / "composer_raw_response.txt", raw or "")
    if obj is None:
        repair_prompt = {
            "instruction": "Repair the previous response into valid Page Spec Composer JSON. Preserve all semantic content; return JSON only.",
            "schema": "page_spec_composer_v2",
            "previous_error": err or "invalid_json",
            "previous_response": raw or "",
        }
        raw2, obj2, err2 = _call_claude_json(
            model=model,
            system_prompt=_COMPOSER_SYSTEM_PROMPT,
            user_text=json.dumps(repair_prompt, ensure_ascii=False, indent=2),
            max_output_tokens=4096,
            retries=0,
            tag="fill.composer_repair",
            reasoning_effort="minimal",
        )
        write_text(turn_dir / "composer_repair_response.txt", raw2 or "")
        obj = obj2
        if obj is None:
            raise RuntimeError(f"composer produced invalid JSON: {err2 or err or 'unknown'}")

    spec = _validate_composer_output(obj, assets)
    write_json(turn_dir / "composer_output.json", spec)
    return spec


def _validate_composer_output(obj: dict[str, Any], assets: list[dict[str, Any]]) -> dict[str, Any]:
    if obj.get("schema_version") != "page_spec_composer_v2":
        raise ValueError("composer schema_version must be page_spec_composer_v2")

    allowed_kinds = {"title", "subheading", "body", "bullet_item", "caption", "label", "date"}
    texts: list[dict[str, Any]] = []
    for idx, item in enumerate(obj.get("texts") or []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "body").strip()
        if kind not in allowed_kinds:
            kind = "body"
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        out: dict[str, Any] = {"kind": kind, "text": text}
        segments = item.get("segments")
        if isinstance(segments, list):
            segs = [str(s).strip() for s in segments if str(s).strip()]
            if len(segs) > 1:
                out["segments"] = segs
        texts.append(out)
    if not texts and not assets and not obj.get("tables"):
        raise ValueError("composer output has no text, image, or table content")

    expected = [str(a["asset_key"]) for a in assets]
    images_in = obj.get("images") or []
    images: list[dict[str, Any]] = []
    seen: list[str] = []
    for item in images_in:
        if not isinstance(item, dict):
            continue
        key = str(item.get("asset_key") or "").strip()
        if not key:
            continue
        seen.append(key)
        images.append(
            {
                "asset_key": key,
                "description_en": str(item.get("description_en") or "").strip(),
                "usage": str(item.get("usage") or "").strip(),
            }
        )
    if sorted(seen) != sorted(expected) or len(seen) != len(set(seen)):
        raise ValueError("composer must include every staged image exactly once")

    tables: list[dict[str, Any]] = []
    for item in obj.get("tables") or []:
        if not isinstance(item, dict):
            continue
        rows = item.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        matrix: list[list[str]] = []
        width: int | None = None
        for row in rows:
            if not isinstance(row, list):
                raise ValueError("table rows must be arrays")
            vals = [str(cell or "").strip() for cell in row]
            if width is None:
                width = len(vals)
            if width <= 0 or len(vals) != width:
                raise ValueError("tables must be rectangular")
            matrix.append(vals)
        header_rows = item.get("header_rows")
        try:
            hr = int(header_rows)
        except Exception:
            hr = 0
        tables.append({"header_rows": 1 if hr > 0 else 0, "rows": matrix})

    vi = obj.get("visual_intent") if isinstance(obj.get("visual_intent"), dict) else {}
    visual = {"requirements_text": str(vi.get("requirements_text") or "").strip()}
    return {
        "schema_version": "page_spec_composer_v2",
        "texts": texts,
        "images": images,
        "tables": tables,
        "visual_intent": visual,
    }


def _deck_palette(deck_style: dict[str, Any]) -> dict[str, Any]:
    colors = deck_style.get("colors") if isinstance(deck_style.get("colors"), dict) else {}
    top3 = [str(c).strip() for c in (colors.get("top3") or []) if str(c).strip()]
    return {"primary": top3[0] if top3 else "", "fills": top3[1:5]}


def _soft_image_size(asset: dict[str, Any]) -> tuple[float, float]:
    w = asset.get("width")
    h = asset.get("height")
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 0 or h <= 0:
        return 240.0, 135.0
    ratio = float(w) / float(h)
    if ratio >= 1:
        return 260.0, max(90.0, 260.0 / ratio)
    return max(90.0, 190.0 * ratio), 190.0


def _compile_create_page_spec(
    *,
    composer: dict[str, Any],
    assets: list[dict[str, Any]],
    page_size_pt: dict[str, Any],
    deck_style: dict[str, Any],
    demand: str,
    slot: int,
) -> dict[str, Any]:
    texts: list[dict[str, Any]] = []
    for idx, item in enumerate(composer.get("texts") or []):
        text_item = {
            "id": f"t{len(texts)}",
            "kind": str(item.get("kind") or "body"),
            "text": str(item.get("text") or ""),
        }
        if isinstance(item.get("segments"), list) and len(item["segments"]) > 1:
            text_item["segments"] = [str(s) for s in item["segments"]]
        texts.append(text_item)

    tables: list[dict[str, Any]] = []
    for table_idx, tbl in enumerate(composer.get("tables") or []):
        if not isinstance(tbl, dict):
            continue
        try:
            tables.append(table_from_composer(tbl.get("rows"), f"tbl{table_idx}", int(tbl.get("header_rows") or 0)))
        except ValueError:
            continue

    by_asset_key = {str(a["asset_key"]): a for a in assets}
    images: list[dict[str, Any]] = []
    for idx, item in enumerate(composer.get("images") or []):
        key = str(item.get("asset_key") or "")
        asset = by_asset_key.get(key)
        if not asset:
            continue
        dw, dh = _soft_image_size(asset)
        desc = str(item.get("description_en") or "").strip()
        usage = str(item.get("usage") or "").strip()
        if usage:
            desc = f"{desc} Usage: {usage}".strip()
        images.append(
            {
                "id": f"img{idx}",
                "src": str(asset.get("filename") or ""),
                "description_en": desc,
                "display_w_pt": dw,
                "display_h_pt": dh,
            }
        )

    visual_req = str((composer.get("visual_intent") or {}).get("requirements_text") or "").strip()
    if not visual_req:
        visual_req = demand

    state = {
        "schema_version": "understand_output_v2",
        "page_num": int(slot),
        "page_id": f"page{int(slot) - 1}",
        "page_size_pt": page_size_pt,
        "bundle_dir": "",
        "palette": _deck_palette(deck_style),
        "texts": texts,
        "images": images,
        "tables": tables,
        "original_layout_description_en": "",
        "page_png_path": "",
        "warnings": [],
    }
    return {
        "schema_version": "step2_output_v2",
        "user_request": demand,
        "selected_refs": [],
        "visual_intent": {
            "enabled": True,
            "requirements_text": visual_req,
            "default_details": [],
            "requested_by_plan": True,
            "requested_by_skill": False,
        },
        "plan": {
            "schema_version": "fill_empty_page_plan_v1",
            "content_intents": [],
            "visual_intent": {"enabled": True, "requirements_text": visual_req},
            "skip_reasons": ["fill_empty_pages bypasses edit Step2 planner and skills"],
        },
        "compile": {
            "schema_version": "compile_output_v1",
            "executed_intent_ids": [],
            "visual_intent": {"enabled": True, "requirements_text": visual_req, "default_details": []},
            "understand_modified": state,
        },
        "warnings": {"composer": [], "compile": []},
    }


def _inject_bundle_dir(step2_output: dict[str, Any], paths, slot: int) -> None:
    um = step2_output["compile"]["understand_modified"]
    um["bundle_dir"] = str(page_asset_source_dir(paths, int(slot)))


def _clean_consumed_run_uploads(paths, slot: int, run_id: str) -> None:
    manifest = paths.pending_uploads_json(int(slot))
    if manifest.exists():
        try:
            payload = read_json(manifest)
            uploads = payload.get("uploads") if isinstance(payload, dict) else []
            remaining = [
                item
                for item in (uploads if isinstance(uploads, list) else [])
                if not (isinstance(item, dict) and str(item.get("run_id") or "") == str(run_id))
            ]
            write_json(manifest, {"schema_version": "pending_uploads_v1", "uploads": remaining})
        except Exception:
            pass
    run_upload_dir = paths.page_assets_dir(int(slot)) / "uploads" / str(run_id)
    shutil.rmtree(run_upload_dir, ignore_errors=True)


@tool
def fill_empty_pages(fills: list[PageFill], runtime: ToolRuntime) -> dict[str, Any]:
    """Fill already-inserted blank pages with complete editable slide content.

    Each item supplies a stable reference to an existing blank page and a complete
    natural-language page brief. The brief carries exact required copy, permitted
    content generation, the page goal, visual requirements, and staged-image roles.
    The tool accepts only pages whose PPTist `elements` list is empty.
    """
    pid = require_project_id(runtime)
    try:
        sid = require_session_id(runtime)
    except Exception:
        sid = None
    run_id = require_agent_run_id(runtime)
    paths = workspace_for(pid)
    model = agent_model_name()

    parsed: list[dict[str, Any]] = []
    for item in fills or []:
        slot = pageorder.slot_for_page_ref(paths, str(item.get("page_ref") or ""))
        if slot is None:
            continue
        demand = str(item.get("demand") or "").strip()
        if not demand:
            continue
        parsed.append(
            {
                "slot": int(slot),
                "page": int(pageorder.position_for_slot(paths, int(slot)) or 0),
                "page_ref": pageorder.page_ref_for_slot(int(slot)),
                "demand": demand,
            }
        )
    if not parsed:
        return {"project_id": pid, "ok": False, "error": "no valid fills"}

    deck_style_row = require_ready_style(pid, interrupt_when_unready=True)
    emit(pid, {"type": "batch_started", "project_id": pid, "count": len(parsed), "agent_run_id": run_id})

    def _progress(item: dict[str, Any], index: int, stage: str, label: str) -> None:
        emit(
            pid,
            {
                "type": "task_progress",
                "page": int(item["page"]),
                "slot": int(item["slot"]),
                "index": int(index),
                "stage": stage,
                "label": label,
                "agent_run_id": run_id,
            },
        )

    def _run_one(index: int, item: dict[str, Any], style_row: dict[str, Any]) -> dict[str, Any]:
        slot = int(item["slot"])
        page = int(item["page"])
        demand = str(item["demand"])
        if not _is_blank_slide(paths, slot):
            return {"page": page, "ok": False, "status": "page_is_not_blank"}

        turn_dir = next_turn_dir(paths, slot)
        started_at = time.time()
        manifest = {
            "project_id": pid,
            "page_num": slot,
            "display_page": page,
            "demand": demand,
            "tool": "fill_empty_pages",
            "model": model,
            "deck_style_revision": int(style_row.get("revision") or 0),
            "started_at": started_at,
            "status": "running",
        }
        write_json(turn_dir / "manifest.json", manifest)
        emit(pid, {"type": "task_started", "page": page, "slot": slot, "demand": demand, "index": index, "agent_run_id": run_id})
        try:
            page_size_pt = _project_page_size(paths)
            deck_style = style_row.get("style_json") if isinstance(style_row.get("style_json"), dict) else {}
            assets, _pending_manifest = _pending_assets_for_run(paths, slot, run_id)

            _progress(item, index, "fill_compose", "正在整理页面内容")
            composer = _compose_page_spec(
                demand=demand,
                page=page,
                page_ref=str(item["page_ref"]),
                page_size_pt=page_size_pt,
                deck_style=deck_style,
                assets=assets,
                model=model,
                turn_dir=turn_dir,
            )
            step2_output = _compile_create_page_spec(
                composer=composer,
                assets=assets,
                page_size_pt=page_size_pt,
                deck_style=deck_style,
                demand=demand,
                slot=slot,
            )
            _inject_bundle_dir(step2_output, paths, slot)
            write_json(turn_dir / "create_page_spec.json", step2_output)

            _progress(item, index, "reassemble", "正在构建可编辑页面")
            step2_output["fill_mode"] = True
            step3_result = run_step3_without_reference(
                step2_output=step2_output,
                paths=paths,
                page_num=slot,
                model=model,
                dry_run=False,
                turn_dir=turn_dir,
                deck_style=deck_style,
                mode="create",
            )
            visual_check = step3_result.get("visual_self_check") or {}
            if visual_check.get("status") in {"failed", "unavailable"}:
                raise RuntimeError("step3 visual self-check did not produce an acceptable page")
            if visual_check.get("verdict") == "revise" and not visual_check.get("repair_applied"):
                raise RuntimeError("step3 visual self-check found an unrepaired major issue")
            write_json(
                turn_dir / "step3_result.json",
                {
                    "chunk_path": step3_result.get("chunk_path"),
                    "prep_warnings": step3_result.get("prep_warnings") or [],
                    "soft_warnings": step3_result.get("soft_warnings") or [],
                    "has_layout_intent": bool(step3_result.get("has_layout_intent")),
                    "used_beautify_reference": bool(step3_result.get("used_beautify_reference")),
                    "visual_self_check": step3_result.get("visual_self_check") or {},
                },
            )
            try:
                chunk_path = Path(str(step3_result.get("chunk_path") or ""))
                if chunk_path.exists():
                    write_text(turn_dir / "chunk_after_step3.html", chunk_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass

            _progress(item, index, "qa", "正在检查页面布局")
            try:
                qa = run_step4_qa(
                    paths=paths,
                    page_num=slot,
                    title="PPTAgent",
                    bundle_dir=Path(str(step3_result.get("bundle_dir") or "")),
                    chunk_path=Path(str(step3_result.get("chunk_path") or "")),
                )
            except Exception as exc:  # noqa: BLE001
                qa = {"skipped": True, "reason": "qa_failed", "error": f"{type(exc).__name__}: {exc}"}
            write_json(turn_dir / "qa_result.json", qa)
            try:
                chunk_path = Path(str(step3_result.get("chunk_path") or ""))
                if chunk_path.exists():
                    write_text(turn_dir / "chunk_after_step4.html", chunk_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass

            _progress(item, index, "commit", "正在应用页面修改")
            commit = commit_turn_html_to_pptist(
                paths=paths,
                page_num=slot,
                turn_dir=turn_dir,
                step3_result=step3_result,
                title="PPTAgent",
            )
            _clean_consumed_run_uploads(paths, slot, run_id)
            mark_pending_reread(paths, slot)

            manifest.update(
                {
                    "status": "ok",
                    "finished_at": time.time(),
                    "duration_seconds": time.time() - started_at,
                    "commit_result": commit,
                    "used_staged_images": [a.get("filename") for a in assets],
                }
            )
            write_json(turn_dir / "manifest.json", manifest)
            record_turn(
                project_id=pid,
                session_id=sid,
                page_num=slot,
                demand=demand,
                ok=True,
                error=None,
                turn_dir=str(turn_dir),
            )
            emit(pid, {"type": "task_finished", "page": page, "slot": slot, "demand": demand, "index": index, "turn_dir": str(turn_dir), "errors": [], "prepare_reread_png": False, "agent_run_id": run_id})
            return {"page": page, "ok": True, "status": "filled", "turn_dir": str(turn_dir)}
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            write_json(turn_dir / "error.json", {"error": err, "traceback": traceback.format_exc()})
            manifest.update({"status": "failed", "finished_at": time.time(), "duration_seconds": time.time() - started_at, "error": err})
            write_json(turn_dir / "manifest.json", manifest)
            record_turn(project_id=pid, session_id=sid, page_num=slot, demand=demand, ok=False, error=err, turn_dir=str(turn_dir))
            emit(pid, {"type": "task_failed", "page": page, "slot": slot, "demand": demand, "index": index, "turn_dir": str(turn_dir), "error": err, "agent_run_id": run_id})
            return {"page": page, "ok": False, "status": "fill_failed", "turn_dir": str(turn_dir)}

    assert_unique_page_jobs([int(item["slot"]) for item in parsed])
    latest = public_style_row(pid)
    if latest.get("status") != "ready" or not isinstance(latest.get("style_json"), dict):
        raise RuntimeError("deck style became unavailable before filling blank pages")
    deck_style_row = latest

    def _run_one_coordinated(index: int) -> dict[str, Any]:
        item = parsed[index]
        with page_job(pid, int(item["slot"])):
            current_page = pageorder.position_for_slot(paths, int(item["slot"]))
            if current_page is None:
                return {"page": item["page"], "slot": item["slot"], "ok": False, "status": "page_not_found"}
            item = {**item, "page": int(current_page)}
            return _run_one(index, item, deck_style_row)

    results_by_index = execute_page_jobs(
        pid,
        [(idx, int(item["slot"])) for idx, item in enumerate(parsed)],
        _run_one_coordinated,
    )

    emit(pid, {"type": "batch_finished", "project_id": pid, "agent_run_id": run_id})
    results = [results_by_index[i] for i in range(len(parsed))]
    return {"project_id": pid, "ok": all(r.get("ok") for r in results), "results": results}


__all__ = ["fill_empty_pages", "PageFill"]
