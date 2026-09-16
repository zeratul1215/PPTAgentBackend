"""Agent tools for discovering and capturing resources from the active deck."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from agent_backend.agent.tools.context import (
    page_lock, require_agent_run_id, require_project_id, require_session_id,
    require_user_id, workspace_for,
)
from agent_backend.workspace import pageorder, repo
from agent_backend.workspace.assets import materialize_pptist_slide_assets, page_asset_manifest
from agent_backend.workspace.paths import read_json
from agent_backend.workspace.session_resources import delete_session_resource_dir, resource_path, write_resource_files


def _position_text(element: dict[str, Any] | None) -> str:
    if not element:
        return "位置未记录"
    try:
        return "x={:.0f}, y={:.0f}, w={:.0f}, h={:.0f}".format(
            float(element.get("left") or 0), float(element.get("top") or 0),
            float(element.get("width") or 0), float(element.get("height") or 0),
        )
    except Exception:
        return "位置未记录"


def _message_context(runtime: ToolRuntime, session_id: str, user_id: str) -> tuple[str | None, int | None]:
    context = runtime.context or {}
    message_id = context.get("current_message_id") if isinstance(context, dict) else getattr(context, "current_message_id", None)
    message_id = str(message_id or "") or None
    return message_id, repo.get_message_seq(message_id=message_id or "", session_id=session_id, user_id=user_id)


@tool
def list_page_resources(page_ref: str, runtime: ToolRuntime) -> dict[str, Any]:
    """List capturable image and table elements on a page."""
    pid = require_project_id(runtime)
    paths = workspace_for(pid)
    slot = pageorder.slot_for_page_ref(paths, str(page_ref))
    if slot is None:
        return {"ok": False, "error": "page_not_found"}
    with page_lock(pid, int(slot)):
        slide_path = paths.pptist_slide_json(int(slot))
        slide = read_json(slide_path) if slide_path.exists() else {}
        elements = slide.get("elements") if isinstance(slide, dict) else []
        items = []
        for element in elements if isinstance(elements, list) else []:
            if not isinstance(element, dict) or not element.get("id"):
                continue
            kind = str(element.get("type") or "").lower()
            if kind in {"image", "table"}:
                items.append({
                    "element_id": str(element["id"]), "kind": kind,
                    "position": _position_text(element),
                    "text_preview": str(element.get("text") or "")[:200] if kind == "table" else "",
                })
        return {"ok": True, "page": pageorder.position_for_slot(paths, int(slot)), "page_ref": pageorder.page_ref_for_slot(int(slot)), "resources": items, "whole_page": {"kind": "page", "description": "当前页面的结构和视觉快照"}}


def _page_png(paths, slot: int) -> Path | None:
    for path in (paths.reread_page_png(slot), paths.page_png(slot)):
        if path.is_file():
            return path
    return None


def _capture_payload(paths, slot: int, kind: str, element_id: str | None) -> list[tuple[str, bytes, str]]:
    slide_path = paths.pptist_slide_json(slot)
    slide = read_json(slide_path) if slide_path.exists() else {}
    elements = slide.get("elements") if isinstance(slide, dict) else []
    element = next((e for e in elements if isinstance(e, dict) and str(e.get("id")) == str(element_id)), None) if isinstance(elements, list) else None
    if kind == "image":
        if not element:
            raise ValueError("element_not_found")
        manifest_path = page_asset_manifest(paths, slot)
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        image = next((x for x in manifest.get("images", []) if str(x.get("element_id")) == str(element_id)), None)
        if not image:
            manifest = materialize_pptist_slide_assets(paths, slot, slide)
            image = next((x for x in manifest.get("images", []) if str(x.get("element_id")) == str(element_id)), None)
        source = Path(str((image or {}).get("path") or ""))
        if not source.is_file():
            raise ValueError("image_asset_not_found")
        return [(f"original{source.suffix or '.bin'}", source.read_bytes(), str((image or {}).get("mime") or "application/octet-stream"))]
    if kind == "table":
        if not element:
            raise ValueError("element_not_found")
        return [("structure.json", json.dumps(element, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")]
    if kind == "page":
        if not slide:
            raise ValueError("slide_not_found")
        return [("structure.json", json.dumps(slide, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")]
    raise ValueError("unsupported_resource_kind")


@tool
def capture_page_resources(resources: list[dict[str, Any]], runtime: ToolRuntime) -> dict[str, Any]:
    """Copy selected page elements into immutable resources for this session."""
    pid = require_project_id(runtime)
    sid = require_session_id(runtime)
    uid = require_user_id(runtime)
    run_id = require_agent_run_id(runtime)
    message_id, message_seq = _message_context(runtime, sid, uid)
    user_request = repo.get_chat_message_content(message_id=message_id or "", session_id=sid, user_id=uid)
    output: list[dict[str, Any]] = []
    for request in resources or []:
        if not isinstance(request, dict):
            continue
        page_ref = str(request.get("page_ref") or "")
        paths = workspace_for(pid)
        slot = pageorder.slot_for_page_ref(paths, page_ref)
        if slot is None:
            output.append({"ok": False, "error": "page_not_found", "page_ref": page_ref})
            continue
        element_id = str(request.get("element_id") or "") or None
        kind = str(request.get("kind") or ("page" if not element_id else "image"))
        with page_lock(pid, int(slot)):
            try:
                payloads = _capture_payload(paths, int(slot), kind, element_id)
                preview = _page_png(paths, int(slot))
                if preview:
                    payloads.append(("preview.png", preview.read_bytes(), "image/png"))
                ref = f"res_{uuid.uuid4().hex[:20]}"
                written = write_resource_files(user_id=uid, session_id=sid, resource_ref=ref, files=[(n, d) for n, d, _ in payloads])
                file_rows = []
                digest = hashlib.sha256()
                for name, _data, mime in payloads:
                    _path, sha, size = written[name]
                    role = "preview" if name == "preview.png" else ("structure" if name == "structure.json" else "original")
                    file_rows.append({"role": role, "relative_path": name, "mime_type": mime, "byte_size": size, "sha256": sha})
                canonical = next(
                    (data for name, data, _mime in payloads if name in {"original.png", "original.jpg", "original.jpeg", "original.webp", "structure.json"}),
                    payloads[0][1],
                )
                digest.update(canonical)
                position = pageorder.position_for_slot(paths, int(slot))
                slide = read_json(paths.pptist_slide_json(int(slot)))
                elements = slide.get("elements") if isinstance(slide, dict) else []
                element = next((e for e in elements if isinstance(e, dict) and str(e.get("id")) == element_id), None) if isinstance(elements, list) and element_id else None
                description = str(request.get("description") or f"从当前文档第{position}页捕获的{kind}资源，捕获时元素位置为 {_position_text(element)}。")
                if user_request:
                    description = f"{description} 用户本轮用途说明：{user_request}"
                source_locator = {"project_id": pid, "slot": int(slot), "element_id": element_id}
                canonical_kind = kind if kind in {"image", "table", "page"} else "file"
                existing = repo.find_session_resource_by_content(session_id=sid, user_id=uid, kind=canonical_kind, content_hash=digest.hexdigest())
                if existing:
                    delete_session_resource_dir(uid, sid, ref)
                    ref = str(existing["resource_ref"])
                    repo.update_session_resource_capture(resource_ref=ref, session_id=sid, user_id=uid, source_locator=source_locator, source_kind="deck_element")
                    repo.touch_session_resources(
                        resource_refs=[ref], session_id=sid, user_id=uid, relation="captured",
                        agent_run_id=run_id, message_id=message_id, message_seq=message_seq,
                        details={"source_locator": source_locator},
                    )
                    output.append({"ok": True, "resource_ref": ref, "kind": canonical_kind, "description": existing.get("description") or description, "reused": True})
                    continue
                repo.create_session_resource(resource_ref=ref, session_id=sid, user_id=uid, kind=canonical_kind, source_kind="deck_element", description=description, description_status="ready", status="ready", content_hash=digest.hexdigest(), source_locator=source_locator, created_message_id=message_id, created_run_id=run_id, created_seq=message_seq, files=file_rows)
                repo.touch_session_resources(
                    resource_refs=[ref], session_id=sid, user_id=uid, relation="captured",
                    agent_run_id=run_id, message_id=message_id, message_seq=message_seq,
                    details={"source_locator": source_locator},
                )
                from agent_backend.workspace.resource_processing import describe_and_embed_resource
                threading.Thread(target=describe_and_embed_resource, kwargs={"resource_ref": ref, "session_id": sid, "user_id": uid, "origin": "从当前文档捕获的资源"}, daemon=True, name=f"resource-describe-{ref[-8:]}" ).start()
                output.append({"ok": True, "resource_ref": ref, "kind": kind, "description": description})
            except Exception as exc:
                output.append({"ok": False, "error": f"capture_failed: {type(exc).__name__}: {exc}", "page_ref": page_ref})
    return {"ok": bool(output) and all(item.get("ok") for item in output), "resources": output}


__all__ = ["list_page_resources", "capture_page_resources"]
