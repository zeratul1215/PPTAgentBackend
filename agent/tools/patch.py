"""Lightweight, layout-preserving PPTist JSON patch tool.

This module deliberately does not reuse the HTML editing pipeline. The planner
sees a compact projection plus the current page render; deterministic mutators
then edit the authoritative PPTist slide JSON in place while preserving every
unrelated field.
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import re
import time
import traceback
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from lxml import html
from typing_extensions import NotRequired, TypedDict

from agent_backend.agent.models import agent_model_name, build_chat_model
from agent_backend.agent.tools.context import (
    emit,
    project_lock,
    require_project_id,
    require_session_id,
    workspace_for,
)
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.reread import (
    _load_slide_for_reread,
)
from agent_backend.agent.tools.deck_style import public_style_row, require_ready_style
from agent_backend.workspace import pageorder
from agent_backend.workspace.paths import next_turn_dir, read_json, write_json, write_text
from agent_backend.workspace.repo import record_turn


class PagePatch(TypedDict):
    page_ref: str
    demand: str
    use_deck_style: NotRequired[bool]


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.I | re.S)
_TEXT_STYLE_KEYS = {
    "fontFamily": "font-family",
    "color": "color",
    "bold": "font-weight",
    "italic": "font-style",
    "underline": "text-decoration",
    "strikethrough": "text-decoration",
}
_TEXT_ELEMENT_STYLE_KEYS = {
    "fill": "fill",
    "outline": "outline",
    "shadow": "shadow",
    "opacity": "opacity",
    "lineHeight": "lineHeight",
    "wordSpace": "wordSpace",
    "paragraphSpace": "paragraphSpace",
    "inset": "inset",
    "defaultFontName": "defaultFontName",
    "defaultColor": "defaultColor",
}
_SHAPE_ELEMENT_STYLE_KEYS = {"fill", "gradient", "outline", "shadow", "opacity"}
_IMAGE_STYLE_KEYS = {"radius", "opacity", "filters", "outline", "shadow"}
_PAGE_ELEMENT_ID = "$page"
_PAGE_STYLE_KEYS = {"backgroundColor", "fill", "color"}


def _content_of(element: dict[str, Any]) -> str | None:
    if element.get("type") == "text":
        return element.get("content") if isinstance(element.get("content"), str) else None
    if element.get("type") == "shape":
        text = element.get("text")
        if isinstance(text, dict) and isinstance(text.get("content"), str):
            return text["content"]
    return None


def _document(content: str) -> html.HtmlElement:
    root = html.Element("div")
    try:
        fragments = html.fragments_fromstring(content or "")
    except (ValueError, TypeError):
        fragments = []
    for fragment in fragments:
        if isinstance(fragment, str):
            p = html.Element("p")
            p.text = fragment
            root.append(p)
        else:
            root.append(fragment)
    if not root.xpath("./p"):
        p = html.Element("p")
        p.text = root.text_content() if len(root) else (root.text or "")
        root.clear()
        root.append(p)
    return root


def _serialize_document(root: html.HtmlElement) -> str:
    return "".join(html.tostring(child, encoding="unicode") for child in root)


def _paragraphs(content: str) -> list[dict[str, str]]:
    root = _document(content)
    return [
        {"paragraphId": f"p{idx}", "text": " ".join(p.text_content().split())}
        for idx, p in enumerate(root.xpath("./p"))
    ]


def _resource_item(element: dict[str, Any]) -> dict[str, Any] | None:
    element_id = element.get("id")
    etype = element.get("type")
    if not isinstance(element_id, str) or not element_id:
        return None
    if etype == "text":
        return {
            "elementId": element_id,
            "type": "text",
            "paragraphs": _paragraphs(str(element.get("content") or "")),
            "allowedOps": ["replace", "style_change", "delete"],
        }
    if etype == "image":
        return {"elementId": element_id, "type": "image", "allowedOps": ["replace", "style_change", "delete"]}
    if etype == "shape":
        text = element.get("text")
        item: dict[str, Any] = {"elementId": element_id, "type": "shape", "allowedOps": ["delete"]}
        if isinstance(text, dict) and isinstance(text.get("content"), str):
            item["paragraphs"] = _paragraphs(text["content"])
            item["allowedOps"] = ["replace", "style_change", "delete"]
        return item
    # Unsupported objects remain visible only as opaque page content. This keeps
    # the Planner from inventing an operation for line/table/chart/etc.
    return {"elementId": element_id, "type": str(etype or "unknown"), "allowedOps": []}


def _slide_background_resource(slide: dict[str, Any]) -> dict[str, Any]:
    background = slide.get("background")
    color = "#ffffff"
    if isinstance(background, dict) and isinstance(background.get("color"), str) and background["color"].strip():
        color = background["color"].strip()
    return {
        "elementId": _PAGE_ELEMENT_ID,
        "type": "page",
        "description": "page canvas background; use this for requests that change the whole slide/page background",
        "background": {"type": "solid", "color": color},
        "allowedOps": ["style_change"],
    }


def build_resource_list(slide: dict[str, Any]) -> list[dict[str, Any]]:
    elements = slide.get("elements")
    if not isinstance(elements, list):
        return [_slide_background_resource(slide)]
    return [_slide_background_resource(slide)] + [
        item for el in elements if isinstance(el, dict) if (item := _resource_item(el)) is not None
    ]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _slide_size(slide: dict[str, Any]) -> tuple[float, float]:
    width = _number(slide.get("width")) or 1000.0
    height = _number(slide.get("height"))
    if height is None:
        for element in slide.get("elements") or []:
            if not isinstance(element, dict):
                continue
            left = _number(element.get("left")) or 0.0
            top = _number(element.get("top")) or 0.0
            el_width = _number(element.get("width")) or 0.0
            el_height = _number(element.get("height")) or 0.0
            if abs(left) <= 1.0 and abs(top) <= 1.0 and el_width >= width * 0.95 and el_height > 0:
                height = max(height or 0.0, top + el_height)
    return width, height or 562.5


def _is_full_canvas_shape(element: dict[str, Any], slide_width: float, slide_height: float) -> bool:
    if element.get("type") != "shape":
        return False
    left = _number(element.get("left"))
    top = _number(element.get("top"))
    width = _number(element.get("width"))
    height = _number(element.get("height"))
    if left is None or top is None or width is None or height is None:
        return False
    tolerance = max(slide_width, slide_height) * 0.02
    return (
        abs(left) <= tolerance
        and abs(top) <= tolerance
        and abs((left + width) - slide_width) <= tolerance
        and abs((top + height) - slide_height) <= tolerance
    )


def _page_background_color(changes: dict[str, Any]) -> str | None:
    for key in ("backgroundColor", "fill", "color"):
        value = changes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _apply_page_style_change(slide: dict[str, Any], changes: dict[str, Any]) -> None:
    if not changes or set(changes) - _PAGE_STYLE_KEYS:
        raise ValueError("page style_change contains unsupported fields")
    color = _page_background_color(changes)
    if color is None:
        raise ValueError("page style_change requires a background color")

    background = slide.get("background")
    if not isinstance(background, dict):
        background = {}
        slide["background"] = background
    background.clear()
    background.update({"type": "solid", "color": color})

    slide_width, slide_height = _slide_size(slide)
    for element in slide.get("elements") or []:
        if isinstance(element, dict) and _is_full_canvas_shape(element, slide_width, slide_height):
            element["fill"] = color
            element.pop("gradient", None)


def _parse_json(content: Any) -> Any:
    raw = content if isinstance(content, str) else str(content or "")
    match = _JSON_BLOCK.search(raw)
    if match:
        raw = match.group(1)
    return json.loads(raw.strip())


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _slide_images(slide: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for element in slide.get("elements") or []:
        if not isinstance(element, dict) or element.get("type") != "image":
            continue
        element_id, src = element.get("id"), element.get("src")
        if isinstance(element_id, str) and isinstance(src, str) and src:
            out.append((element_id, src))
    return out


def _describe_images(slide: dict[str, Any]) -> dict[str, str]:
    images = _slide_images(slide)
    if not images:
        return {}
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "Describe each image briefly in order. Return JSON only: {\"images\":[{\"elementId\":\"...\",\"description\":\"...\"}]}."}
    ]
    ids: list[str] = []
    for element_id, src in images:
        if not src.startswith("data:image/"):
            continue
        ids.append(element_id)
        parts.append({"type": "image_url", "image_url": {"url": src}})
    if not ids:
        return {}
    response = build_chat_model().invoke([HumanMessage(content=parts)])
    data = _parse_json(response.content)
    described = data.get("images") if isinstance(data, dict) else None
    return {
        str(item.get("elementId")): str(item.get("description") or "").strip()
        for item in (described or [])
        if isinstance(item, dict) and str(item.get("elementId")) in ids and str(item.get("description") or "").strip()
    }


def _pending_assets(paths, slot: int) -> tuple[list[dict[str, str]], Path]:
    manifest = paths.pending_uploads_json(slot)
    uploads: list[dict[str, Any]] = []
    if manifest.exists():
        try:
            payload = read_json(manifest)
            uploads = payload.get("uploads") if isinstance(payload, dict) else []
        except Exception:
            uploads = []
    assets: list[dict[str, str]] = []
    for index, item in enumerate(uploads if isinstance(uploads, list) else []):
        name = str(item.get("filename") or "") if isinstance(item, dict) else ""
        path = paths.page_assets_dir(slot) / "uploads" / name
        if name and path.is_file():
            assets.append({"assetId": f"upload_{index}", "filename": name})
    return assets, manifest


def _plan_operations(
    *,
    demand: str,
    resources: list[dict[str, Any]],
    page_png: Path | None,
    image_descriptions: dict[str, str],
    assets: list[dict[str, str]],
    deck_style: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    resource_copy = copy.deepcopy(resources)
    for item in resource_copy:
        if item.get("type") == "image" and item.get("elementId") in image_descriptions:
            item["description"] = image_descriptions[item["elementId"]]
    prompt = {
        "role": "You are a PPT patch planner. Return JSON only.",
        "user_request": demand,
        "rules": [
            "Use only replace, style_change, delete.",
            "Never add, move, resize, rotate, crop, fit, or change font size.",
            "replace text by paragraphId; replace image only with an available assetId.",
            "style_change must use only existing elementId and supported styles.",
            "For whole-page/slide background color changes, target elementId '$page' with scope 'element'.",
            "Do not create operations for items with empty allowedOps.",
        ],
        "resources": resource_copy,
        "available_assets": [{"assetId": item["assetId"]} for item in assets],
        "deck_style": deck_style if isinstance(deck_style, dict) else None,
        "deck_style_rules": [
            "If deck_style is provided, use its colors/fonts only for style_change operations that the user explicitly requested to align with the document/current style.",
            "Do not change layout, element bounding boxes, reading order, or add elements because of deck_style.",
        ] if isinstance(deck_style, dict) else [],
        "schema": {
            "operations": [
                {"op": "replace", "elementId": "id", "paragraphs": [{"paragraphId": "p0", "text": "new text"}]},
                {"op": "replace", "elementId": "image-id", "assetId": "upload_0"},
                {"op": "style_change", "elementId": "id", "scope": "text|element", "paragraphIds": ["p0"], "changes": {"color": "#..."}},
                {"op": "style_change", "elementId": "$page", "scope": "element", "changes": {"backgroundColor": "#..."}},
                {"op": "delete", "elementId": "id"},
            ]
        },
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
    if page_png is not None and page_png.exists():
        content.append({"type": "image_url", "image_url": {"url": _data_url(page_png)}})
    response = build_chat_model().invoke([HumanMessage(content=content)])
    raw_response = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False, default=str)
    parsed = _parse_json(raw_response)
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    return [op for op in (operations or []) if isinstance(op, dict)], prompt, raw_response


def _element_index(slide: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(el.get("id")): el for el in slide.get("elements") or [] if isinstance(el, dict) and isinstance(el.get("id"), str)}


def _style_dict(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in raw.split(";"):
        key, sep, value = pair.partition(":")
        if sep and key.strip() and value.strip():
            result[key.strip().lower()] = value.strip()
    return result


def _set_style(node: html.HtmlElement, changes: dict[str, Any]) -> None:
    style = _style_dict(node.get("style") or "")
    for key, css in _TEXT_STYLE_KEYS.items():
        if key not in changes:
            continue
        value = changes[key]
        if key == "bold":
            style[css] = "bold" if value else "normal"
        elif key == "italic":
            style[css] = "italic" if value else "normal"
        elif key == "underline":
            style[css] = "underline" if value else "none"
        elif key == "strikethrough":
            style[css] = "line-through" if value else "none"
        elif isinstance(value, str) and value.strip():
            style[css] = value.strip()
    if style:
        node.set("style", ";".join(f"{key}:{value}" for key, value in style.items()) + ";")


def _baseline_span(paragraph: html.HtmlElement) -> html.HtmlElement:
    for node in paragraph.iterdescendants():
        if isinstance(node.tag, str) and (node.text or "").strip():
            span = html.Element("span")
            if node.get("style"):
                span.set("style", node.get("style"))
            if node.tag == "strong":
                _set_style(span, {"bold": True})
            if node.tag == "em":
                _set_style(span, {"italic": True})
            return span
    return html.Element("span")


def _replace_paragraphs(content: str, replacements: dict[str, str]) -> str:
    root = _document(content)
    paragraphs = root.xpath("./p")
    for index, paragraph in enumerate(paragraphs):
        replacement = replacements.get(f"p{index}")
        if replacement is None:
            continue
        attrs = dict(paragraph.attrib)
        baseline = _baseline_span(paragraph)
        paragraph.clear()
        paragraph.attrib.update(attrs)
        baseline.text = replacement
        paragraph.append(baseline)
    return _serialize_document(root)


def _apply_text_style(content: str, paragraph_ids: list[str], changes: dict[str, Any]) -> str:
    root = _document(content)
    targets = set(paragraph_ids)
    for index, paragraph in enumerate(root.xpath("./p")):
        if targets and f"p{index}" not in targets:
            continue
        if "textAlign" in changes and isinstance(changes["textAlign"], str):
            style = _style_dict(paragraph.get("style") or "")
            style["text-align"] = changes["textAlign"]
            paragraph.set("style", ";".join(f"{k}:{v}" for k, v in style.items()) + ";")
        _set_style(paragraph, changes)
        for node in paragraph.iterdescendants():
            if isinstance(node.tag, str):
                _set_style(node, changes)
    return _serialize_document(root)


def _apply_replace(slide: dict[str, Any], op: dict[str, Any], assets: dict[str, Path]) -> None:
    element = _element_index(slide).get(str(op.get("elementId") or ""))
    if element is None:
        raise ValueError("replace target does not exist")
    etype = element.get("type")
    if etype == "image":
        asset_id = str(op.get("assetId") or "")
        path = assets.get(asset_id)
        if path is None:
            raise ValueError("image replacement requires an available assetId")
        element["src"] = _data_url(path)
        return
    content = _content_of(element)
    raw_paragraphs = op.get("paragraphs")
    if content is None or not isinstance(raw_paragraphs, list):
        raise ValueError("text replacement requires paragraphs on a text-bearing element")
    replacements = {
        str(item.get("paragraphId")): str(item.get("text") or "")
        for item in raw_paragraphs
        if isinstance(item, dict) and str(item.get("paragraphId") or "")
    }
    if not replacements:
        raise ValueError("text replacement has no valid paragraphs")
    valid_ids = {f"p{index}" for index, _ in enumerate(_paragraphs(content))}
    unknown_ids = set(replacements) - valid_ids
    if unknown_ids:
        raise ValueError(f"text replacement references unknown paragraphs: {sorted(unknown_ids)}")
    updated = _replace_paragraphs(content, replacements)
    if etype == "text":
        element["content"] = updated
    else:
        element["text"]["content"] = updated


def _apply_style_change(slide: dict[str, Any], op: dict[str, Any]) -> None:
    if str(op.get("elementId") or "") == _PAGE_ELEMENT_ID:
        changes = op.get("changes")
        forbidden = {"fontsize", "font-size"}
        if not isinstance(changes, dict) or any(str(key).lower() in forbidden for key in changes):
            raise ValueError("style_change requires allowed changes and may not change fontSize")
        _apply_page_style_change(slide, changes)
        return
    element = _element_index(slide).get(str(op.get("elementId") or ""))
    if element is None:
        raise ValueError("style_change target does not exist")
    changes = op.get("changes")
    forbidden = {"fontsize", "font-size"}
    if not isinstance(changes, dict) or any(str(key).lower() in forbidden for key in changes):
        raise ValueError("style_change requires allowed changes and may not change fontSize")
    scope = str(op.get("scope") or "element")
    etype = element.get("type")
    if scope == "text":
        content = _content_of(element)
        if content is None:
            raise ValueError("text scope requires a text-bearing element")
        paragraph_ids = [str(v) for v in op.get("paragraphIds") or [] if isinstance(v, str)]
        allowed = set(_TEXT_STYLE_KEYS) | {"textAlign"}
        if not changes or set(changes) - allowed:
            raise ValueError("text style_change contains unsupported fields")
        valid_ids = {f"p{index}" for index, _ in enumerate(_paragraphs(content))}
        unknown_ids = set(paragraph_ids) - valid_ids
        if unknown_ids:
            raise ValueError(f"text style_change references unknown paragraphs: {sorted(unknown_ids)}")
        updated = _apply_text_style(content, paragraph_ids, changes)
        if etype == "text":
            element["content"] = updated
        else:
            element["text"]["content"] = updated
        return
    if etype == "text":
        if not changes or set(changes) - set(_TEXT_ELEMENT_STYLE_KEYS):
            raise ValueError("text element style_change contains unsupported fields")
        for external, internal in _TEXT_ELEMENT_STYLE_KEYS.items():
            if external in changes:
                element[internal] = changes[external]
        return
    if etype == "shape":
        if not changes or set(changes) - _SHAPE_ELEMENT_STYLE_KEYS:
            raise ValueError("shape style_change contains unsupported fields")
        for key in _SHAPE_ELEMENT_STYLE_KEYS:
            if key in changes:
                element[key] = changes[key]
        return
    if etype == "image":
        if not changes or set(changes) - _IMAGE_STYLE_KEYS:
            raise ValueError("image style_change contains unsupported fields")
        for key in _IMAGE_STYLE_KEYS:
            if key in changes:
                element[key] = changes[key]
        return
    raise ValueError("style_change type is not supported")


def _apply_delete(slide: dict[str, Any], op: dict[str, Any]) -> None:
    element_id = str(op.get("elementId") or "")
    elements = slide.get("elements")
    if not isinstance(elements, list) or not any(isinstance(el, dict) and el.get("id") == element_id for el in elements):
        raise ValueError("delete target does not exist")
    slide["elements"] = [el for el in elements if not (isinstance(el, dict) and el.get("id") == element_id)]
    animations = slide.get("animations")
    if isinstance(animations, list):
        slide["animations"] = [a for a in animations if not (isinstance(a, dict) and a.get("elId") == element_id)]


def apply_operations(slide: dict[str, Any], operations: list[dict[str, Any]], assets: dict[str, Path]) -> dict[str, Any]:
    updated = copy.deepcopy(slide)
    for op in operations:
        kind = op.get("op")
        if kind == "replace":
            _apply_replace(updated, op, assets)
        elif kind == "style_change":
            _apply_style_change(updated, op)
        elif kind == "delete":
            _apply_delete(updated, op)
        else:
            raise ValueError(f"unsupported patch operation: {kind!r}")
    return updated


@tool
def patch_pages(patches: list[PagePatch], runtime: ToolRuntime, images_involved: bool = False) -> dict[str, Any]:
    """Apply layout-preserving edits directly to existing PPTist JSON pages.

    Use this only when every requested final effect can be achieved without
    adding/moving/resizing/reflowing elements or changing the page composition.
    It supports replacing existing text, shape-contained text or an existing
    image with a user-uploaded staged asset; changing existing styles except font
    size; and deleting a complete element. Content-only translation, rewrite,
    proofreading, redaction, and deletion of existing text belong here even when
    a passage is split across multiple existing text elements. Use `edit_pages`
    only when the final result needs layout changes, new content space,
    reordering, or dependent changes to other elements. `patches` uses stable
    page refs returned by get_deck_outline/locate_pages/understand_pages.
    """
    pid = require_project_id(runtime)
    try:
        sid = require_session_id(runtime)
    except Exception:
        sid = None
    paths = workspace_for(pid)
    parsed: list[tuple[int, int | None, str, bool]] = []
    for item in patches or []:
        slot = pageorder.slot_for_page_ref(paths, str(item.get("page_ref") or ""))
        if slot is None:
            continue
        demand = str(item.get("demand") or "").strip()
        if demand:
            parsed.append((int(pageorder.position_for_slot(paths, slot) or 0), slot, demand, bool(item.get("use_deck_style"))))
    if not parsed:
        return {"project_id": pid, "ok": False, "error": "no valid patches"}

    style_needed = any(item[3] for item in parsed)
    deck_style_row = require_ready_style(pid, interrupt_when_unready=True) if style_needed else None

    results: list[dict[str, Any]] = []
    with project_lock(pid):
        if style_needed:
            latest = public_style_row(pid)
            if latest.get("status") != "ready" or not isinstance(latest.get("style_json"), dict):
                raise RuntimeError("deck style became unavailable before patching")
            deck_style_row = latest
        for index, (page, slot, demand, use_deck_style) in enumerate(parsed):
            if slot is None:
                results.append({"page": page, "ok": False, "status": "page_not_found"})
                continue
            turn_dir = next_turn_dir(paths, int(slot))
            started_at = time.time()
            manifest_data: dict[str, Any] = {
                "project_id": pid,
                "page_num": int(slot),
                "display_page": page,
                "demand": demand,
                "tool": "patch_pages",
                "images_involved": bool(images_involved),
                "use_deck_style": bool(use_deck_style),
                "deck_style_revision": int(deck_style_row.get("revision") or 0) if use_deck_style and isinstance(deck_style_row, dict) else 0,
                "model": agent_model_name(),
                "started_at": started_at,
                "status": "running",
            }
            write_json(turn_dir / "manifest.json", manifest_data)
            emit(pid, {"type": "task_started", "page": page, "slot": slot, "demand": demand, "index": index})
            try:
                write_json(
                    turn_dir / "reread_result.json",
                    {"performed": False, "reason": "patch_does_not_run_step1"},
                )
                slide = _load_slide_for_reread(paths, int(slot))
                write_json(turn_dir / "slide_before.json", slide)
                resources = build_resource_list(slide)
                assets, manifest = _pending_assets(paths, int(slot))
                asset_paths = {item["assetId"]: paths.page_assets_dir(int(slot)) / "uploads" / item["filename"] for item in assets}
                descriptions = _describe_images(slide) if images_involved else {}
                write_json(turn_dir / "resource_list.json", resources)
                write_json(turn_dir / "available_assets.json", [{"assetId": item["assetId"]} for item in assets])
                if images_involved:
                    write_json(turn_dir / "image_descriptions.json", descriptions)
                page_png = paths.reread_page_png(int(slot))
                if not page_png.exists():
                    page_png = paths.page_png(int(slot))
                operations, planner_request, planner_response = _plan_operations(
                    demand=demand,
                    resources=resources,
                    page_png=page_png if page_png.exists() else None,
                    image_descriptions=descriptions,
                    assets=assets,
                    deck_style=deck_style_row.get("style_json") if use_deck_style and isinstance(deck_style_row, dict) else None,
                )
                planner_request["page_image_attached"] = bool(page_png.exists())
                write_json(turn_dir / "planner_request.json", planner_request)
                write_text(turn_dir / "planner_response.txt", planner_response)
                write_json(turn_dir / "operations.json", {"operations": operations})
                if not operations:
                    raise ValueError("patch planner returned no valid operations")
                updated = apply_operations(slide, operations, asset_paths)
                write_json(turn_dir / "slide_after.json", updated)
                write_json(paths.pptist_slide_json(int(slot)), updated)
                from agent_backend.workspace.assets import materialize_pptist_slide_assets

                materialize_pptist_slide_assets(paths, int(slot), updated)
                used_assets = {
                    str(op.get("assetId"))
                    for op in operations
                    if op.get("op") == "replace" and op.get("assetId")
                }
                if manifest.exists() and used_assets:
                    try:
                        payload = read_json(manifest)
                        uploads = payload.get("uploads") if isinstance(payload, dict) else []
                        remaining = [
                            item
                            for upload_index, item in enumerate(uploads if isinstance(uploads, list) else [])
                            if f"upload_{upload_index}" not in used_assets
                        ]
                        write_json(manifest, {"schema_version": "pending_uploads_v1", "uploads": remaining})
                    except Exception:
                        # A stale manifest must not turn a committed JSON patch
                        # into a reported failure.
                        pass
                manifest_data.update({
                    "status": "ok",
                    "finished_at": time.time(),
                    "operations_count": len(operations),
                    "prepare_reread_png": True,
                })
                manifest_data["duration_seconds"] = manifest_data["finished_at"] - started_at
                write_json(turn_dir / "commit_result.json", {
                    "pptist_slide_json": str(paths.pptist_slide_json(int(slot))),
                    "prepare_reread_png": True,
                    "operations_count": len(operations),
                })
                write_json(turn_dir / "manifest.json", manifest_data)
                record_turn(
                    project_id=pid,
                    session_id=sid,
                    page_num=int(slot),
                    demand=demand,
                    ok=True,
                    error=None,
                    turn_dir=str(turn_dir),
                )
                emit(pid, {"type": "task_finished", "page": page, "slot": slot, "demand": demand, "index": index, "turn_dir": str(turn_dir), "errors": [], "prepare_reread_png": True})
                # The planner output and concrete mutator operations are audit
                # artifacts, not conversational tool output. Keeping this
                # response small prevents the outer agent from echoing element
                # ids or operation JSON to the user.
                results.append({"page": page, "ok": True, "status": "patched"})
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                manifest_data.update({
                    "status": "failed",
                    "finished_at": time.time(),
                    "error": error,
                })
                manifest_data["duration_seconds"] = manifest_data["finished_at"] - started_at
                write_json(turn_dir / "error.json", {"error": error, "traceback": traceback.format_exc()})
                write_json(turn_dir / "manifest.json", manifest_data)
                record_turn(
                    project_id=pid,
                    session_id=sid,
                    page_num=int(slot),
                    demand=demand,
                    ok=False,
                    error=error,
                    turn_dir=str(turn_dir),
                )
                emit(pid, {"type": "task_failed", "page": page, "slot": slot, "demand": demand, "index": index, "turn_dir": str(turn_dir), "error": error})
                # Keep the detailed exception in this turn's error.json. The
                # outer agent only needs a stable failure state to respond.
                results.append({"page": page, "ok": False, "status": "patch_failed"})
    return {"project_id": pid, "ok": all(item.get("ok") for item in results), "results": results}


__all__ = ["patch_pages", "build_resource_list", "apply_operations"]
