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
from .table_spec import iter_cells, validate_table_spec
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from lxml import html
from typing_extensions import NotRequired, TypedDict

from agent_backend.agent.models import agent_model_name, build_chat_model
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
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.reread import (
    _load_slide_for_reread,
)
from agent_backend.agent.tools.deck_style import public_style_row, require_ready_style
from agent_backend.workspace import pageorder
from agent_backend.workspace.paths import next_turn_dir, read_json, write_json, write_text
from agent_backend.workspace.repo import record_turn
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.step_2_plan.skills import table_calc


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
_TABLE_SPEC_FIELDS = {"data", "colWidths", "cellMinHeight", "outline", "theme"}
_TABLE_CELL_STYLE_KEYS = {"bold", "em", "underline", "strikethrough", "color", "backcolor", "fontname", "align", "vAlign"}
_TABLE_OUTLINE_KEYS = {"width", "style", "color"}
_TABLE_THEME_KEYS = {"color", "rowHeader", "rowFooter", "colHeader", "colFooter"}


def _visible_table_cells(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index visible cells by id without mutating the validator's deep copy."""
    visible: dict[str, dict[str, Any]] = {}
    data = table.get("data") if isinstance(table.get("data"), list) else []
    occupied: set[tuple[int, int]] = set()
    for r, row in enumerate(data):
        if not isinstance(row, list): continue
        for c, cell in enumerate(row):
            if not isinstance(cell, dict) or (r, c) in occupied: continue
            rs, cs = int(cell.get("rowspan", 1) or 1), int(cell.get("colspan", 1) or 1)
            for rr in range(r, r + rs):
                for cc in range(c, c + cs): occupied.add((rr, cc))
            cid = str(cell.get("id") or "")
            if cid: visible[cid] = cell
    return visible


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
    if etype == "table":
        try:
            table = validate_table_spec(element)
            cells = [
                {
                    "cellId": cell["id"],
                    "row": r,
                    "col": c,
                    "text": cell["text"],
                    "rowspan": cell["rowspan"],
                    "colspan": cell["colspan"],
                }
                for r, c, cell in iter_cells(table, include_placeholders=False)
            ]
            return {"elementId": element_id, "type": "table", "cells": cells, "allowedOps": ["replace", "style_change", "delete"]}
        except ValueError:
            return {"elementId": element_id, "type": "table", "allowedOps": []}
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
        "description": "page canvas background; supports backgroundColor style changes",
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
            "Use only replace, style_change, delete, or calculate_and_replace.",
            "Never add, move, resize, rotate, crop, fit, or change font size.",
            "replace text by paragraphId; replace image only with an available assetId.",
            "For a table, replace cell text with {\"op\":\"replace\",\"elementId\":\"table-id\",\"cells\":[{\"cellId\":\"cell-id\",\"text\":\"new text\"}]}; never use delete to clear a cell.",
            "For table styles, use scope cells with cellIds or scope table with outline/theme; never change table geometry, rows, columns, spans, or font size.",
            "For numeric calculations use calculate_and_replace with op sum|mean|min|max|count, sourceCellIds/sourceElementIds and one targetCellId/targetElementId; never change table geometry.",
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
                {"op": "replace", "elementId": "table-id", "cells": [{"cellId": "cell-id", "text": "new text"}]},
                {"op": "style_change", "elementId": "table-id", "scope": "cells", "cellIds": ["cell-id"], "changes": {"color": "#..."}},
                {"op": "style_change", "elementId": "table-id", "scope": "table", "changes": {"outline": {}, "theme": {}}},
                {"op": "style_change", "elementId": "id", "scope": "text|element", "paragraphIds": ["p0"], "changes": {"color": "#..."}},
                {"op": "style_change", "elementId": "$page", "scope": "element", "changes": {"backgroundColor": "#..."}},
                {"op": "calculate_and_replace", "opName": "sum", "sourceCellIds": ["cell-a"], "targetCellId": "cell-b"},
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
    if etype == "table":
        changes = op.get("cells")
        if not isinstance(changes, list) or not changes:
            raise ValueError("table replacement requires cells")
        table = validate_table_spec(element)
        by_id = _visible_table_cells(table)
        for item in changes:
            cid = str(item.get("cellId") or "") if isinstance(item, dict) else ""
            if cid not in by_id: raise ValueError("table replacement references an unknown cell")
            if not isinstance(item, dict) or "text" not in item:
                raise ValueError("table replacement requires explicit cell text")
            by_id[cid]["text"] = str(item.get("text") or "")
        element.update({key: table[key] for key in _TABLE_SPEC_FIELDS})
        return
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
    if etype == "table":
        table = validate_table_spec(element)
        if scope == "cells":
            ids = {str(x) for x in op.get("cellIds") or []}
            visible = _visible_table_cells(table)
            if not ids or ids - set(visible): raise ValueError("table style_change references unknown or hidden cells")
            if set(changes) - _TABLE_CELL_STYLE_KEYS: raise ValueError("table cell style contains unsupported fields")
            for cid in ids: visible[cid]["style"].update(changes)
        elif scope == "table":
            if set(changes) - {"outline", "theme"}: raise ValueError("table style contains unsupported fields")
            if isinstance(changes.get("outline"), dict):
                if set(changes["outline"]) - _TABLE_OUTLINE_KEYS: raise ValueError("table outline contains unsupported fields")
                table["outline"].update(changes["outline"])
            if isinstance(changes.get("theme"), dict):
                if set(changes["theme"]) - _TABLE_THEME_KEYS: raise ValueError("table theme contains unsupported fields")
                table["theme"].update(changes["theme"])
        else: raise ValueError("table style_change scope must be cells or table")
        element.update({key: table[key] for key in _TABLE_SPEC_FIELDS})
        return
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


def _apply_calculate_and_replace(slide: dict[str, Any], op: dict[str, Any]) -> None:
    operator = str(op.get("opName") or op.get("operator") or "sum")
    if operator not in set(table_calc.operator_names()):
        raise ValueError("unsupported calculation operator")
    sources = {str(x) for x in (op.get("sourceCellIds") or []) if str(x)}
    target_id = str(op.get("targetCellId") or "")
    if not sources or not target_id:
        raise ValueError("calculation requires sourceCellIds and targetCellId")
    target = None
    values: list[float] = []
    for element in slide.get("elements") or []:
        if not isinstance(element, dict): continue
        if element.get("type") == "table":
            table = validate_table_spec(element)
            visible = _visible_table_cells(element)
            for cid in sources:
                if cid in visible:
                    parsed = table_calc.parse_number(str(visible[cid].get("text") or ""))
                    if parsed: values.append(parsed[0])
            if target_id in visible: target = visible[target_id]
        elif element.get("type") == "text" and str(element.get("id")) in {str(x) for x in (op.get("sourceElementIds") or [])}:
            parsed = table_calc.parse_number(str(element.get("content") or ""))
            if parsed: values.append(parsed[0])
        elif element.get("type") == "text" and str(element.get("id")) == target_id:
            target = element
    if target is None: raise ValueError("calculation target does not exist")
    if not values: raise ValueError("calculation has no numeric sources")
    value = table_calc.format_number(table_calc.reduce_values(operator, values) or 0)
    if "rowspan" in target or "colspan" in target:
        target["text"] = value
    elif target.get("type") == "text":
        target["content"] = value
    else:
        raise ValueError("calculation target must be a text or table cell")


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
    original_tables = {
        str(element.get("id")): copy.deepcopy(element)
        for element in slide.get("elements") or []
        if isinstance(element, dict) and element.get("type") == "table" and element.get("id")
    }
    for op in operations:
        kind = op.get("op")
        if kind == "replace":
            _apply_replace(updated, op, assets)
        elif kind == "style_change":
            _apply_style_change(updated, op)
        elif kind == "calculate_and_replace":
            _apply_calculate_and_replace(updated, op)
        elif kind == "delete":
            _apply_delete(updated, op)
        else:
            raise ValueError(f"unsupported patch operation: {kind!r}")
    if updated == slide:
        raise ValueError("patch operations produced no change")
    updated_by_id = _element_index(updated)
    for element_id, original in original_tables.items():
        current = updated_by_id.get(element_id)
        if current is None:
            if not any(op.get("op") == "delete" and str(op.get("elementId") or "") == element_id for op in operations):
                raise ValueError(f"table element disappeared during patch: {element_id}")
            continue
        for key in set(original) | set(current):
            if key in _TABLE_SPEC_FIELDS:
                continue
            if current.get(key) != original.get(key):
                raise ValueError(f"table element envelope changed during patch: {element_id}.{key}")
    return updated


@tool
def patch_pages(patches: list[PagePatch], runtime: ToolRuntime, images_involved: bool = False) -> dict[str, Any]:
    """Apply layout-preserving edits directly to existing PPTist JSON pages.

    This tool mutates supported properties of existing elements without adding,
    moving, resizing, or reflowing elements and without changing page composition.
    Its supported operations are existing-content replacement, staged in-place
    image replacement, supported non-font-size style changes, deterministic
    calculation into an existing target, and complete-element deletion. `patches`
    uses stable page refs returned by page-selection tools.
    """
    pid = require_project_id(runtime)
    run_id = require_agent_run_id(runtime)
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

    assert_unique_page_jobs([int(item[1]) for item in parsed if item[1] is not None])
    if style_needed:
        latest = public_style_row(pid)
        if latest.get("status") != "ready" or not isinstance(latest.get("style_json"), dict):
            raise RuntimeError("deck style became unavailable before patching")
        deck_style_row = latest

    def _run_one(index: int) -> dict[str, Any]:
        page, slot, demand, use_deck_style = parsed[index]
        if slot is None:
            return {"page": page, "ok": False, "status": "page_not_found"}
        with page_job(pid, int(slot)):
            current_page = pageorder.position_for_slot(paths, int(slot))
            if current_page is None:
                return {"page": page, "slot": int(slot), "ok": False, "status": "page_not_found"}
            page = int(current_page)
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
            emit(pid, {"type": "task_started", "page": page, "slot": slot, "demand": demand, "index": index, "agent_run_id": run_id})
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
                emit(pid, {"type": "task_finished", "page": page, "slot": slot, "demand": demand, "index": index, "turn_dir": str(turn_dir), "errors": [], "prepare_reread_png": True, "agent_run_id": run_id})
                # The planner output and concrete mutator operations are audit
                # artifacts, not conversational tool output. Keeping this
                # response small prevents the outer agent from echoing element
                # ids or operation JSON to the user.
                return {"page": page, "ok": True, "status": "patched"}
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
                emit(pid, {"type": "task_failed", "page": page, "slot": slot, "demand": demand, "index": index, "turn_dir": str(turn_dir), "error": error, "agent_run_id": run_id})
                # Keep the detailed exception in this turn's error.json. The
                # outer agent only needs a stable failure state to respond.
                return {"page": page, "ok": False, "status": "patch_failed"}

    results_by_index = execute_page_jobs(pid, [(i, int(item[1])) for i, item in enumerate(parsed)], _run_one)
    results = [results_by_index[i] for i in range(len(parsed))]
    return {"project_id": pid, "ok": all(item.get("ok") for item in results), "results": results}


__all__ = ["patch_pages", "build_resource_list", "apply_operations"]
