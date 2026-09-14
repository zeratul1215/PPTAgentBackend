"""Replace one existing native table matrix without appending a second table."""
from __future__ import annotations
from copy import deepcopy
import json
from typing import Any
from .base import Skill, SkillResult, _call_claude_json
from .....table_spec import validate_table_spec

_DOC = """Capability: replace one existing native table with a complete new
matrix. The output may change rows, columns, ordering, grouping, text, and spans.
It preserves the target table ID and envelope, and fixed code validates the full
replacement matrix. Structured params are `{"table_id":"existing-table-id"}`;
the id is required when more than one table exists on the page."""
_PROMPT = """Return JSON only: {\"rows\":1,\"cols\":1,\"cells\":[[{\"text\":\"...\",\"rowspan\":1,\"colspan\":1,\"style_source_cell_id\":\"optional-existing-cell-id\"}]]}.
Produce the complete replacement matrix for the requested table. Keep facts and
user-provided wording. Use merges only when clearly required. You may use
page_items as factual source material. Set style_source_cell_id only when a new
cell should inherit one supplied existing cell's style."""

def _repair(params: dict[str, Any]) -> list[str]: return []

def _run(*, intent, state, api_key, model, dry_run, user_request="") -> SkillResult:
    objective = str(intent.get("objective") or "").strip()
    tables = [t for t in state.get("tables", []) if isinstance(t, dict)]
    tid = str((intent.get("params") or {}).get("table_id") or "") if isinstance(intent.get("params"), dict) else ""
    target = next((t for t in tables if tid and str(t.get("id")) == tid), None)
    if target is None and len(tables) == 1: target = tables[0]
    if not objective or target is None: return SkillResult(warnings=["table_rebuild_target_missing"], status="failed")
    original = validate_table_spec(target)
    if dry_run: return SkillResult(warnings=["dry_run_table_rebuild"], status="already_satisfied")
    page_items = [
        {"id": str(item.get("id") or ""), "kind": str(item.get("kind") or ""), "text": str(item.get("text") or "")}
        for item in state.get("texts") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    for other_table in tables:
        if other_table is target:
            continue
        for row in other_table.get("data") or []:
            for cell in row if isinstance(row, list) else []:
                if isinstance(cell, dict) and str(cell.get("text") or "").strip():
                    page_items.append({"id": str(cell.get("id") or ""), "kind": "table_cell", "text": str(cell.get("text") or "")})
    for image in state.get("images") or []:
        if isinstance(image, dict) and str(image.get("description_en") or "").strip():
            page_items.append({"id": str(image.get("id") or ""), "kind": "image_description", "text": str(image.get("description_en") or "")})
    payload = {
        "objective": objective,
        "table": {"id": original["id"], "data": [[{"id": c["id"], "text": c["text"], "rowspan": c["rowspan"], "colspan": c["colspan"]} for c in row] for row in original["data"]]},
        "page_items": page_items,
    }
    _raw, obj, err = _call_claude_json(model=model, system_prompt=_PROMPT, user_text=json.dumps(payload, ensure_ascii=False), max_output_tokens=8192, tag="step2.table_rebuild", reasoning_effort="minimal")
    if err or not isinstance(obj, dict) or not isinstance(obj.get("cells"), list): return SkillResult(warnings=[f"table_rebuild_call_error: {err or 'invalid'}"], status="failed")
    raw_rows = obj["cells"]
    rows, cols = len(raw_rows), len(raw_rows[0]) if raw_rows and isinstance(raw_rows[0], list) else 0
    if not rows or not cols or any(not isinstance(row, list) or len(row) != cols for row in raw_rows): return SkillResult(warnings=["table_rebuild_bad_matrix"], status="failed")
    semantic_original = [[
        {"text": cell["text"], "rowspan": cell["rowspan"], "colspan": cell["colspan"]}
        for cell in row
    ] for row in original["data"]]
    semantic_candidate = [[
        {
            "text": str((raw if isinstance(raw, dict) else {}).get("text") or ""),
            "rowspan": (raw if isinstance(raw, dict) else {}).get("rowspan", 1),
            "colspan": (raw if isinstance(raw, dict) else {}).get("colspan", 1),
        }
        for raw in row
    ] for row in raw_rows]
    if semantic_candidate == semantic_original:
        return SkillResult(warnings=["table_rebuild_no_change"], status="already_satisfied")

    source_styles = {
        str(cell.get("id") or ""): deepcopy(cell.get("style") or {})
        for row in original["data"]
        for cell in row
        if isinstance(cell, dict) and str(cell.get("id") or "")
    }
    same_shape = rows == len(original["data"]) and cols == len(original["data"][0])
    data = []
    for r, row in enumerate(raw_rows):
        out = []
        for c, raw in enumerate(row):
            raw = raw if isinstance(raw, dict) else {}
            source_id = str(raw.get("style_source_cell_id") or "")
            fallback_style = original["data"][r][c].get("style") if same_shape else {}
            style = deepcopy(source_styles.get(source_id, fallback_style or {}))
            cell_id = str(original["data"][r][c].get("id") or "") if same_shape else f"{original['id']}_r{r}_c{c}"
            out.append({"id": cell_id, "text": str(raw.get("text") or ""), "rowspan": raw.get("rowspan", 1), "colspan": raw.get("colspan", 1), "style": style})
        data.append(out)
    candidate = dict(original)
    candidate["data"] = data
    candidate["colWidths"] = (
        deepcopy(original["colWidths"])
        if cols == len(original["colWidths"])
        else [1.0 / cols] * cols
    )
    try: candidate = validate_table_spec(candidate)
    except ValueError as exc: return SkillResult(warnings=[f"table_rebuild_invalid_matrix: {exc}"], status="failed")
    target.clear(); target.update(candidate)
    return SkillResult(warnings=[f"table_rebuilt: {rows}x{cols}"], triggered_visual=True, status="applied")

SKILL = Skill(id="table.rebuild", canonical_rank=4, summary="replace one existing native table matrix", plan_doc=_DOC, repair=_repair, execute=_run, ordering_note="Usually follows content transformations whose results must be represented in the replacement matrix.", phase="table")
