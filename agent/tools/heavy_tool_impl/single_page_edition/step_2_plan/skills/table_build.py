"""Build one native PPTist table from semantic page content."""
from __future__ import annotations
import json
from typing import Any
from .base import Skill, SkillResult, _call_claude_json
from .....table_spec import validate_table_spec

_DOC = """Capability: create exactly one new native table from the objective and
available semantic page content. Sources may include text, shape text, existing
tables, and image descriptions. It outputs a rectangular matrix; fixed code owns
IDs, widths, and validation. It can remove source text only through explicit
consume_source_ids in its result."""
_PROMPT = """Return JSON only:
{\"rows\":1,\"cols\":1,\"cells\":[[{\"text\":\"...\",\"source_ids\":[]}]],\"consume_source_ids\":[]}
Create one rectangular table for objective. Keep user-specified text exact.
Use source_ids when a cell is derived from supplied items; otherwise write the
final cell text. Do not create merges."""

def _repair(params): return []

def _run(*, intent, state, api_key, model, dry_run, user_request="") -> SkillResult:
    objective = str(intent.get("objective") or "").strip()
    texts = [t for t in state.get("texts", []) if isinstance(t, dict) and str(t.get("text") or "").strip() and t.get("kind") != "table_cell"]
    for table in state.get("tables") or []:
        if isinstance(table, dict):
            for row in table.get("data") or []:
                for cell in row if isinstance(row, list) else []:
                    if isinstance(cell, dict) and str(cell.get("text") or "").strip():
                        texts.append({"id": str(cell.get("id") or ""), "kind": "table_cell", "text": str(cell.get("text") or "")})
    for image in state.get("images") or []:
        if not isinstance(image, dict):
            continue
        description = str(image.get("description_en") or "").strip()
        if description:
            texts.append({
                "id": str(image.get("id") or ""),
                "kind": "image_description",
                "text": description,
            })
    if not objective: return SkillResult(warnings=["table_build_empty_objective"], status="failed")
    if dry_run: return SkillResult(warnings=["dry_run_table_build"], status="already_satisfied")
    payload = {"objective": objective, "items":[{"id":t.get("id"),"kind":t.get("kind"),"text":t.get("text")} for t in texts]}
    raw, obj, err = _call_claude_json(model=model, system_prompt=_PROMPT, user_text=json.dumps(payload, ensure_ascii=False), max_output_tokens=4096, tag="step2.table_build", reasoning_effort="minimal")
    if err or not isinstance(obj, dict): return SkillResult(warnings=[f"table_build_call_error: {err or 'invalid'}"], status="failed")
    try: rows, cols = int(obj["rows"]), int(obj["cols"])
    except (KeyError, TypeError, ValueError): return SkillResult(warnings=["table_build_bad_dimensions"], status="failed")
    if rows < 1 or cols < 1 or rows > 100 or cols > 100: return SkillResult(warnings=["table_build_bad_dimensions"], status="failed")
    by_id = {str(t.get("id")): t for t in texts}; raw_rows = obj.get("cells")
    if not isinstance(raw_rows, list) or len(raw_rows) != rows: return SkillResult(warnings=["table_build_bad_cells"], status="failed")
    data = []
    used = set()
    for r in range(rows):
        out = []
        source_row = raw_rows[r] if r < len(raw_rows) and isinstance(raw_rows[r], list) else []
        for c in range(cols):
            spec = source_row[c] if c < len(source_row) and isinstance(source_row[c], dict) else {}
            ids = spec.get("source_ids") if isinstance(spec.get("source_ids"), list) else []
            vals = [str(by_id[i].get("text") or "") for i in ids if str(i) in by_id]
            used.update(str(i) for i in ids if str(i) in by_id)
            text = str(spec.get("text") or "") if "text" in spec else " ".join(vals)
            out.append({"id":f"tbl0_r{r}_c{c}","text":text,"rowspan":1,"colspan":1,"style":{}})
        data.append(out)
    existing_ids = {str(t.get("id") or "") for t in state.get("tables") or [] if isinstance(t, dict)}
    table_id = next((f"tbl{i}" for i in range(1000) if f"tbl{i}" not in existing_ids), "tbl_new")
    for row_index, row in enumerate(data):
        for col_index, cell in enumerate(row):
            cell["id"] = f"{table_id}_r{row_index}_c{col_index}"
    table = {"id":table_id,"colWidths":[1/cols]*cols,"cellMinHeight":24,"outline":{"width":1,"style":"solid","color":"#eeece1"},"theme":{"color":"#67508F","rowHeader":False,"rowFooter":False,"colHeader":False,"colFooter":False},"data":data}
    try: table = validate_table_spec(table)
    except ValueError as exc: return SkillResult(warnings=[f"table_build_invalid_output: {exc}"], status="failed")
    state.setdefault("tables", []).append(table)
    consume = obj.get("consume_source_ids") if isinstance(obj.get("consume_source_ids"), list) else []
    consume_ids = {str(x) for x in consume if str(x) in used}
    state["texts"] = [t for t in state.get("texts", []) if str(t.get("id")) not in consume_ids]
    return SkillResult(warnings=[f"table_built: {rows}x{cols}"], triggered_visual=True, status="applied")

SKILL = Skill(id="table.build", canonical_rank=3, summary="create one native table from semantic page content", plan_doc=_DOC, repair=_repair, execute=_run, ordering_note="Usually follows content transformations whose final results are sources for the new table.", phase="table")
