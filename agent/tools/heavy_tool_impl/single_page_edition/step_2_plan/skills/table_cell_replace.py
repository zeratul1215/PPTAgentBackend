"""Replace the complete text of one or more existing table cells."""

from __future__ import annotations

import json
from typing import Any

from .base import Skill, SkillResult, _call_claude_json
from .....table_spec import iter_cell_refs, validate_table_spec


_DOC = """Capability: replace the complete text of one or more existing visible
table cells while preserving the table grid and all geometry and styles. Use it
when the final cell text is already specified or can be copied exactly from the
current page content. It may update several cells in one execution. It does not
translate, rewrite, redact, calculate, add/delete rows or columns, change spans,
or change style; use the corresponding capability for those operations."""

_PROMPT = """You are the TABLE CELL REPLACE executor for one PPT page.

You receive a natural-language objective, the user's original request, all
visible table cells, and the page's current standalone text items. Select only
the existing table cells named by the objective and return their COMPLETE final
text. A single response may replace multiple cells.

Rules:
- Use only supplied cell ids. Never invent an id and never target a merged-cell
  placeholder.
- Preserve user-specified final wording verbatim.
- Use current page items only when the objective asks to copy already available
  content. Do not invent facts or perform translation, rewriting, redaction, or
  calculation here.
- Empty text is allowed only when the objective explicitly asks to clear the
  cell while retaining the table structure.
- Do not change rows, columns, spans, style, ids, or table geometry.
- If no cell can be identified unambiguously, return an empty replacements list.

Output STRICT JSON only:
{"replacements":[{"cell_id":"existing-cell-id","text":"complete final text"}]}
"""


def _repair(params: dict[str, Any]) -> list[str]:
    return []


def _run(
    *,
    intent: dict[str, Any],
    state: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
    user_request: str = "",
) -> SkillResult:
    objective = str(intent.get("objective") or "").strip()
    if not objective:
        return SkillResult(warnings=["table_cell_replace_empty_objective"], status="failed")

    by_id: dict[str, dict[str, Any]] = {}
    table_by_cell_id: dict[str, dict[str, Any]] = {}
    table_payload: list[dict[str, Any]] = []
    for table in state.get("tables") or []:
        if not isinstance(table, dict):
            continue
        validate_table_spec(table)
        cells: list[dict[str, Any]] = []
        for row, col, cell in iter_cell_refs(table, include_placeholders=False):
            cell_id = str(cell.get("id") or "")
            if not cell_id:
                continue
            if cell_id in by_id:
                return SkillResult(
                    warnings=[f"table_cell_replace_duplicate_cell_id:{cell_id}"],
                    status="failed",
                )
            by_id[cell_id] = cell
            table_by_cell_id[cell_id] = table
            cells.append(
                {
                    "cell_id": cell_id,
                    "row": row + 1,
                    "col": col + 1,
                    "text": str(cell.get("text") or ""),
                    "rowspan": max(1, int(cell.get("rowspan", 1) or 1)),
                    "colspan": max(1, int(cell.get("colspan", 1) or 1)),
                }
            )
        table_payload.append({"table_id": str(table.get("id") or ""), "cells": cells})

    if not by_id:
        return SkillResult(warnings=["table_cell_replace_no_visible_cells"], status="failed")
    if dry_run:
        return SkillResult(warnings=["dry_run_table_cell_replace"], status="already_satisfied")

    page_items = [
        {
            "id": str(item.get("id") or ""),
            "kind": str(item.get("kind") or ""),
            "text": str(item.get("text") or ""),
        }
        for item in state.get("texts") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    payload = {
        "user_request": user_request,
        "objective": objective,
        "tables": table_payload,
        "page_items": page_items,
    }
    _raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=4096,
        tag="step2.table_cell_replace",
        reasoning_effort="minimal",
    )
    if err or not isinstance(obj, dict) or not isinstance(obj.get("replacements"), list):
        return SkillResult(
            warnings=[f"table_cell_replace_call_error:{err or 'invalid_response'}"],
            status="failed",
        )

    replacements: dict[str, str] = {}
    for index, raw in enumerate(obj["replacements"]):
        if not isinstance(raw, dict):
            return SkillResult(
                warnings=[f"table_cell_replace_item_not_object:{index}"], status="failed"
            )
        cell_id = str(raw.get("cell_id") or "")
        if cell_id not in by_id:
            return SkillResult(
                warnings=[f"table_cell_replace_unknown_cell:{cell_id or index}"], status="failed"
            )
        if "text" not in raw or not isinstance(raw.get("text"), str):
            return SkillResult(
                warnings=[f"table_cell_replace_text_not_string:{cell_id}"], status="failed"
            )
        text = str(raw["text"])
        if cell_id in replacements and replacements[cell_id] != text:
            return SkillResult(
                warnings=[f"table_cell_replace_conflicting_target:{cell_id}"], status="failed"
            )
        replacements[cell_id] = text

    if not replacements:
        return SkillResult(warnings=["table_cell_replace_selected_nothing"], status="failed")

    changed = [cell_id for cell_id, text in replacements.items() if str(by_id[cell_id].get("text") or "") != text]
    if not changed:
        return SkillResult(warnings=["table_cell_replace_no_change"], status="already_satisfied")

    for cell_id in changed:
        by_id[cell_id]["text"] = replacements[cell_id]
    for table in {id(table): table for table in table_by_cell_id.values()}.values():
        validate_table_spec(table)
    return SkillResult(
        warnings=[f"table_cell_replace_applied:{len(changed)}"],
        status="applied",
        triggered_visual=False,
    )


SKILL = Skill(
    id="table.cell.replace",
    canonical_rank=3,
    summary="replace complete text in one or more existing table cells",
    plan_doc=_DOC,
    repair=_repair,
    execute=_run,
    ordering_note=(
        "Usually follows any capability whose finalized page content must be copied "
        "into existing cells; it is independent when the replacement text is already final."
    ),
    phase="table",
)
