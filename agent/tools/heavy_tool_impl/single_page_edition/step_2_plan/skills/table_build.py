"""Skill: table.build — organize final text nodes into a table grid.

Objective-driven: the planner passes a natural-language `objective` (e.g. "put
the year / metric / value content into a 3-column table, with the English
translation as a 4th column"); NO rows/cols/cells params. This executor runs in
the "table" phase — AFTER every text edit — so it sees the FINAL text nodes
(rewrites, translations) with stable ids. It makes ONE model call to decide the
row/col → text-id mapping over those final nodes, then records the grid using the
schema step1/step3 share: { id, rows, cols, cells:[{row,col,ref}] }.

It does NOT edit any text; it only records structure.
"""

from __future__ import annotations

import json
from typing import Any

from .base import _LOCATE_GUIDE, Skill, SkillResult, _call_claude_json


_BUILD_SYSTEM_PROMPT = """You are the "TABLE-BUILD" subagent for a single-page PPT-editing pipeline.

You are given:
- `user_request`: the user's full original instruction (context only).
- `objective`: what table to build, in natural language (which content goes into
  the table and roughly how it should be organized into columns/rows).
- `items`: EVERY text item on the page. Each has `id`, `kind`, `text`, and (if
  multi-line) a `segments` array. Translations added earlier carry
  `translation_of` = the source id they translate.

""" + _LOCATE_GUIDE + """

Then design a table grid over the IN-SCOPE content and output the row/col → id
mapping.

Grid rules:
- Decide `rows` and `cols` (both >= 1) from the content and the objective.
- Emit one `cells` entry per NON-EMPTY cell: {row, col, ref}. `row`/`col` are
  0-based. Omit empty cells.
- `ref` MUST be an `id` from `items`. Never invent an id.
- A column that is ONE multi-segment node (one segment per row): put that SAME
  `ref` on that column's cell in EVERY row (the `row` index selects the segment).
  Repeating a ref within a column is expected and correct.
- Combine parallel content into ONE table: e.g. a year column + a Chinese column
  + its English-translation column are three `col` values of the SAME table.
- For a translation column, reference the TRANSLATION node directly by its `id`
  (the item whose `translation_of` points at the source) — translations already
  exist at this stage, so use their real ids.

Output STRICT JSON only. No markdown. No commentary:
{
  "rows": <int>,
  "cols": <int>,
  "cells": [ { "row": <int>, "col": <int>, "ref": "<item id>" } ]
}
If the objective cannot be satisfied (e.g. no matching content), return
{"rows": 0, "cols": 0, "cells": []}.
"""


def _repair_build_params(params: dict[str, Any]) -> list[str]:
    # table.build is objective-driven now; it carries no structured params.
    return []


def _slim_items_for_build(texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for t in texts:
        if not isinstance(t, dict):
            continue
        if not str(t.get("text") or "").strip():
            continue
        item: dict[str, Any] = {
            "id": str(t.get("id") or ""),
            "kind": str(t.get("kind") or ""),
            "text": str(t.get("text") or ""),
        }
        segs = t.get("segments")
        if isinstance(segs, list) and len(segs) > 1 and all(isinstance(x, str) for x in segs):
            item["segments"] = [str(x) for x in segs]
        tof = str(t.get("translation_of") or "")
        if tof:
            item["translation_of"] = tof
        items.append(item)
    return items


def _run_table(
    *,
    intent: dict[str, Any],
    state: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
    user_request: str = "",
) -> SkillResult:
    warnings: list[str] = []
    iid = str(intent.get("id") or "")
    objective = str(intent.get("objective") or "").strip()
    if not objective:
        warnings.append(f"table_empty_objective[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    texts = state.get("texts") if isinstance(state.get("texts"), list) else []
    valid_ids = {str(t.get("id") or "") for t in texts if isinstance(t, dict)}
    items = _slim_items_for_build(texts)
    if not items:
        warnings.append(f"table_no_text_on_page[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    if dry_run:
        warnings.append(f"dry_run_stub_table_build: {iid}")
        return SkillResult(warnings=warnings, triggered_visual=False)

    payload = {
        "user_request": user_request,
        "objective": objective,
        "items": items,
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_BUILD_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=4096,
        tag="step2.table_build",
        reasoning_effort="minimal",
    )
    if err:
        warnings.append(f"table_build_call_error[{iid}]: {err}")
        return SkillResult(warnings=warnings, triggered_visual=False)
    if not isinstance(obj, dict):
        warnings.append(f"table_build_invalid_response[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    try:
        rows = int(obj.get("rows"))
        cols = int(obj.get("cols"))
    except (TypeError, ValueError):
        warnings.append(f"table_bad_dims[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)
    if rows <= 0 or cols <= 0:
        warnings.append(f"table_nonpositive_dims[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    raw_cells = obj.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        warnings.append(f"table_no_cells[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    seen_rc: set[tuple[int, int]] = set()
    cells_out: list[dict[str, Any]] = []
    used_refs: set[str] = set()
    for c_i, cell in enumerate(raw_cells):
        if not isinstance(cell, dict):
            warnings.append(f"table_cell[{iid}][{c_i}]_not_object")
            continue
        try:
            r = int(cell.get("row"))
            c = int(cell.get("col"))
        except (TypeError, ValueError):
            warnings.append(f"table_cell[{iid}][{c_i}]_bad_index")
            continue
        ref = str(cell.get("ref") or "")
        if not (0 <= r < rows and 0 <= c < cols):
            warnings.append(f"table_cell[{iid}][{c_i}]_out_of_bounds: {(r, c)}")
            continue
        if ref not in valid_ids:
            warnings.append(f"table_cell[{iid}][{c_i}]_unknown_ref: {ref}")
            continue
        if (r, c) in seen_rc:
            warnings.append(f"table_cell[{iid}][{c_i}]_duplicate_rc: {(r, c)}")
            continue
        seen_rc.add((r, c))
        used_refs.add(ref)
        cells_out.append({"row": r, "col": c, "ref": ref})

    if not cells_out:
        warnings.append(f"table_no_valid_cells[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    tables = state.get("tables")
    if not isinstance(tables, list):
        tables = []
        state["tables"] = tables
    table_id = f"tbl{len(tables)}"
    tables.append({"id": table_id, "rows": rows, "cols": cols, "cells": cells_out})
    warnings.append(f"table_built[{iid}]: {rows}x{cols}, {len(used_refs)} refs")
    return SkillResult(warnings=warnings, triggered_visual=True)


_TABLE_PLAN_DOC = """  Organize content into a TABLE grid (rows x columns). Use ONLY when the user
  EXPLICITLY asks for a table / columns / grid, e.g. "把这些做成表格", "make this
  a two-column table", "add an English column next to the Chinese". Do NOT emit
  this for ordinary lists/bullets or just because content looks aligned — step1
  already detects visual tables on its own every round.
  `objective` (natural language) MUST describe WHICH content goes into the table
  and how it should be organized into columns/rows — e.g. "Put the year, metric,
  and value content into a 3-column table, one row per year, and add the English
  translation of the metric names as a 4th column." No `params` — the executor
  runs after all text edits, sees the final text (including translations), and
  works out the row/col → id mapping itself. Creating a table changes the layout,
  so this skill REQUESTS a visual re-layout at runtime."""


_TABLE_ORDERING_NOTE = (
    "Runs in the TABLE phase, which the compiler forces to run AFTER all text "
    "edits (redact/rewrite/translate). You do not need `after` edges to text "
    "intents — just describe the table you want; the executor sees the final "
    "text (including any translations) when it builds the grid."
)


SKILL = Skill(
    id="table.build",
    canonical_rank=3,
    summary="organize content into a table grid (rows x columns)",
    plan_doc=_TABLE_PLAN_DOC,
    repair=_repair_build_params,
    execute=_run_table,
    ordering_note=_TABLE_ORDERING_NOTE,
    phase="table",
)
