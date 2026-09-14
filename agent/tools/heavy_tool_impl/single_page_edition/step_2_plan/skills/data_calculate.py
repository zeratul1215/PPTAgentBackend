"""Generic deterministic calculation skill writing to an existing target."""
from __future__ import annotations
from typing import Any
from .base import Skill, SkillResult
from . import table_calc as calc
from .....table_spec import iter_cell_refs

_DOC = """Capability: calculate sum, mean, minimum, maximum, or count from
numeric page content and write the result into an explicitly identified existing
text node or visible table cell. Arithmetic is deterministic and table geometry
is unchanged. Structured params are
`{"op":"sum|mean|min|max|count","source_ids":["..."],"target_id":"..."}`.
The planner chooses existing source and target ids from the current PageSpec;
the executor performs the calculation and direct writeback."""

def _repair(params: dict[str, Any]) -> list[str]:
    if params.get("op") not in set(calc.operator_names()): params["op"] = "sum"
    return []

def _run(*, intent, state, api_key, model, dry_run, user_request="") -> SkillResult:
    p = intent.get("params") if isinstance(intent.get("params"), dict) else {}
    op = str(p.get("op") or "sum")
    if op not in set(calc.operator_names()): return SkillResult(warnings=["calculate_invalid_operator"], status="failed")
    target_id = str(p.get("target_id") or "")
    source_ids = [str(x) for x in (p.get("source_ids") or []) if str(x)]
    values: list[float] = []
    for table in state.get("tables") or []:
        if not isinstance(table, dict): continue
        for _r, _c, cell in iter_cell_refs(table, include_placeholders=False):
            if str(cell.get("id")) in source_ids:
                parsed = calc.parse_number(str(cell.get("text") or ""))
                if parsed: values.append(parsed[0])
    for text in state.get("texts") or []:
        if isinstance(text, dict) and str(text.get("id")) in source_ids:
            parsed = calc.parse_number(str(text.get("text") or ""))
            if parsed: values.append(parsed[0])
    if not values: return SkillResult(warnings=["calculate_no_numeric_sources"], status="failed")
    result = calc.format_number(calc.reduce_values(op, values) or 0)
    if dry_run: return SkillResult(warnings=["dry_run_calculate"], status="already_satisfied")
    for text in state.get("texts") or []:
        if isinstance(text, dict) and str(text.get("id")) == target_id:
            if str(text.get("text") or "") == result: return SkillResult(status="already_satisfied")
            text["text"] = result; text["segments"] = [result]
            return SkillResult(warnings=["calculate_applied"], status="applied")
    for table in state.get("tables") or []:
        if not isinstance(table, dict): continue
        for _r, _c, cell in iter_cell_refs(table, include_placeholders=False):
            if str(cell.get("id")) == target_id:
                if str(cell.get("text") or "") == result: return SkillResult(status="already_satisfied")
                cell["text"] = result
                return SkillResult(warnings=["calculate_applied"], status="applied")
    return SkillResult(warnings=["calculate_target_missing"], status="failed")

SKILL = Skill(id="data.calculate", canonical_rank=5, summary="calculate numeric values and write a result", plan_doc=_DOC, repair=_repair, execute=_run, ordering_note="Must follow any transformation that changes its numeric sources and precede any transformation that consumes its result.", phase="table")
