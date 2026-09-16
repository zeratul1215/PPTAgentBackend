"""Delete one complete page element from the native understanding state."""
from __future__ import annotations
from typing import Any
from .base import Skill, SkillResult

_DOC = """Capability: remove one complete existing PageSpec element. Supported
targets are standalone text, image, and table elements. It removes the container
itself and cannot clear nested content while retaining that container.
Structured params are `{"element_id":"existing-element-id"}`."""

def _repair(params: dict[str, Any]) -> list[str]: return []
def _run(*, intent, state, api_key, model, dry_run, user_request="") -> SkillResult:
    p = intent.get("params") if isinstance(intent.get("params"), dict) else {}
    target_id = str(p.get("element_id") or p.get("target_id") or "")
    if not target_id: return SkillResult(warnings=["element_delete_target_missing"], status="failed")
    for key in ("texts", "images", "tables"):
        values = state.get(key)
        if not isinstance(values, list): continue
        kept = [v for v in values if not (isinstance(v, dict) and str(v.get("id") or "") == target_id)]
        if len(kept) != len(values):
            if dry_run: return SkillResult(status="already_satisfied")
            state[key] = kept
            return SkillResult(warnings=[f"element_deleted:{target_id}"], triggered_visual=True, status="applied")
    return SkillResult(warnings=[f"element_delete_not_found:{target_id}"], status="failed")

SKILL = Skill(id="element.delete", canonical_rank=6, summary="delete a complete page element", plan_doc=_DOC, repair=_repair, execute=_run, ordering_note="Usually precedes work on the same element; it is independent of transformations targeting other elements.", phase="table")
