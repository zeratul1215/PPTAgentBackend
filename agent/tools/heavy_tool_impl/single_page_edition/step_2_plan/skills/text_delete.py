"""Skill: text.delete — remove existing text block(s)/line(s) the user asked to drop.

Objective-driven: the planner passes a natural-language `objective` naming WHICH
text to remove (by content / role / position). The executor sees the WHOLE page,
picks the in-scope items (shared `_LOCATE_GUIDE`), and deletes exactly those.

Deletion granularity is the SEGMENT (one visible line/item): removing some lines
of a multi-line block keeps the block with the rest; removing every line of a
block removes the block entirely from `state["texts"]`. Dangling references left
by removed nodes (another node's `translation_of`/`derived_from`, a table cell's
`ref`) are cleaned up centrally in the compile layer (`_prune_dangling_text_refs`)
— the referring text is NOT cascade-deleted, its dangling link is just dropped.

Removing elements changes the element set → `triggered_visual=True` so the page
re-flows to close the gap left behind. No `default_visual_detail` is contributed.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    _LOCATE_GUIDE,
    Skill,
    SkillResult,
    _all_text_ids,
    _call_claude_json,
    _flatten_to_segment_items,
    _get_segments,
    _set_segments,
    _text_targets,
)


_DELETE_SYSTEM_PROMPT = """You are the "DELETE TEXT" subagent for a single-page PPT-editing pipeline.

You decide WHICH existing text to REMOVE from the page.

You are given:
- `user_request`: the user's full original instruction (context only).
- `objective`: what THIS skill must delete, in natural language — it names WHICH
  text to remove (e.g. "delete the footer disclaimer", "remove the last bullet",
  "drop the paragraph about pricing", "delete the whole subtitle").
- `items`: EVERY text item on the page (each is one visible line/segment, with
  `id`, `kind`, `text`).

""" + _LOCATE_GUIDE + """

Report the ids of the items to DELETE.

Output STRICT JSON only, no markdown:
{ "delete_ids": ["t0::seg0", "t2::seg1", ...] }

Rules:
- Include an id ONLY if the objective clearly asks to remove that exact item.
- Do NOT rewrite, translate, or edit anything — you only mark items for deletion.
- Deleting some lines of a multi-line block is fine (list just those line ids);
  to delete a whole block, list ALL of its line ids.
- If NOTHING clearly matches, return an empty `delete_ids` array (better to delete
  nothing than to delete the wrong text).
- Never invent ids. Output STRICT JSON only. No commentary.
"""


def _repair_delete_params(params: dict[str, Any]) -> list[str]:
    # Fully objective-driven; no structured params.
    return []


def _run_delete(
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
        warnings.append(f"delete_empty_objective[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    texts: list[dict[str, Any]] = _text_targets(state)
    by_id: dict[str, dict[str, Any]] = {str(t.get("id") or ""): t for t in texts if isinstance(t, dict)}
    all_ids = _all_text_ids(by_id)
    if not all_ids:
        warnings.append(f"delete_no_text_on_page[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    seg_items, seg_map, _seg_counts = _flatten_to_segment_items(
        ids=all_ids, by_id=by_id, selected=None
    )
    if not seg_items:
        warnings.append(f"delete_empty_segment_items[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    if dry_run:
        warnings.append(f"dry_run_stub_delete: {iid}")
        return SkillResult(warnings=warnings, status="already_satisfied", triggered_visual=False)

    payload = {
        "user_request": user_request,
        "objective": objective,
        "items": seg_items,
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_DELETE_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=2048,
        tag="step2.delete",
        reasoning_effort="minimal",
    )
    if err:
        warnings.append(f"delete_call_error[{iid}]: {err}")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("delete_ids"), list):
        warnings.append(f"delete_invalid_response[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    known_seg_ids = set(seg_map.keys())
    # Group selected segment indices per text node.
    seg_idxs_by_tid: dict[str, set[int]] = {}
    for x in obj.get("delete_ids") or []:
        sid = str(x or "").strip()
        if sid not in known_seg_ids:
            if sid:
                warnings.append(f"delete_unknown_id[{iid}]: {sid}")
            continue
        tid, idx = seg_map.get(sid, ("", -1))
        if not tid or idx < 0:
            continue
        seg_idxs_by_tid.setdefault(tid, set()).add(idx)

    if not seg_idxs_by_tid:
        warnings.append(f"delete_selected_nothing[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    removed_nodes = 0
    removed_segments = 0
    ids_to_drop: set[str] = set()
    for tid, drop_idxs in seg_idxs_by_tid.items():
        node = by_id.get(tid)
        if not isinstance(node, dict):
            continue
        segs = _get_segments(node) or [str(node.get("text") or "")]
        kept = [s for i, s in enumerate(segs) if i not in drop_idxs]
        removed_segments += len(segs) - len(kept)
        if kept:
            _set_segments(node, kept)
        else:
            ids_to_drop.add(tid)

    table_ids_to_clear: set[str] = set()
    if ids_to_drop:
        for tid in ids_to_drop:
            node = by_id.get(tid)
            if isinstance(node, dict) and isinstance(node.get("_table_cell_ref"), dict):
                node["_table_cell_ref"]["text"] = ""
                table_ids_to_clear.add(tid)
        original_texts = state.get("texts") if isinstance(state.get("texts"), list) else []
        state["texts"] = [
            t for t in original_texts
            if not (isinstance(t, dict) and str(t.get("id") or "") in ids_to_drop)
        ]
        removed_nodes = len(ids_to_drop - table_ids_to_clear)

    if removed_nodes == 0 and removed_segments == 0:
        warnings.append(f"delete_selected_nothing[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    warnings.append(
        f"delete_applied[{iid}]: {removed_nodes} block(s), {removed_segments} line(s)"
    )
    # Removing elements changes the element set → re-flow to close the gap. The
    # compile layer prunes any dangling references left by dropped nodes.
    return SkillResult(warnings=warnings, status="applied", triggered_visual=True)


_DELETE_PLAN_DOC = """  Capability: remove existing text blocks or selected
  lines/items. The natural-language objective must identify the target by visible
  content, semantic role, or position. It removes text rather than masking or
  rewriting it. Ordinary and Shape-contained text blocks may be removed; deleting
  all text from a table cell clears that cell but preserves the table and grid.
  Removing content may require visual re-layout."""


SKILL = Skill(
    id="text.delete",
    canonical_rank=4,
    summary="delete existing text block(s)/line(s) (triggers re-layout)",
    plan_doc=_DELETE_PLAN_DOC,
    repair=_repair_delete_params,
    execute=_run_delete,
    ordering_note="Usually precedes transformations of the same scope so content scheduled for removal is not processed unnecessarily.",
)
