"""Skill: text.add — add NEW text block(s) the user asked for.

Objective-driven: the planner passes a natural-language `objective` describing
WHAT text to add (and, if the user said so, roughly where / what role). The
executor sees the WHOLE page for context (tone, language, existing structure),
authors the new block(s), and appends each as a fresh text node via
`_append_added_text`.

The new block carries ONLY `kind` + content — NO position. Placement is left to
the reference-image model / step3 (page semantics decide where "a paragraph the
user added" goes; there is no reusable default location, unlike bilingual
translation). Adding an element changes the element set → `triggered_visual=True`
so the layout is re-flowed and the new block lands somewhere sensible. No
`default_visual_detail` is contributed (see changelog 2026-08-04_01, correction 2).
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    Skill,
    SkillResult,
    _append_added_text,
    _call_claude_json,
    _text_targets,
)


# Kinds the new block MAY use — the SAME open set already present on pages; no
# whitelist (there is no "kind you can't add"). The model self-selects from the
# objective's semantics and falls back to "body".
_ADD_SYSTEM_PROMPT = """You are the "ADD TEXT" subagent for a single-page PPT-editing pipeline.

You AUTHOR one or more NEW text blocks that the user asked to add to the page.

You are given:
- `user_request`: the user's full original instruction (context only).
- `objective`: what THIS skill must add, in natural language — the content to
  create (and possibly its role, e.g. "add a closing takeaway line", "add three
  bullet points summarizing the benefits", "add a subtitle under the title").
- `items`: semantic context already on the page, including ordinary text,
  shape text, table-cell text, and factual image descriptions. These are
  CONTEXT ONLY, so the new text matches the page's language, tone, terminology,
  and level of detail. Do NOT edit or repeat an existing item; only CREATE new
  text blocks.

Author the new block(s):
1) Write each block's `text` to satisfy `objective`, in the SAME language as the
   page's existing text unless the objective explicitly says otherwise. Match the
   page's tone and be concise / PPT-ready (信达雅).
2) For each block choose a `kind` describing its role. Use the SAME vocabulary the
   page already uses (e.g. title, heading, subheading, body, paragraph,
   bullet_item, caption, footer, label). There is NO whitelist — pick whatever
   fits. If you truly cannot tell, use "body".
3) If a block is a LIST (several bullets/lines), return its lines in `segments`
   (one string per line); otherwise omit `segments` and just give `text`. For a
   multi-item bullet list the user asked for, prefer ONE block whose `segments`
   are the individual bullets (kind "bullet_item").

Do NOT decide WHERE the block goes on the page — position is chosen later. Do not
mention coordinates, columns, or sides in the text itself.

Output STRICT JSON only, no markdown:
{
  "blocks": [
    { "kind": "body|title|bullet_item|caption|...", "text": "<the new text>",
      "segments": ["line 1", "line 2"] }
  ]
}

Rules:
- Return ONLY brand-new blocks. If the objective does not actually call for any
  new text (e.g. it's really a rewrite of existing text), return an empty
  `blocks` array.
- `segments` is optional; include it only for genuine multi-line/list blocks.
- Output STRICT JSON only. No commentary.
"""


def _repair_add_params(params: dict[str, Any]) -> list[str]:
    # Fully objective-driven; no structured params.
    return []


def _run_add(
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
        warnings.append(f"add_empty_objective[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    texts: list[dict[str, Any]] = _text_targets(state)
    context_items = [
        {
            "id": str(t.get("id") or ""),
            "kind": str(t.get("kind") or ""),
            "text": str(t.get("text") or ""),
        }
        for t in texts
        if isinstance(t, dict) and str(t.get("text") or "").strip()
    ]
    for image in state.get("images") or []:
        if not isinstance(image, dict):
            continue
        description = str(image.get("description_en") or "").strip()
        if description:
            context_items.append(
                {
                    "id": str(image.get("id") or ""),
                    "kind": "image_description",
                    "text": description,
                }
            )

    if dry_run:
        _append_added_text(state=state, text=f"[added] {objective}"[:120], kind="body")
        warnings.append(f"dry_run_stub_add: {iid}")
        return SkillResult(warnings=warnings, status="applied", triggered_visual=True)

    payload = {
        "user_request": user_request,
        "objective": objective,
        "items": context_items,
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_ADD_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=4096,
        tag="step2.add",
        reasoning_effort="minimal",
    )
    if err:
        warnings.append(f"add_call_error[{iid}]: {err}")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("blocks"), list):
        warnings.append(f"add_invalid_response[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    added = 0
    for b_idx, block in enumerate(obj.get("blocks") or []):
        if not isinstance(block, dict):
            warnings.append(f"add_block_not_object[{iid}][{b_idx}]")
            continue
        kind = str(block.get("kind") or "").strip() or "body"
        segs_raw = block.get("segments")
        segments = (
            [str(x) for x in segs_raw if isinstance(x, str) and str(x).strip()]
            if isinstance(segs_raw, list)
            else None
        )
        text = str(block.get("text") or "").strip()
        if not text and segments:
            text = "".join(segments)
        if not text:
            warnings.append(f"add_block_empty[{iid}][{b_idx}]")
            continue
        _append_added_text(state=state, text=text, kind=kind, segments=segments)
        added += 1

    if added == 0:
        warnings.append(f"add_added_nothing[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    warnings.append(f"add_applied[{iid}]: {added} block(s)")
    # New elements change the element set → re-flow so they land well. No default
    # visual detail: there is no reusable default placement for arbitrary added
    # text (unlike bilingual). If the user gave a placement, it's already in
    # visual_intent.requirements_text (visual_detail_provided=true) and wins.
    return SkillResult(warnings=warnings, status="applied", triggered_visual=True)


_ADD_PLAN_DOC = """  Capability: add brand-new text blocks that do not already
  exist on the page. The natural-language objective must specify the content or
  permitted generation scope and the intended semantic role. The executor uses
  the objective and current page context. Placement belongs in visual_intent.
  Adding text may require visual re-layout."""


SKILL = Skill(
    id="text.add",
    canonical_rank=3,
    summary="add brand-new text block(s) the user asked for (triggers re-layout)",
    plan_doc=_ADD_PLAN_DOC,
    repair=_repair_add_params,
    execute=_run_add,
    ordering_note="Usually follows edits whose results the new text must summarize, match, or otherwise depend on; it is independent when no such dependency exists.",
)
