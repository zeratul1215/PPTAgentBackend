"""Skill: text.rewrite — rewrite in-scope text in the same language.

Objective-driven: the planner passes a natural-language `objective` (no ids, no
params). This executor is handed the WHOLE page, decides for itself which items
the objective refers to (using the shared `_LOCATE_GUIDE`), and rewrites ONLY
those — selection + transform in a single model call.
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
)


_REWRITE_SYSTEM_PROMPT = """You are the "REWRITE" subagent for a single-page PPT-editing pipeline.

You are given:
- `user_request`: the user's full original instruction (context only).
- `objective`: what THIS rewrite must accomplish, in natural language. It also
  tells you WHICH text to act on (e.g. "the main title", "the body paragraphs").
- `items`: EVERY text item on the page. Each item is one visible line/segment
  with its `id`, `kind`, and `text`.

""" + _LOCATE_GUIDE + """

Then rewrite ONLY the in-scope items:

1) **Rewrite (in the SAME language as the source)**: produce `tgt_text` for each
   in-scope item by rewriting its `text` to satisfy `objective`.
   - **Language lock**: `tgt_text` MUST be in the SAME language as the source.
     Chinese stays Chinese; English stays English. Do NOT translate.
   - **Preserve meaning by default**: keep facts, named entities, numbers, dates,
     URLs, abbreviations intact unless the objective explicitly asks otherwise.
   - **Preserve placeholder tokens VERBATIM**: tokens like `<ORG>`, `<REDACTED>`,
     `<EMAIL>` (uppercase inside angle brackets) must be reproduced
     character-for-character. Never translate/split/strip them.
   - **信达雅**: faithful, fluent, elegant — natural PPT-ready phrasing.
   - Interpret any style/length hints from `objective` (e.g. "更简洁/more
     concise", "标题式/headline", "营销化/marketing", "精简到 12 字"). If the
     objective implies a length ceiling, respect it without sacrificing meaning.
   - **Default style when the objective names none**: if `objective` gives no
     style/tone/length direction at all, rewrite to be MORE FORMAL / more
     professional while keeping the original meaning and roughly the original
     length.
   - For a headline style: keep each `tgt_text` a complete, self-contained phrase
     (never a stub that only makes sense joined with another fragment).

2) **Track sources**: emit ONE output paragraph per in-scope item, with `sources`
   containing exactly that item's id.

Hard rules:
- Output paragraphs ONLY for in-scope items. Items you leave out are kept
  unchanged — that is how you express "not in scope". Do NOT invent ids.
- Each in-scope id MUST appear in exactly ONE `paragraphs[i].sources`.
- `paragraphs[i].kind` MUST equal the kind of the source item.
- `paragraphs[i].src_text` MUST equal the source item's text.
- If NOTHING matches the objective, return an empty `paragraphs` array.
- Output STRICT JSON only. No markdown. No commentary.

Output format:
{
  "paragraphs": [
    { "kind": "title|heading|body|bullet_item|...", "sources": ["t0::seg0"],
      "src_text": "...", "tgt_text": "..." }
  ]
}
"""


def _repair_rewrite_params(params: dict[str, Any]) -> list[str]:
    # Rewrite is fully objective-driven now; it carries no structured params.
    return []


def _run_rewrite(
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
        warnings.append(f"rewrite_empty_objective[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    texts: list[dict[str, Any]] = state.get("texts") or []
    by_id: dict[str, dict[str, Any]] = {str(t.get("id") or ""): t for t in texts if isinstance(t, dict)}
    all_ids = _all_text_ids(by_id)
    if not all_ids:
        warnings.append(f"rewrite_no_text_on_page[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    seg_items, seg_map, _seg_counts = _flatten_to_segment_items(
        ids=all_ids, by_id=by_id, selected=None
    )
    if not seg_items:
        warnings.append(f"rewrite_empty_segment_items[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    if dry_run:
        # Dry-run cannot resolve scope semantically; stub-rewrite everything so
        # the pipeline stays exercisable without a model call.
        for it in seg_items:
            sid = str(it.get("id") or "")
            tid, idx = seg_map.get(sid, ("", -1))
            node = by_id.get(tid)
            if not isinstance(node, dict) or idx < 0:
                continue
            segs = _get_segments(node) or [str(node.get("text") or "")]
            while len(segs) <= idx:
                segs.append("")
            segs[idx] = f"[rewrite] {str(it.get('text') or '')}"
            _set_segments(node, segs)
        warnings.append(f"dry_run_stub_rewrite: {iid}")
        return SkillResult(warnings=warnings, triggered_visual=False)

    payload = {
        "user_request": user_request,
        "objective": objective,
        "items": seg_items,
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_REWRITE_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=4096,
        tag="step2.rewrite",
        reasoning_effort="minimal",
    )
    if err:
        warnings.append(f"rewrite_call_error[{iid}]: {err}")
        return SkillResult(warnings=warnings, triggered_visual=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("paragraphs"), list):
        warnings.append(f"rewrite_invalid_response[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    known_seg_ids = set(seg_map.keys())
    seen_ids: set[str] = set()
    touched = 0

    for p_idx, para in enumerate(obj["paragraphs"]):
        if not isinstance(para, dict):
            warnings.append(f"rewrite_paragraph_not_object[{iid}][{p_idx}]")
            continue
        sources = para.get("sources") or []
        valid_sources = [s for s in sources if isinstance(s, str) and s in known_seg_ids]
        if not valid_sources:
            warnings.append(f"rewrite_paragraph_no_valid_sources[{iid}][{p_idx}]: {sources}")
            continue
        dups = [s for s in valid_sources if s in seen_ids]
        if dups:
            warnings.append(f"rewrite_paragraph_duplicate_sources[{iid}][{p_idx}]: {dups}")
        tgt_text = str(para.get("tgt_text") or "").strip()
        if not tgt_text:
            warnings.append(f"rewrite_paragraph_empty_tgt[{iid}][{p_idx}]")
            continue
        for sid in valid_sources:
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            tid, idx = seg_map.get(sid, ("", -1))
            node = by_id.get(tid)
            if not isinstance(node, dict) or idx < 0:
                continue
            segs = _get_segments(node) or [str(node.get("text") or "")]
            while len(segs) <= idx:
                segs.append("")
            segs[idx] = tgt_text
            _set_segments(node, segs)
            touched += 1

    if touched == 0:
        warnings.append(f"rewrite_selected_nothing[{iid}]")
    else:
        warnings.append(f"rewrite_applied[{iid}]: {touched} segments")
    # Rewrite keeps one fragment per source (no element-set change); it does not
    # force a visual re-flow on its own.
    return SkillResult(warnings=warnings, triggered_visual=False)


_REWRITE_PLAN_DOC = """  Rewrite text IN THE SAME LANGUAGE (never translates). Use for
  "改得更简洁/更正式/换成标题式/口语化/营销化/学术化/精简到 N 字".
  `objective` (natural language) MUST say BOTH:
    - WHICH text to rewrite (by content/role/position, e.g. "the main title",
      "the body paragraphs on the right"); and
    - HOW to rewrite it (the target style / tone / length, e.g. "make it more
      concise and punchy", "headline style, ≤ 12 characters").
  No `params` — the executor selects the target text and infers the style from
  your objective. Does NOT trigger a visual re-layout on its own."""


_REWRITE_ORDERING_NOTE = (
    "When combined with translate, rewriting usually happens BEFORE translate so "
    "the translation is made from the FINAL polished source text. Preference, not "
    "a rule — follow the user's stated order if they give one."
)


SKILL = Skill(
    id="text.rewrite",
    canonical_rank=1,
    summary="rewrite text in the same language (style/length change)",
    plan_doc=_REWRITE_PLAN_DOC,
    repair=_repair_rewrite_params,
    execute=_run_rewrite,
    ordering_note=_REWRITE_ORDERING_NOTE,
)
