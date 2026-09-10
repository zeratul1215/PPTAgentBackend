"""Skill: text.translate — translate in-scope text (replace, or bilingual).

Objective-driven: the planner passes a natural-language `objective`; no ids, no
params. The executor sees the WHOLE page, decides which items are in scope, and
also decides PER ITEM whether to REPLACE the source with its translation or KEEP
the source and ADD a parallel translation (bilingual) — driven entirely by the
objective's wording (e.g. "make bilingual" vs "translate, replacing the source").
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    _LOCATE_GUIDE,
    Skill,
    SkillResult,
    VisualDefault,
    _all_text_ids,
    _append_parallel_translation,
    _call_claude_json,
    _flatten_to_segment_items,
    _get_segments,
    _set_segments,
)


# Default bilingual layout, used ONLY when the user gave no visual requirement
# for the translate intent (intent.visual_detail_provided is false). Chinese
# sits directly above its English translation, shrunk to fit the original box,
# with no extra separators/boxes/elements added.
_BILINGUAL_VISUAL_DEFAULT = VisualDefault(
    key="bilingual_zh_over_en",
    image_text=(
        "Bilingual layout: place each source line directly ABOVE its translation "
        "as one tight pair that shares the original block's position. Make the "
        "translation slightly smaller with tighter line spacing so the pair fits "
        "the original area. Do NOT add any separators, divider lines, translation "
        "background boxes, or any other extra element."
    ),
    step3_text=(
        "Bilingual text: for each translated block, render the source language on "
        "top and its translation immediately below WITHIN THE SAME text container "
        "(do not create a new column/box). Set the translation font-size to about "
        "0.85 of the source and tighten line-height so the pair fits the source "
        "block's footprint. Never insert separator lines, background boxes, or any "
        "extra decorative element."
    ),
)


_TRANSLATE_SYSTEM_PROMPT = """You are the "TRANSLATE" subagent for a single-page PPT-editing pipeline.

You are given:
- `user_request`: the user's full original instruction (context only).
- `objective`: what THIS translation must do, in natural language. It tells you
  WHICH text to translate, the TARGET language, and whether the result should be
  BILINGUAL (keep the source and add a translation next to it) or a REPLACEMENT
  (overwrite the source with its translation).
- `items`: EVERY text item on the page (each with `id`, `kind`, `text`).

""" + _LOCATE_GUIDE + """

Then, for each IN-SCOPE item:

1) **Translate** its `text` into the target language named in `objective`.
   - If an item is ALREADY entirely in the target language, set `tgt_text` equal
     to its source verbatim (do not retranslate or "improve").
   - Do NOT paraphrase, add, or drop information.
   - Preserve placeholder tokens (`<ORG>`, `<REDACTED>`, ...) verbatim.

2) **Choose `mode`** for the item from the objective:
   - "bilingual" — keep the source AND add this translation as a new parallel
     fragment. Use when the objective says "双语/中英对照/keep both/add a
     translation".
   - "replace" — overwrite the source with the translation. Use when the
     objective says "翻译成X/translate to X/换成X" with no "keep both" sense.
   If the objective mixes modes for different parts (e.g. "replace the title but
   make the body bilingual"), set `mode` per item accordingly.

Defaults when the objective is silent:
- TARGET LANGUAGE: if the objective does not name a target language, translate
  into English.
- MODE: if the objective does not explicitly ask to keep the source (no "双语/
  中英对照/keep both/bilingual/add a translation" sense), default `mode` =
  "replace" (overwrite the source; do NOT keep the original). Choose "bilingual"
  ONLY when the objective explicitly asks to keep both.

3) **Track sources**: emit ONE paragraph per in-scope item, `sources` = [that id].

Hard rules:
- Emit paragraphs ONLY for in-scope items; omit the rest (they stay unchanged).
- Each in-scope id appears in exactly ONE `sources`. Never invent ids.
- `paragraphs[i].kind` MUST equal the source item's kind.
- If NOTHING matches, return an empty `paragraphs` array.
- Output STRICT JSON only. No markdown. No commentary.

Output format:
{
  "paragraphs": [
    { "kind": "title|body|bullet_item|...", "sources": ["t0::seg0"],
      "src_text": "...", "tgt_text": "...", "mode": "bilingual|replace" }
  ]
}
"""


def _repair_translate_params(params: dict[str, Any]) -> list[str]:
    # Translate is fully objective-driven now; no structured params.
    return []


def _run_translate(
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
        warnings.append(f"translate_empty_objective[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    texts: list[dict[str, Any]] = state.get("texts") or []
    by_id: dict[str, dict[str, Any]] = {str(t.get("id") or ""): t for t in texts if isinstance(t, dict)}
    all_ids = _all_text_ids(by_id)
    if not all_ids:
        warnings.append(f"translate_no_text_on_page[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    seg_items, seg_map, _seg_counts = _flatten_to_segment_items(
        ids=all_ids, by_id=by_id, selected=None
    )
    if not seg_items:
        warnings.append(f"translate_empty_segment_items[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    if dry_run:
        warnings.append(f"dry_run_stub_translate: {iid}")
        return SkillResult(warnings=warnings, triggered_visual=False)

    payload = {
        "user_request": user_request,
        "objective": objective,
        "items": seg_items,
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_TRANSLATE_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=4096,
        tag="step2.translate",
        reasoning_effort="minimal",
    )
    if err:
        warnings.append(f"translate_call_error[{iid}]: {err}")
        return SkillResult(warnings=warnings, triggered_visual=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("paragraphs"), list):
        warnings.append(f"translate_invalid_response[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    known_seg_ids = set(seg_map.keys())
    seen_ids: set[str] = set()
    # Collect per-node replacements and per-node bilingual additions.
    replace_by_tid: dict[str, dict[int, str]] = {}
    bilingual_by_tid: dict[str, dict[int, str]] = {}

    for p_idx, para in enumerate(obj["paragraphs"]):
        if not isinstance(para, dict):
            warnings.append(f"translate_paragraph_not_object[{iid}][{p_idx}]")
            continue
        sources = para.get("sources") or []
        valid_sources = [s for s in sources if isinstance(s, str) and s in known_seg_ids]
        if not valid_sources:
            warnings.append(f"translate_paragraph_no_valid_sources[{iid}][{p_idx}]: {sources}")
            continue
        tgt_text = str(para.get("tgt_text") or "").strip()
        if not tgt_text:
            warnings.append(f"translate_paragraph_empty_tgt[{iid}][{p_idx}]")
            continue
        mode = str(para.get("mode") or "replace").strip().lower()
        if mode not in ("bilingual", "replace"):
            mode = "replace"
        for sid in valid_sources:
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            tid, idx = seg_map.get(sid, ("", -1))
            if not tid or idx < 0:
                continue
            if mode == "bilingual":
                bilingual_by_tid.setdefault(tid, {})[idx] = tgt_text
            else:
                replace_by_tid.setdefault(tid, {})[idx] = tgt_text

    # Apply replacements in place.
    for tid, idx_map in replace_by_tid.items():
        node = by_id.get(tid)
        if not isinstance(node, dict):
            continue
        segs = _get_segments(node) or [str(node.get("text") or "")]
        for idx, tgt in idx_map.items():
            while len(segs) <= idx:
                segs.append("")
            segs[idx] = tgt
        _set_segments(node, segs)

    # Append one parallel translation node per source node (bilingual).
    for tid, idx_map in bilingual_by_tid.items():
        tgt_segments = [idx_map[i] for i in sorted(idx_map.keys())]
        if tgt_segments:
            _append_parallel_translation(
                state=state,
                source_id=tid,
                tgt_text="".join(tgt_segments),
                tgt_segments=tgt_segments,
            )

    touched = len(seen_ids)
    if touched == 0:
        warnings.append(f"translate_selected_nothing[{iid}]")
    else:
        warnings.append(
            f"translate_applied[{iid}]: {touched} segments "
            f"(replace={sum(len(v) for v in replace_by_tid.values())}, "
            f"bilingual={sum(len(v) for v in bilingual_by_tid.values())})"
        )
    # Bilingual roughly doubles text volume → request a visual re-flow AND supply
    # a default bilingual layout (adopted only if the user gave no visual
    # requirement for this intent). Pure replacement keeps volume → neither.
    triggered_visual = bool(bilingual_by_tid)
    return SkillResult(
        warnings=warnings,
        triggered_visual=triggered_visual,
        default_visual_detail=_BILINGUAL_VISUAL_DEFAULT if triggered_visual else None,
    )


_TRANSLATE_PLAN_DOC = """  Translate text into another language. Use for "翻译成英文/日文", "做成中英双语",
  "add an English translation".
  `objective` (natural language) MUST say: WHICH text to translate (by content/
  role/position, or "the whole page"), the TARGET language, and whether the
  result is BILINGUAL (keep source + add translation) or a REPLACEMENT (overwrite
  the source). Examples:
    - "Make the whole page bilingual: keep the Chinese and add an English
      translation of every item."   (bilingual)
    - "Translate the title into Japanese, replacing the original."  (replace)
  No `params` — the executor selects the target text and the bilingual/replace
  mode from your objective (it may even use different modes for different parts if
  your objective says so). Bilingual mode DOES request a visual re-layout;
  pure replacement does NOT."""


_TRANSLATE_ORDERING_NOTE = (
    "Usually runs LAST among content edits: translate the FINAL text, after any "
    "redaction (so masked spans aren't translated) and any rewrite (so the "
    "polished wording is what gets translated). Preference, not a rule — honor "
    "the user's explicit order if they state one."
)


SKILL = Skill(
    id="text.translate",
    canonical_rank=2,
    summary="translate text (replace, or bilingual — decided from the objective)",
    plan_doc=_TRANSLATE_PLAN_DOC,
    repair=_repair_translate_params,
    execute=_run_translate,
    ordering_note=_TRANSLATE_ORDERING_NOTE,
)
