"""Skill: text.redact — mask/redact sensitive spans in in-scope text.

Objective-driven: the planner passes a natural-language `objective` (e.g.
"redact all organization names in the footer"); no ids, no params. The executor
sees the WHOLE page, decides which items are in scope AND which spans to mask,
and the deterministic code below performs the actual masking.
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


_REDACT_SYSTEM_PROMPT = """You are the "REDACT" subagent for a single-page PPT-editing pipeline.

You are given:
- `user_request`: the user's full original instruction (context only).
- `objective`: what THIS redaction must do, in natural language — it names WHICH
  text to look at AND what kind of sensitive content to mask (e.g. "mask all
  organization names", "hide the email addresses in the footer").
- `items`: EVERY text item on the page (each with `id`, `kind`, `text`).

""" + _LOCATE_GUIDE + """

For each IN-SCOPE item, IDENTIFY the sensitive spans described by `objective`
(person names, company/organization names, emails, phone numbers, monetary
figures, addresses, id numbers, or specific literal strings the user named).

- DO NOT perform the replacement yourself. REPORT each span you found, exactly as
  it appears in the source `text`, in the `redactions` array. Downstream code
  does the masking.
- Return an item entry ONLY for in-scope items that actually contain something to
  redact. Items you omit are left untouched.

Output STRICT JSON only:
{
  "items": [
    { "id": "t0::seg0", "redactions": [ { "span": "<exact substring from text>",
      "kind": "organization|email|person_name|literal|..." } ] }
  ]
}

Hard rules:
- Each `redactions[i].span` MUST be a verbatim substring of that item's source
  `text` (case- and whitespace-exact). Do NOT report spans not present.
- Do NOT translate or rewrite anything. Only report spans.
- If nothing needs redacting, return an empty `items` array.
- Never invent ids. Output STRICT JSON only. No markdown. No commentary.
"""


def _repair_redact_params(params: dict[str, Any]) -> list[str]:
    # Redact is fully objective-driven now; no structured params.
    return []


def _apply_redaction(src_text: str, spans: list[str], replacement: str) -> tuple[str, list[str]]:
    if not src_text or not spans:
        return src_text, []

    uniq_spans: list[str] = []
    seen_spans: set[str] = set()
    for s in spans:
        if not isinstance(s, str) or not s or s in seen_spans:
            continue
        seen_spans.add(s)
        uniq_spans.append(s)
    uniq_spans.sort(key=lambda s: -len(s))

    offsets: list[tuple[int, int, str]] = []
    missing: list[str] = []
    for span in uniq_spans:
        start = 0
        found_any = False
        while True:
            idx = src_text.find(span, start)
            if idx < 0:
                break
            found_any = True
            offsets.append((idx, idx + len(span), span))
            start = idx + max(1, len(span))
        if not found_any:
            missing.append(span)

    if not offsets:
        return src_text, missing

    offsets.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    kept: list[tuple[int, int, str]] = []
    for start, end, span in offsets:
        if not any(not (end <= ks or start >= ke) for ks, ke, _ in kept):
            kept.append((start, end, span))

    result = src_text
    for start, end, span in sorted(kept, key=lambda x: -x[0]):
        result = result[:start] + replacement + result[end:]
    return result, missing


def _run_redact(
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
        warnings.append(f"redact_empty_objective[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)
    replacement = "<REDACTED>"

    texts: list[dict[str, Any]] = state.get("texts") or []
    by_id: dict[str, dict[str, Any]] = {str(t.get("id") or ""): t for t in texts if isinstance(t, dict)}
    all_ids = _all_text_ids(by_id)
    if not all_ids:
        warnings.append(f"redact_no_text_on_page[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    seg_items, seg_map, _seg_counts = _flatten_to_segment_items(
        ids=all_ids, by_id=by_id, selected=None
    )
    if not seg_items:
        warnings.append(f"redact_empty_segment_items[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    if dry_run:
        warnings.append(f"dry_run_stub_redact: {iid}")
        return SkillResult(warnings=warnings, triggered_visual=False)

    payload = {
        "user_request": user_request,
        "objective": objective,
        "replacement": replacement,
        "items": [{"id": str(it.get("id") or ""), "text": str(it.get("text") or "")} for it in seg_items],
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_REDACT_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=4096,
        tag="step2.redact",
        reasoning_effort="minimal",
    )
    if err:
        warnings.append(f"redact_call_error[{iid}]: {err}")
        return SkillResult(warnings=warnings, triggered_visual=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        warnings.append(f"redact_invalid_response[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    known_seg_ids = set(seg_map.keys())
    seg_text_by_id = {str(it.get("id") or ""): str(it.get("text") or "") for it in seg_items}
    seen: set[str] = set()
    touched = 0

    for r_idx, it in enumerate(obj.get("items") or []):
        if not isinstance(it, dict):
            warnings.append(f"redact_item_not_object[{iid}][{r_idx}]")
            continue
        sid = str(it.get("id") or "")
        if sid not in known_seg_ids:
            warnings.append(f"redact_unknown_id[{iid}][{r_idx}]: {sid}")
            continue
        if sid in seen:
            warnings.append(f"redact_duplicate_id[{iid}]: {sid}")
            continue
        seen.add(sid)

        redactions = it.get("redactions") if isinstance(it.get("redactions"), list) else []
        spans: list[str] = []
        for red in redactions:
            if isinstance(red, dict) and str(red.get("span") or ""):
                spans.append(str(red.get("span")))
        if not spans:
            continue

        src_text = seg_text_by_id.get(sid, "")
        redacted, missing_spans = _apply_redaction(src_text, spans, replacement)
        for ms in missing_spans:
            warnings.append(f"redact_span_not_found_in_source[{iid}][{sid}]: {ms[:40]}")
        if redacted == src_text:
            continue

        tid, idx = seg_map.get(sid, ("", -1))
        node = by_id.get(tid)
        if not isinstance(node, dict) or idx < 0:
            continue
        segs = _get_segments(node) or [str(node.get("text") or "")]
        while len(segs) <= idx:
            segs.append("")
        segs[idx] = redacted
        _set_segments(node, segs)
        touched += 1

    if touched == 0:
        warnings.append(f"redact_selected_nothing[{iid}]")
    else:
        warnings.append(f"redact_applied[{iid}]: {touched} segments")
    # Redaction masks spans but keeps the element set and rough sizes; no re-flow.
    return SkillResult(warnings=warnings, triggered_visual=False)


_REDACT_PLAN_DOC = """  Redact / mask sensitive spans in text (does NOT translate or rewrite unrelated
  words). Use for "打码/脱敏/隐藏公司名/hide the emails".
  `objective` (natural language) MUST say WHICH text to scan (by content/role/
  position, or "the whole page") AND what to mask (e.g. "mask all organization
  names", "hide every email address", or a specific literal like "remove the
  string 'Acme Corp'"). No `params` — the executor finds both the scope and the
  spans. Does NOT trigger a visual re-layout on its own."""


_REDACT_ORDERING_NOTE = (
    "Usually runs BEFORE any skill that re-emits or copies the text (rewrite, "
    "translate), so sensitive spans are removed before they can be rewritten or "
    "duplicated into another language. This is a safety concern, not just a style "
    "preference — keep redaction first unless the user explicitly wants otherwise."
)


SKILL = Skill(
    id="text.redact",
    canonical_rank=0,
    summary="mask/redact sensitive spans (same language, no rewrite)",
    plan_doc=_REDACT_PLAN_DOC,
    repair=_repair_redact_params,
    execute=_run_redact,
    ordering_note=_REDACT_ORDERING_NOTE,
    hard_before=frozenset({"text.rewrite", "text.translate"}),
)
