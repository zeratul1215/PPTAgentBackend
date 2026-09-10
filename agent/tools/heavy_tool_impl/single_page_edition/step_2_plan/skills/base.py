"""
Skill framework + shared execution skeleton for step2 content editing.

This module is intentionally self-contained (only stdlib + httpx) so the whole
`skills/` package could later be lifted to a cross-tool shared location without
dragging the plan/compile orchestrator with it. Each concrete skill lives in
its own module (`text_translate.py`, ...) and depends ONLY on this base.

The plan/compile stage no longer hard-codes a closed set of capability types.
Every content-editing capability is a self-describing `Skill`: it carries the
doc the planner needs to fill its params, a `repair` pass that plugs missing
params with safe defaults, a local `ordering_note` (a per-skill sequencing
suggestion shown to the planner) plus an optional `hard_before` set for the rare
safety-critical ordering edges, and an `execute` that mutates the cloned page
state. Adding a capability = registering one more `Skill` (see
`skills/__init__.py`).

Visual re-layout is deliberately NOT a skill. It is a single boolean phase
(`visual_intent.enabled`) handled by the fixed reference-image pipeline
downstream. A skill can REQUEST a visual re-layout at runtime by returning
`SkillResult.triggered_visual=True` when it changed the page's element set or
element geometry (e.g. bilingual translation roughly doubles the text volume).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class VisualDefault:
    """A default visual/layout requirement a skill contributes at runtime when
    it re-flows the page AND the user gave no visual requirement for that intent.

    The SAME semantic default is phrased for two different consumers, because
    the reference-image model wants COMPOSITION language while step3 wants
    concrete DOM/render discipline:
    - `image_text`: appended to the beautify reference-image prompt.
    - `step3_text`: appended to the step3 HTML-reassembly prompt.
    `key` is a stable identifier (for logging / de-duplication)."""

    key: str
    image_text: str
    step3_text: str


@dataclass
class SkillResult:
    """What a skill's `execute` reports back to the compiler."""

    warnings: list[str] = field(default_factory=list)
    # True when this skill changed the page's element set or element geometry
    # enough that the layout should be visually re-flowed downstream.
    triggered_visual: bool = False
    # A default visual requirement this execution suggests (e.g. bilingual
    # translation → "Chinese above English, shrunk to fit, no extra elements").
    # The compiler adopts it ONLY when the user gave no visual requirement for
    # this intent (`intent.visual_detail_provided` is false). None = nothing to
    # add. Skills that never re-flow leave this None.
    default_visual_detail: "VisualDefault | None" = None


@dataclass
class Skill:
    """A self-describing content-editing capability.

    `plan_doc` is spliced into the planner system prompt (so the planner knows
    how to fill `params`). `repair` fills missing/invalid params with safe
    defaults in place and returns a list of repair notes. `execute` mutates
    `state` (the cloned understand_output) and returns a `SkillResult`.

    Ordering is expressed LOCALLY, not by a global total order:
    - `ordering_note` is a short natural-language hint (spliced into the planner
      prompt) describing when this skill usually runs relative to others and WHY.
      It is a SUGGESTION the planner weighs against the user's stated intent; the
      user's explicit ordering always wins. Adding a skill = writing its own note,
      never re-deriving a global sequence.
    - `hard_before` is the rare exception: a set of skill ids that MUST run AFTER
      this one for CORRECTNESS/SAFETY reasons (e.g. redact before any skill that
      re-emits content, so sensitive spans are removed before they propagate).
      These few edges are enforced in code regardless of the planner. Keep this
      empty unless there is a real safety/correctness reason.
    - `canonical_rank` is retained ONLY as a stable tiebreaker for deterministic
      output ordering of otherwise-independent intents; it no longer forces
      `after` edges between skills.
    """

    id: str
    canonical_rank: int
    summary: str
    plan_doc: str
    repair: Callable[[dict[str, Any]], list[str]]
    execute: Callable[..., SkillResult]
    # Free-text ordering suggestion shown to the planner (empty = no preference).
    ordering_note: str = ""
    # Skill ids that MUST run after this one for safety/correctness (rare).
    hard_before: frozenset[str] = frozenset()
    # Coarse execution phase used to enforce a code-level global order at compile
    # time (Scheme A): every "text" skill runs to completion before any "table"
    # skill, so table skills always see the FINAL text nodes (translations,
    # rewrites) and stable ids. Within a phase, ordering is still decided by the
    # planner's `after` edges + the few `hard_before` safety edges. Phases:
    #   "text"  — edits the actual text of nodes (redact/rewrite/translate).
    #   "table" — organizes/reshapes/derives table structure over final text.
    phase: str = "text"


# Populated by `skills/__init__.py` (explicit registration in canonical order).
# Keyed by skill id; iteration order is the registration order.
SKILLS: dict[str, "Skill"] = {}


def _skill_ids() -> list[str]:
    return list(SKILLS.keys())


def _hard_ordering_edges() -> list[tuple[str, str]]:
    """Collect the few (before_skill_id, after_skill_id) edges that MUST hold for
    safety/correctness, declared LOCALLY by each skill's `hard_before`.

    Returns edges as (A, B) meaning: a B-intent must run AFTER an A-intent when
    both are present. This is the only cross-skill ordering enforced in code;
    everything else is left to the planner (guided by `ordering_note`).
    """
    edges: list[tuple[str, str]] = []
    for skill in SKILLS.values():
        for after_id in skill.hard_before:
            edges.append((skill.id, str(after_id)))
    return edges


def _build_skill_docs() -> str:
    """Concatenate every registered skill's planner doc (order = registration).

    Each skill's own `ordering_note` (if any) is appended as a SUGGESTION so the
    planner reasons about sequencing per-skill, weighing it against the user's
    stated intent, rather than following a hard-coded global order.
    """
    parts: list[str] = []
    for skill in SKILLS.values():
        block = f"* skill id = \"{skill.id}\" — {skill.summary}\n{skill.plan_doc.strip()}"
        note = (skill.ordering_note or "").strip()
        if note:
            block += f"\n  ordering suggestion: {note}"
        parts.append(block)
    return "\n\n".join(parts)


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)
_STRAY_FENCE_LINE_RE = re.compile(r"(?im)^\s*```(?:json)?\s*$")


def _normalize_model_output(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return ""
    m = _FENCE_RE.search(t)
    if m:
        return (m.group(1) or "").strip()
    t = _STRAY_FENCE_LINE_RE.sub("", t).strip()
    t = re.sub(r"(?is)^\s*```(?:json)?", "", t).strip()
    t = re.sub(r"(?is)```\s*$", "", t).strip()
    return t


def _loads_json_with_light_repair(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a model JSON object, repairing only syntax that is unambiguous."""
    text = _normalize_model_output(raw)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None, None
    except Exception as first_exc:
        # Common LLM slip: a trailing comma before } or ]. This is deterministic
        # syntax repair and does not invent or alter any field values.
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        if repaired != text:
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj, f"json_repaired: trailing_comma_after_{type(first_exc).__name__}"
            except Exception:
                pass
        return None, f"json_parse_error: {type(first_exc).__name__}: {first_exc}"


def _downscale_jpeg(raw: bytes, *, max_edge: int = 1568, quality: int = 85) -> bytes:
    """Resize an image so its long edge is <= max_edge and re-encode as JPEG,
    to keep the multimodal request within the token/context budget. Falls back
    to the original bytes if PIL is unavailable or anything fails."""
    try:
        from io import BytesIO

        from PIL import Image  # type: ignore

        img = Image.open(BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_edge) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        return buf.getvalue()
    except Exception:
        return raw


def _llm_timeout_s(default: float = 300.0) -> float:
    """Per-request HTTP timeout for text-LLM calls, in seconds.

    Configurable at launch via the `PPT_LLM_TIMEOUT_S` env var so ops can retune
    it without a code change (e.g. Opus 4.8 on large prompts routinely needs more
    than the old hard-coded 60s). Falls back to `default` on missing/invalid
    values, and floors at 1s so a typo can't disable the timeout entirely."""
    raw = (os.getenv("PPT_LLM_TIMEOUT_S") or "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val >= 1.0 else default


def _log_llm_timing(*, tag: str, model: str, attempt: int, elapsed: float, status: Any) -> None:
    """Temporary instrumentation: print how long each text-LLM request took.

    Emitted to stderr with a stable `[llm_timing]` prefix so it can be grepped
    out of the uvicorn log. Meant to validate whether the 180s per-request
    timeout is oversized for Opus 4.8 (most calls are expected to finish well
    under 30s). Remove once the timeout has been retuned."""
    import sys

    print(
        f"[llm_timing] tag={tag} model={model} attempt={attempt} "
        f"elapsed={elapsed:.2f}s status={status}",
        file=sys.stderr,
        flush=True,
    )


def _call_claude_json(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    labeled_images: list[tuple[str, bytes]] | None = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 2,
    tag: str = "step2",
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Call the OpenAI-compatible chat-completions endpoint (a Claude proxy set
    via PPT_LLM_BASE_URL + PPT_LLM_API_KEY). Supports optional inline images
    (each preceded by a short text label). We rely on the prompts' "STRICT JSON
    only" instruction plus the existing validators/repair passes rather than a
    server-side response schema. Returns (raw_text, parsed_obj_or_none, error).

    `reasoning_effort` (OpenAI-style: "minimal"|"low"|"medium"|"high"): when set,
    it is injected into the request body. Skills whose execute is a simple,
    well-scoped transform (rewrite/redact/translate/table.build) pass "minimal"
    to keep latency low now that each execute also does its own scope selection.
    """
    base_url = (os.getenv("PPT_LLM_BASE_URL") or "").strip()
    api_key = (os.getenv("PPT_LLM_API_KEY") or "").strip()
    if not base_url or not api_key:
        return "", None, "missing_env: set PPT_LLM_BASE_URL and PPT_LLM_API_KEY"

    def _img_part(b: bytes) -> dict[str, Any]:
        jb = _downscale_jpeg(b)
        b64 = base64.b64encode(jb).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for label, b in labeled_images or []:
        if label:
            content.append({"type": "text", "text": label})
        content.append(_img_part(b))

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }
    effort = (reasoning_effort or "").strip()
    if effort:
        body["reasoning_effort"] = effort
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.4.0",
    }
    url = base_url.rstrip("/") + "/chat/completions"

    last_err: str | None = None
    for attempt in range(max(0, int(retries)) + 1):
        t0 = time.monotonic()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=_llm_timeout_s())
            _log_llm_timing(tag=tag, model=model, attempt=attempt, elapsed=time.monotonic() - t0, status=resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                raw = str(data["choices"][0]["message"]["content"] or "")
                obj, parse_note = _loads_json_with_light_repair(raw)
                if obj is not None:
                    return raw, obj, parse_note
                return raw, None, parse_note
            last_err = f"http_{resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as e:
            _log_llm_timing(tag=tag, model=model, attempt=attempt, elapsed=time.monotonic() - t0, status=type(e).__name__)
            last_err = f"{type(e).__name__}: {e}"
        if attempt < int(retries):
            time.sleep(min(30.0, 2.0 * (2.0 ** attempt)))
    return "", None, f"model_call_failed: {last_err}"
def _parse_segment_range(spec: str, seg_count: int) -> list[int]:
    """Parse the segment-selector part of an "id#<spec>" scope entry.

    `spec` supports comma-separated ranges/indices, all 0-based and inclusive:
      "0-4"      -> [0,1,2,3,4]
      "0,2,5"    -> [0,2,5]
      "2-"       -> [2 .. seg_count-1]
      "3"        -> [3]
    Indices are clamped to [0, seg_count); out-of-range / malformed pieces are
    dropped. Returns a sorted, de-duplicated list of valid indices.
    """
    if seg_count <= 0:
        return []
    picked: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, hi_s = piece.split("-", 1)
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            try:
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else seg_count - 1
            except ValueError:
                continue
            if lo < 0:
                lo = 0
            if hi > seg_count - 1:
                hi = seg_count - 1
            for i in range(lo, hi + 1):
                picked.add(i)
        else:
            try:
                i = int(piece)
            except ValueError:
                continue
            if 0 <= i < seg_count:
                picked.add(i)
    return sorted(picked)


def _resolve_scope_selection(
    scope: Any,
    by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, list[int]], list[str]]:
    """Resolve a scope value into (ordered text ids, per-id segment selection).

    Each scope entry is either a bare id ("t2") meaning the WHOLE node, or an
    "id#<spec>" entry (e.g. "t2#0-4") selecting specific 0-based segments.
    A bare id (or "page") maps to ALL of the node's segment indices.

    Returns:
      - ids: ordered list of in-scope text ids (deduplicated, order preserved)
      - selection: tid -> sorted list of selected segment indices
      - warnings: notes about dropped/clamped selectors
    """
    warnings: list[str] = []
    ids: list[str] = []
    selection: dict[str, list[int]] = {}

    def _seg_count(tid: str) -> int:
        node = by_id.get(tid)
        if not isinstance(node, dict):
            return 0
        return len(_get_segments(node))

    if scope == "page":
        entries = [tid for tid in by_id.keys()]
    elif isinstance(scope, list):
        entries = [s for s in scope if isinstance(s, str)]
    else:
        entries = []

    for entry in entries:
        tid, _, spec = entry.partition("#")
        tid = tid.strip()
        if tid not in by_id:
            continue
        count = _seg_count(tid)
        if spec.strip():
            idxs = _parse_segment_range(spec.strip(), count)
            if not idxs:
                warnings.append(f"scope_segment_selector_empty_after_clamp: {entry}")
                continue
        else:
            idxs = list(range(count))
        if tid in selection:
            merged = sorted(set(selection[tid]) | set(idxs))
            selection[tid] = merged
        else:
            ids.append(tid)
            selection[tid] = idxs
    return ids, selection, warnings


def _get_segments(node: dict[str, Any]) -> list[str]:
    segs = node.get("segments")
    if isinstance(segs, list) and segs and all(isinstance(x, str) for x in segs):
        return [str(x) for x in segs]
    txt = str(node.get("text") or "")
    return [txt] if txt else []


def _set_segments(node: dict[str, Any], segments: list[str]) -> None:
    segs = [str(x) for x in segments if isinstance(x, str)]
    node["segments"] = segs
    node["text"] = "".join(segs)


def _flatten_to_segment_items(
    *,
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    selected: dict[str, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, int]], dict[str, int]]:
    """Expand text ids to segment-level items for stable structure transforms.

    When `selected` is provided, only the listed segment indices per text id are
    emitted as items; unselected segments are left untouched by the caller. When
    `selected` is None, every segment of each id is emitted (whole-node mode).

    Returns:
      - items: list of payload items (id/kind/text)
      - seg_map: seg_id -> (tid, seg_idx)
      - seg_counts: tid -> number of segments
    """
    items: list[dict[str, Any]] = []
    seg_map: dict[str, tuple[str, int]] = {}
    seg_counts: dict[str, int] = {}
    for tid in ids:
        node = by_id.get(tid)
        if not isinstance(node, dict):
            continue
        segs = _get_segments(node)
        if not segs:
            segs = [str(node.get("text") or "")]
        seg_counts[tid] = len(segs)
        if selected is not None:
            want = [i for i in selected.get(tid, []) if 0 <= i < len(segs)]
        else:
            want = list(range(len(segs)))
        for idx in want:
            seg_text = segs[idx]
            seg_id = f"{tid}::seg{idx}"
            seg_map[seg_id] = (tid, idx)
            items.append(
                {
                    "id": seg_id,
                    "kind": str(node.get("kind") or ""),
                    "text": str(seg_text or ""),
                }
            )
    return items, seg_map, seg_counts


# ---------------------------------------------------------------------------
# Shared "objective -> in-scope nodes" guidance for execute-side scope selection
# ---------------------------------------------------------------------------
#
# Intents no longer carry a `scope` id-list. The planner emits a natural-language
# `objective`; each text skill's execute is handed the WHOLE page and must decide
# for itself which nodes/segments the objective refers to, then transform ONLY
# those. This guide is the label-mapping + segment-reasoning that used to live in
# the planner prompt (R3 + segment selector), moved here so the capability is not
# lost — every text skill splices it into its system prompt.
_LOCATE_GUIDE = """SELECTING WHICH ITEMS ARE IN SCOPE (do this yourself):

You are given EVERY text item on the page (each `items[i]` is one visible
line/segment, with its `kind`). Read `objective` (what THIS skill must do)
together with `user_request` (the user's full original wording, for context
only) and decide which items the objective actually refers to.
Return output ONLY for the in-scope items; leave every other item out entirely
(items you omit are left untouched).

How to map a natural-language target to items:
- Whole page / no target word ("整页/全部文字/the whole page", or a bare
  "translate to English") → ALL items are in scope.
- Role / part words → match on `kind`:
    * "标题/title/heading" → kind in {title, heading, subheading}
    * "正文/body/paragraph" → kind in {body, paragraph, bullet_item}
    * "页脚/footer" → kind in {footer, footnote, caption, label}
- Content words ("提到 2016 的那几条 / the entries about revenue") → read each
  item's `text` and pick the ones that match.
- Ordinal words ("前五条/第2到第4条/最后一条") → pick items by their order
  (items are given in natural reading order, top-to-bottom).

Rules:
- Do NOT widen a specific target to the whole page. If the objective clearly
  points at one region, only return items in that region.
- If you genuinely cannot tell which items match, return an empty result rather
  than guessing wrong.
- NEVER invent item ids. Only use the `id` values given to you."""


def _all_text_ids(by_id: dict[str, dict[str, Any]]) -> list[str]:
    """All non-empty text-node ids on the page, in payload order."""
    return [tid for tid in by_id.keys() if str(by_id[tid].get("text") or "").strip()]


def _append_parallel_translation(
    *,
    state: dict[str, Any],
    source_id: str,
    tgt_text: str,
    tgt_segments: list[str] | None = None,
) -> str:
    """Append a new text fragment carrying the translation of `source_id`.

    The new fragment inherits `kind` from the source, and carries a
    `translation_of` link back to the source id. A provisional id is assigned
    (`__tr_<source_id>__<n>`) which is later renumbered to a regular `tN` by
    `_compact_state_text_ids`.

    Returns the provisional id of the new fragment.
    """
    texts = state.get("texts")
    if not isinstance(texts, list):
        return ""
    src = next((t for t in texts if isinstance(t, dict) and t.get("id") == source_id), None)
    kind = str(src.get("kind") or "") if isinstance(src, dict) else ""

    existing_ids = {str(t.get("id") or "") for t in texts if isinstance(t, dict)}
    n = 0
    while True:
        new_id = f"__tr_{source_id}__{n}"
        if new_id not in existing_ids:
            break
        n += 1
    segs = (
        [str(x) for x in tgt_segments if isinstance(x, str)]
        if isinstance(tgt_segments, list) and tgt_segments
        else [tgt_text]
    )
    texts.append(
        {
            "id": new_id,
            "kind": kind,
            "segments": segs,
            "text": "".join(segs),
            "translation_of": source_id,
        }
    )
    return new_id


# ---------------------------------------------------------------------------
# Shared image helpers (used by image.add / image.delete / image.replace)
# ---------------------------------------------------------------------------
#
# Image skills mutate `state["images"]` (the "content list" of pictures the
# reference-image model and step3 consume) exactly like text skills mutate
# `state["texts"]`. Files live in the page's STABLE bundle dir
# (`state["bundle_dir"]`, kept alive across turns by paths.page_assets_dir); the
# external `stage_page_asset` tool drops user uploads there BEFORE the pipeline
# runs. These helpers centralise the file/LLM plumbing so each skill stays thin.

_PENDING_UPLOADS_NAME = "_pending_uploads.json"


def _bundle_dir_from_state(state: dict[str, Any]) -> "os.PathLike[str] | None":
    """Resolve `state["bundle_dir"]` to a real directory Path, or None.

    This is the page's stable asset dir: baseline/reread materialise the slide's
    images here and the prestage tool drops chat uploads here, so both existing
    and newly-uploaded image bytes resolve against it by bare filename."""
    from pathlib import Path

    raw = str((state or {}).get("bundle_dir") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def _read_image_pixels(path: "os.PathLike[str] | str") -> tuple[int, int] | None:
    """Return (width_px, height_px) of an image file, or None on any failure.

    Used only for the ASPECT RATIO: image skills store these raw px values under
    `display_w_pt/h_pt` as a SOFT reference (no DPI→pt conversion — that would be
    unfounded) so step3/the reference-image model know portrait-vs-landscape and
    don't stuff a tall photo into a wide frame. Downstream may rescale freely."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            w, h = img.size
        w, h = int(w), int(h)
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


_IMAGE_DESC_SYSTEM_PROMPT = """You describe ONE image for a slide-editing pipeline's content list.

You are given the image itself and, optionally, a `user_note` (the user's own
words about what this picture is / how they want it used). Write ONE concise
English description (1-3 sentences) covering:
- what it shows (high level; do NOT OCR or transcribe text inside it),
- its likely role on a slide (hero / supporting / icon / logo / chart / photo / background).

If `user_note` is given, let it steer the description (respect the user's intent
and any naming they use), but still describe what you actually see. Do NOT invent
a placement — the layout is decided later.

Output STRICT JSON only, no markdown:
{ "description_en": "<one to three sentences>" }"""


def _gen_image_description_en(
    *,
    image_bytes: bytes,
    model: str,
    user_note: str = "",
    dry_run: bool = False,
) -> tuple[str, list[str]]:
    """Generate an English `description_en` for one image via the vision model.

    Returns (description, warnings). On dry-run or any failure returns a short
    safe fallback so the skill can still register the image (an empty
    description would make it invisible to the reference-image model/step3)."""
    warnings: list[str] = []
    note = (user_note or "").strip()
    if dry_run:
        return (note[:200] if note else "user-provided image"), ["dry_run_stub_image_desc"]
    if not image_bytes:
        return (note[:200] if note else "user-provided image"), ["image_desc_no_bytes"]
    payload = {"user_note": note}
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_IMAGE_DESC_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False),
        labeled_images=[("The image to describe:", image_bytes)],
        max_output_tokens=512,
        tag="step2.image_desc",
        reasoning_effort="minimal",
    )
    if err or not isinstance(obj, dict):
        warnings.append(f"image_desc_call_error: {err or 'invalid_response'}")
        return (note[:200] if note else "user-provided image"), warnings
    desc = str(obj.get("description_en") or "").strip()
    if not desc:
        warnings.append("image_desc_empty")
        return (note[:200] if note else "user-provided image"), warnings
    return desc, warnings


def _next_add_image_id(state: dict[str, Any]) -> str:
    """Mint a turn-unique `add_i{n}` id that doesn't collide with any existing
    image id. Only needs to be unique within this turn: after the new image
    lands as a PPTist element, the next reread renumbers everything to the
    regular `{page_id}_i{i}` scheme, so no cross-turn stability is required."""
    images = state.get("images")
    existing = {
        str(im.get("id") or "")
        for im in (images if isinstance(images, list) else [])
        if isinstance(im, dict)
    }
    n = 0
    while True:
        cand = f"add_i{n}"
        if cand not in existing:
            return cand
        n += 1


def _read_pending_uploads(bundle_dir: "os.PathLike[str] | str") -> list[dict[str, Any]]:
    """Read the page's pending-uploads manifest (list of {filename, user_note})
    that `stage_page_asset` wrote into the bundle dir. Missing/malformed → []."""
    from pathlib import Path

    b = Path(bundle_dir)
    p = b / _PENDING_UPLOADS_NAME
    if not p.is_file() and b.name == "source":
        p = b.parent / "uploads" / _PENDING_UPLOADS_NAME
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    items = data.get("uploads") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict) and str(it.get("filename") or "").strip():
            out.append(
                {
                    "filename": str(it["filename"]).strip(),
                    "user_note": str(it.get("user_note") or "").strip(),
                }
            )
    return out


def _clear_pending_uploads(bundle_dir: "os.PathLike[str] | str") -> None:
    """Consume the pending-uploads manifest so a later turn won't re-add the same
    files. The image bytes stay in the bundle (they're now real state images)."""
    from pathlib import Path

    b = Path(bundle_dir)
    for p in (b / _PENDING_UPLOADS_NAME, b.parent / "uploads" / _PENDING_UPLOADS_NAME):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def _resolve_pending_upload_file(bundle_dir: "os.PathLike[str] | str", filename: str) -> Path:
    """Return a stable source asset, promoting a staged upload when needed.

    Staged files are run-isolated (``uploads/<run_id>/<hash>.<ext>``), while
    content-list ``src`` values must resolve from the flat, stable ``source``
    directory. Promote by basename so Step 2.5 and Step 3 share one durable
    asset namespace after the pending manifest is consumed.
    """
    from pathlib import Path
    import shutil

    b = Path(bundle_dir)
    raw = Path(str(filename or ""))
    if not raw.name or raw.is_absolute() or ".." in raw.parts:
        return b / "__invalid_pending_upload__"

    stable = b / raw.name
    if stable.is_file():
        return stable

    legacy_direct = b / raw
    if legacy_direct.is_file():
        return legacy_direct

    if b.name == "source":
        uploaded = b.parent / "uploads" / raw
        if uploaded.is_file():
            b.mkdir(parents=True, exist_ok=True)
            if not stable.exists():
                try:
                    shutil.copy2(uploaded, stable)
                except OSError:
                    return uploaded
            return stable if stable.exists() else uploaded
    return stable


def _delete_bundle_file(bundle_dir: "os.PathLike[str] | str", src: str) -> bool:
    """Delete one image file (by bare `src` name) from the bundle dir.

    Used by delete/replace. Only same-name, in-bundle files are removed; data:
    URIs / absolute / remote srcs are ignored. Failures are swallowed (best
    effort; user accepted the no-rollback risk)."""
    from pathlib import Path

    s = (src or "").strip()
    if not s or s.startswith("data:") or "://" in s or s.startswith("/"):
        return False
    name = Path(s).name
    if not name:
        return False
    target = Path(bundle_dir) / name
    try:
        if target.is_file():
            target.unlink()
            return True
    except OSError:
        pass
    return False


_IMAGE_MATCH_SYSTEM_PROMPT = """You select which picture(s) a user's instruction refers to, for a slide-editing pipeline.

You are given:
- `user_request`: the user's full original wording (context).
- `objective`: what THIS skill must do, naming WHICH image(s) to act on (by
  content/role, e.g. "delete the logo top-right", "replace the team photo").
- `images`: every image currently on the page, each `{id, description_en}`.

Read `objective` (with `user_request` for context) and pick the image ids it
refers to, matching against each image's `description_en`. Choose the smallest
set that clearly satisfies the instruction. If you genuinely cannot tell which
image is meant, return an empty list rather than guessing wrong.

Output STRICT JSON only, no markdown:
{ "ids": ["<image id>", ...] }

Never invent ids — only use ids present in `images`."""


def _match_image_ids(
    *,
    intent: dict[str, Any],
    state: dict[str, Any],
    model: str,
    user_request: str,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    """Semantically resolve which image ids an objective refers to.

    Returns (ids, warnings). Empty ids = "couldn't tell / nothing matched"; the
    caller should skip rather than act. Mirrors the text skills' `_LOCATE_GUIDE`
    approach but over `state["images"]` and their `description_en`."""
    warnings: list[str] = []
    objective = str(intent.get("objective") or "").strip()
    images = [im for im in (state.get("images") or []) if isinstance(im, dict)]
    known_ids = {str(im.get("id") or "") for im in images if str(im.get("id") or "")}
    if not objective:
        return [], ["image_match_empty_objective"]
    if not images:
        return [], ["image_match_no_images_on_page"]
    if dry_run:
        return [], ["dry_run_stub_image_match"]
    payload = {
        "user_request": user_request,
        "objective": objective,
        "images": [
            {"id": str(im.get("id") or ""), "description_en": str(im.get("description_en") or "")}
            for im in images
        ],
    }
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_IMAGE_MATCH_SYSTEM_PROMPT,
        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
        max_output_tokens=512,
        tag="step2.image_match",
        reasoning_effort="minimal",
    )
    if err or not isinstance(obj, dict) or not isinstance(obj.get("ids"), list):
        warnings.append(f"image_match_call_error: {err or 'invalid_response'}")
        return [], warnings
    picked: list[str] = []
    for x in obj.get("ids") or []:
        sid = str(x or "").strip()
        if sid in known_ids and sid not in picked:
            picked.append(sid)
        elif sid and sid not in known_ids:
            warnings.append(f"image_match_unknown_id: {sid}")
    return picked, warnings


def _append_added_text(
    *,
    state: dict[str, Any],
    text: str,
    kind: str = "body",
    segments: list[str] | None = None,
) -> str:
    """Append a NEW, user-requested text node (used by `text.add`) and return its
    provisional id.

    The node carries ONLY `kind` + content — deliberately NO position. Placement
    is decided downstream by the reference-image model / step3 from page semantics
    (there is no reusable default location for "a paragraph the user asked to add",
    unlike bilingual translation). It flows into `required_refs` like any other
    text node, so step3 renders it as a normal ref.

    A provisional id (`__add_<n>`) is assigned and later renumbered to a regular
    `tN` by `_compact_state_text_ids`; it only needs to be unique within this turn
    (reread renumbers everything the next turn), so no cross-turn stability and no
    new id space are introduced (mirrors `__tr_` / `__dv_`).
    """
    texts = state.get("texts")
    if not isinstance(texts, list):
        texts = []
        state["texts"] = texts
    existing_ids = {str(t.get("id") or "") for t in texts if isinstance(t, dict)}
    n = 0
    while True:
        new_id = f"__add_{n}"
        if new_id not in existing_ids:
            break
        n += 1
    segs = (
        [str(x) for x in segments if isinstance(x, str) and x != ""]
        if isinstance(segments, list) and segments
        else []
    )
    if not segs:
        segs = [str(text)]
    node: dict[str, Any] = {
        "id": new_id,
        "kind": (kind or "body"),
        "segments": segs,
        "text": "".join(segs),
    }
    texts.append(node)
    return new_id


def _append_derived_text(
    *,
    state: dict[str, Any],
    text: str,
    kind: str = "body",
    derived_from: list[str] | None = None,
    derived_op: str = "",
) -> str:
    """Append a NEW text node holding a value DERIVED by deterministic Python
    (e.g. a column sum computed by `table.compute`), and return its provisional
    id.

    The value is computed by code, never by the model — the model only picks the
    operator/columns/target. The node carries `derived_from` (the source text ids
    that fed the computation) and `derived_op` (the operator name) purely as
    provenance metadata; downstream renderers treat it like any other text node.
    A provisional id (`__dv_<n>`) is assigned and later renumbered to a regular
    `tN` by `_compact_state_text_ids`.
    """
    texts = state.get("texts")
    if not isinstance(texts, list):
        return ""
    existing_ids = {str(t.get("id") or "") for t in texts if isinstance(t, dict)}
    n = 0
    while True:
        new_id = f"__dv_{n}"
        if new_id not in existing_ids:
            break
        n += 1
    node: dict[str, Any] = {
        "id": new_id,
        "kind": kind or "body",
        "segments": [text],
        "text": text,
    }
    if derived_from:
        node["derived_from"] = [str(x) for x in derived_from if isinstance(x, str)]
    if derived_op:
        node["derived_op"] = str(derived_op)
    texts.append(node)
    return new_id
