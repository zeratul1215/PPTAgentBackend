"""
Step 2 (MVP): plan + compile for one page.

Vendored into `agent_backend` so the LangGraph pipeline is self-contained.
"""

# (Vendored implementation.)

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Skill framework + shared execution skeleton live in the `skills/` package.
# The plan/compile stage no longer hard-codes a closed set of capability types;
# every content-editing capability is a self-describing `Skill` registered in
# `skills/__init__.py`. The framework (plan prompt, validation, dispatch) reads
# from `SKILLS` and adapts itself. Adding a capability = registering one more
# `Skill` there — no edits needed here.
#
# Visual re-layout is deliberately NOT a skill. It is a single boolean phase
# (`visual_intent.enabled`) handled by the fixed reference-image pipeline
# downstream. A skill can REQUEST a visual re-layout at runtime by returning
# `SkillResult.triggered_visual=True` (e.g. bilingual translation).
from .skills import (
    SKILLS,
    Skill,
    SkillResult,
    _build_skill_docs,
    _hard_ordering_edges,
    _skill_ids,
)
from .skills.base import (
    _call_claude_json,
    _get_segments,
)


DEFAULT_MODEL = "claude-opus-4-8"


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"input JSON must be an object: {path}")
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 2A: Plan
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT_BASE = """You are the "PLAN" stage of a single-page PPT-editing agent.

You will receive:
1) `user_request`: a natural-language instruction from the user about how to modify ONE page.
2) `understand_output`: structured page representation produced by Step 1 (texts[], images[], original_layout_description_en, ...).
{SELECTION_INPUT_DOC}

Your job is LIGHTWEIGHT ROUTING, not execution. Read the user's ENTIRE request
and understand the whole goal FIRST, then decide which skills are needed and
write, for each, a self-contained natural-language `objective`. You do NOT
select text ids, you do NOT write parameters for text skills, and you do NOT
perform any edit yourself — each skill's own execution stage reads your
`objective` plus the full page and does the actual work (including figuring out
which text it applies to).

Emit a strictly typed JSON object with TWO independent parts:

  A) `content_intents`: an ordered list of CONTENT edits (change the actual text /
     content of the page). Each entry invokes ONE "skill".
  B) `visual_intent`: a SINGLE object describing whether the page should be
     visually re-laid-out / restyled, and the user's visual requirements verbatim.

You MUST follow these rules:

- Top-level `schema_version` MUST be the exact string "plan_output_v1". Do not invent another value.

============================ PART A: content_intents ============================

- Each content intent is an object:
  {"id", "skill", "objective", "after", "visual_detail_provided"}.
  (A FEW skills also take a small structured `params` object — see AVAILABLE
  SKILLS; every OTHER skill takes ONLY `objective`.)

- `id` is a unique short string like "i0", "i1", ...

- `skill` MUST be one of the CLOSED set of skill ids described in AVAILABLE SKILLS
  below. Do NOT invent a skill id. If the user's content request cannot be
  expressed with any available skill, do NOT emit an intent for it -- put a short
  reason in `skip_reasons`.

- `objective`: a natural-language description of EXACTLY what this skill must do,
  written so the skill's own executor can act on it WITHOUT seeing the rest of
  your plan. HARD RULES for `objective`:
    * NEVER reference text ids / refs (no "t3", "t5"), and NEVER reference cell
      coordinates. Describe WHICH text by its content, role, or position instead
      — e.g. "the main title at the top", "the body paragraphs", "the footer
      line", "the bullet items on the right", "the rows mentioning 2016".
    * Be SELF-CONTAINED: state every part of the work this skill must do, because
      the executor sees only this one `objective` (plus the user's original
      wording for context) and the full page — not your other intents or your
      reasoning.
    * Fold ALL work for one skill into its SINGLE objective (see the one-intent
      rule below).

- ONE INTENT PER SKILL (max). Do NOT emit two intents with the same `skill`.
  If the user wants a skill applied in several places, describe ALL of them in
  that skill's single `objective`. Example: "把标题双语化,并把表格第五行也双语化"
  → ONE `text.translate` intent whose objective is "Make bilingual (keep the
  source and add an English translation of): (1) the page's main title; and
  (2) the content that will go into row 5 of the table (the ... row)."
  Reason: each executor picks its own target text from the whole page; two
  same-skill intents would independently re-scan and could overlap or miss
  parts.

- `visual_detail_provided` (bool): set `true` ONLY when the user gave a SPECIFIC
  visual / placement requirement that governs HOW THIS intent's result is laid
  out — e.g. where the produced text sits, how it is arranged relative to other
  content, its side / columns / order / relative size. Otherwise set `false`.
  This tells downstream code whether to inject a built-in default layout for this
  intent (only when `false`).
    * `true`  — "把这页双语化,英文放左中文放右" (position governs the translation)
                / "翻译成英文并让译文居中" (placement of this intent's output).
    * `false` — "把这页双语化" / "翻译一下" (pure content, no placement said)
                / "双语化,顺便美化一下" (a BARE beautify is NOT a specific layout
                  requirement for this intent) / a visual requirement that is
                  only about OTHER content, not this intent's output.
  When unsure, prefer `false` (let the default layout apply). Any specific visual
  wording you DO see must still be copied VERBATIM into `visual_intent.
  requirements_text` as today — this boolean is in addition to, not instead of,
  that.

- `after`: list of intent ids that MUST run before this one. Express real
  dependencies only.

  Deciding order between intents:
    * THE USER'S STATED ORDER WINS. If the request implies or states a sequence
      ("先…再…", "translate after you rewrite", "首先脱敏"), encode exactly that
      via `after`.
    * When the user does NOT specify an order, consult each skill's "ordering
      suggestion" in AVAILABLE SKILLS below. These are per-skill hints about what
      usually runs before/after and WHY. Weigh them and set `after` accordingly.
      They are SUGGESTIONS, not fixed law — there is no single global order that
      fits every request, so reason about THIS request rather than applying a
      rote sequence.
    * For genuinely unrelated intents, leave `after` empty. Do NOT add fake
      dependencies.
    * Note: a few ordering constraints are enforced downstream for
      safety/correctness even if you omit them (e.g. redaction before any skill
      that re-emits text). You still SHOULD encode the order you intend; the
      enforcement is only a backstop.
    * PHASE RULE (enforced in code): every TABLE skill (`table.build`,
      `table.reshape`, `table.compute`) runs AFTER every TEXT skill
      (`text.redact`, `text.rewrite`, `text.translate`, `text.add`,
      `text.delete`) — table skills always operate on the FINAL text. So you do
      NOT need `after` edges from a table
      intent to a text intent, and you MUST NEVER make a TEXT intent depend on a
      TABLE intent. If the user says "put X in a table, then translate that
      column", express it as the equivalent order: translate that content first
      (a text.translate intent whose objective names that content), then build
      the table (a table.build intent whose objective says to include both the
      source and its translation as columns).

--- AVAILABLE SKILLS ---
{SKILL_DOCS}

============================ PART B: visual_intent ============================

- `visual_intent` is a single object: {"enabled": <bool>, "requirements_text": "<string>"}.

- Set `visual_intent.enabled = true` when the user asked for ANYTHING about how the
  page LOOKS -- layout, arrangement, style, color/atmosphere, emphasis, or a bare
  "美化 / beautify / make it look better". This is NOT a skill: downstream a fixed
  reference-image pipeline redesigns the page and step3 rebuilds it. A "visual
  request" includes (non-exhaustive):
    * position / side: "英文放左边中文放右边", "把标题放到顶部", "logo 移到右下角"
    * columns / grouping: "分成两栏", "并排显示", "做成三列", "side by side"
    * alignment / order: "居中对齐", "从上到下依次排列", "reorder as ..."
    * spacing / emphasis / resizing: "把要点放大", "拉开间距", "突出这块"
    * style / color / mood: "换成蓝色调", "更商务", "扁平风", "背景改深色"
    * a bare beautify: "美化一下", "排版好看点", "make this slide nicer"

- `requirements_text`: copy the user's visual requirement VERBATIM (the exact words
  describing what they want visually). Do NOT distill, translate, or summarize it --
  the downstream reference-image generator interprets it itself. When
  `enabled = false`, set `requirements_text` to "".

- When `visual_intent.enabled = false`, the page keeps its current layout and only
  the content edits from Part A are applied.

- CRITICAL — do NOT over-trigger the visual intent. Set `enabled = true` ONLY when
  the user EXPLICITLY asked for a visual/layout/style change. A pure content edit
  (translate / rewrite / redact, INCLUDING a plain bilingual "双语化 / 中英双语 /
  add an English translation" with NO position/style words) MUST keep
  `enabled = false`. Producing source+translation fragments does NOT by itself need
  a visual re-layout: the renderer already places each translation adjacent to its
  source, and a content skill will itself request a re-flow at runtime if it changed
  the text volume enough. When unsure whether a phrase is a real visual requirement,
  prefer `enabled = false` and note it in `skip_reasons`
  (e.g. "no_visual: request is content-only, no explicit layout/style requirement").

Examples (content + visual together):
  * "rewrite to be more commercial AND translate to English"
      → content_intents:
          i0 text.rewrite   {objective: "Rewrite all the page text in a more
                              commercial/marketing tone, keeping the meaning.",
                              after: []}
          i1 text.translate {objective: "Translate all the page text into
                              English, REPLACING the source (not bilingual).",
                              after: ["i0"], visual_detail_provided: false}
        visual_intent: {enabled:false, requirements_text:""}
  * "把这一页做成中英双语" / "add an English translation"
      → content_intents:
          i0 text.translate {objective: "Make the whole page bilingual: keep the
                              original Chinese and add an English translation of
                              every text item.", after: [],
                              visual_detail_provided: false}
        visual_intent: {enabled:false, requirements_text:""}
      Note: pure content edit, no placement word -> visual stays disabled and
      visual_detail_provided is false (a built-in default bilingual layout will
      be applied downstream). The "bilingual vs replace" decision lives in the
      objective's wording — say "keep ... and add a translation" for bilingual,
      or "translate, replacing the source" for a pure replacement.
  * "美化一下这一页" / "just make it look nicer"
      → content_intents: [];
        visual_intent: {enabled:true, requirements_text:"美化一下这一页"}
  * "Make the body text bilingual, with English on the left and Chinese on the
     right, and redact the organization names."
      → content_intents:
          i0 text.redact    {objective: "Redact/mask all organization (company /
                              institution) names throughout the page text.",
                              after: [], visual_detail_provided: false}
          i1 text.translate {objective: "Make the BODY text bilingual: keep the
                              Chinese body paragraphs and add an English
                              translation of each.", after: ["i0"],
                              visual_detail_provided: true}
        visual_intent: {enabled:true, requirements_text:"English on the left and Chinese on the right"}
      Note: "English left, Chinese right" is a POSITION requirement for the
      translation, so the visual intent is enabled AND the translate intent's
      visual_detail_provided is true (the user's placement wins; no default is
      injected). Redact carries no placement requirement of its own, so its
      visual_detail_provided is false. Redact is listed before translate so
      masked spans are never carried into the translation.

- `skip_reasons` is a list of short English strings explaining anything you intentionally did not do.

Output STRICT JSON only. No markdown, no commentary.
"""


_PLAN_SELECTION_INPUT_DOC_A = """3) `selected_refs`: a non-empty list of text ids the user explicitly selected in
   the frontend (real ids from `understand_output.texts[].id`; image ids are
   filtered out). This tells you WHICH region the user is pointing at.

TARGETING WITH A SELECTION: you still do NOT put ids in `objective`. Instead,
look up each selected id in `understand_output.texts[]`, read its text/role, and
DESCRIBE that selected content in natural language inside the objective so the
skill's executor can re-find it on the page — e.g. "the selected title 'Q3
Results' and the selected paragraph that starts 'Our revenue…'". Treat the
selection as the default target unless `user_request` clearly overrides it
(e.g. "ignore my selection, translate the whole page" → target the whole page)."""

_PLAN_SELECTION_INPUT_DOC_B = "(No frontend selection is provided. Describe each objective's target purely from `user_request` and the page content.)"


def _build_plan_system_prompt(*, has_selection: bool) -> str:
    selection_doc = _PLAN_SELECTION_INPUT_DOC_A if has_selection else _PLAN_SELECTION_INPUT_DOC_B
    return (
        _PLAN_SYSTEM_PROMPT_BASE.replace("{SELECTION_INPUT_DOC}", selection_doc)
        .replace("{SKILL_DOCS}", _build_skill_docs())
    )


def _build_plan_user_text(
    *,
    user_request: str,
    selected_refs: list[str],
    understand_output: dict[str, Any],
) -> str:
    """Build the user-side payload for the planner.

    Mode A (selected_refs non-empty): include the `selected_refs` key so the
    planner uses it as the default scope per Mode A rules.
    Mode B (selected_refs empty): omit the `selected_refs` key entirely so
    the planner doesn't see a misleading empty-array signal; Mode B rules in
    the system prompt locate ids purely from the natural-language request.
    """
    def _slim_text(t: dict[str, Any]) -> dict[str, Any]:
        segs = _get_segments(t)
        item: dict[str, Any] = {
            "id": str(t.get("id") or ""),
            "kind": str(t.get("kind") or ""),
            "text": str(t.get("text") or ""),
        }
        # Only surface segments when the node has real internal structure
        # (more than one line/item); a single-segment node adds no addressing
        # value and would just bloat the payload.
        if len(segs) > 1:
            item["segments"] = segs
            item["segment_count"] = len(segs)
        return item

    slim_texts = [
        _slim_text(t)
        for t in (understand_output.get("texts") or [])
        if isinstance(t, dict)
    ]
    def _slim_image(im: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": str(im.get("id") or ""),
            "description_en": str(im.get("description_en") or ""),
        }
        # Soft reference size (px in the shared 1000-wide space). Present only
        # when the source carried it; downstream re-layout may freely rescale.
        dw = im.get("display_w_pt")
        dh = im.get("display_h_pt")
        if isinstance(dw, (int, float)) and dw > 0:
            out["display_w_pt"] = float(dw)
        if isinstance(dh, (int, float)) and dh > 0:
            out["display_h_pt"] = float(dh)
        return out

    slim_images = [
        _slim_image(im)
        for im in (understand_output.get("images") or [])
        if isinstance(im, dict)
    ]
    payload: dict[str, Any] = {
        "user_request": user_request,
        "understand_output": {
            "page_num": understand_output.get("page_num"),
            "page_id": understand_output.get("page_id"),
            "page_size_pt": understand_output.get("page_size_pt"),
            "palette": understand_output.get("palette"),
            "texts": slim_texts,
            "images": slim_images,
            "original_layout_description_en": understand_output.get("original_layout_description_en") or "",
        },
    }
    if selected_refs:
        payload["selected_refs"] = list(selected_refs)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_visual_intent(plan_out: dict[str, Any]) -> list[str]:
    """Coerce `visual_intent` to {enabled: bool, requirements_text: str}."""
    notes: list[str] = []
    vi = plan_out.get("visual_intent")
    if not isinstance(vi, dict):
        vi = {}
        notes.append("plan_visual_intent_defaulted")
    enabled = bool(vi.get("enabled", False))
    req = vi.get("requirements_text")
    req = req if isinstance(req, str) else ""
    if enabled and not req.strip():
        # A visual intent with no text still works (the reference-image prompt
        # falls back to user_request), but flag it so we can spot planner gaps.
        notes.append("plan_visual_intent_enabled_without_requirements_text")
    plan_out["visual_intent"] = {"enabled": enabled, "requirements_text": req}
    return notes


def _validate_plan_output(plan_out: dict[str, Any], understand_output: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    intents = plan_out.get("content_intents")
    if not isinstance(intents, list):
        warnings.append("plan_content_intents_not_a_list")
        return warnings

    seen_ids: set[str] = set()
    seen_skills: set[str] = set()
    for idx, it in enumerate(intents):
        if not isinstance(it, dict):
            warnings.append(f"intent[{idx}]_not_object")
            continue
        iid = str(it.get("id") or "")
        if not iid:
            warnings.append(f"intent[{idx}]_missing_id")
        elif iid in seen_ids:
            warnings.append(f"intent[{idx}]_duplicate_id: {iid}")
        else:
            seen_ids.add(iid)

        skill_id = str(it.get("skill") or "")
        if skill_id not in SKILLS:
            warnings.append(f"intent[{idx}]_unknown_skill: {skill_id}")
        elif skill_id in seen_skills:
            # One-intent-per-skill: two same-skill intents each re-scan the page
            # for their target and can overlap/miss. Flag it (repair does not
            # merge automatically — that would need semantic judgment).
            warnings.append(f"intent[{idx}]_duplicate_skill: {skill_id}")
        else:
            seen_skills.add(skill_id)

        after = it.get("after") or []
        if not isinstance(after, list):
            warnings.append(f"intent[{idx}]_after_not_list")

        objective = it.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            warnings.append(f"intent[{idx}]_missing_objective")

    all_ids = {str(it.get("id") or "") for it in intents if isinstance(it, dict)}
    for idx, it in enumerate(intents):
        if not isinstance(it, dict):
            continue
        for dep in (it.get("after") or []):
            if isinstance(dep, str) and dep and dep not in all_ids:
                warnings.append(f"intent[{idx}]_after_dangling_dep: {dep}")

    return warnings


def _repair_plan_params(plan_out: dict[str, Any]) -> list[str]:
    """Normalize each content intent's shape and delegate param repair to its
    skill. Each skill plugs its own missing/invalid params with safe defaults so
    the executor never no-ops on an under-specified intent.
    """
    repairs: list[str] = []
    intents = plan_out.get("content_intents")
    if not isinstance(intents, list):
        return repairs

    for idx, it in enumerate(intents):
        if not isinstance(it, dict):
            continue
        if not isinstance(it.get("after"), list):
            it["after"] = []
        if not isinstance(it.get("objective"), str):
            it["objective"] = ""
            repairs.append(f"intent[{idx}]_repaired_objective_to_empty")
        # Missing/invalid -> False, i.e. prefer injecting the built-in default
        # layout (the safe side: the user didn't pin a placement for this intent).
        if not isinstance(it.get("visual_detail_provided"), bool):
            it["visual_detail_provided"] = False
        params = it.get("params")
        if not isinstance(params, dict):
            params = {}
            it["params"] = params

        skill = SKILLS.get(str(it.get("skill") or ""))
        if skill is None:
            continue
        for note in skill.repair(params):
            repairs.append(f"intent[{idx}]_{note}")

    return repairs


def _enforce_hard_ordering(plan_out: dict[str, Any]) -> list[str]:
    """Enforce ONLY the few safety/correctness ordering edges declared locally by
    skills via `Skill.hard_before` (see `_hard_ordering_edges`).

    Cross-skill sequencing is otherwise the planner's job (guided by each skill's
    `ordering_note`); we no longer impose a global `canonical_rank` order. Here we
    only add the missing `after` edge for a hard (A -> B) pair when BOTH skills are
    present, AND only when it would not contradict an order the planner already set
    (i.e. skip if B is already required before A, to avoid creating a cycle — the
    planner's explicit order wins even over a safety default, and a cycle is
    surfaced as a warning by the topological sort).
    """
    notes: list[str] = []
    intents = plan_out.get("content_intents")
    if not isinstance(intents, list):
        return notes

    hard = _hard_ordering_edges()
    if not hard:
        return notes

    # Index intents by skill id -> list of (iid, after-set, obj).
    by_skill: dict[str, list[tuple[str, set[str], dict[str, Any]]]] = {}
    for it in intents:
        if not isinstance(it, dict):
            continue
        iid = str(it.get("id") or "")
        skill_id = str(it.get("skill") or "")
        if not iid or skill_id not in SKILLS:
            continue
        after_list = it.get("after")
        if not isinstance(after_list, list):
            after_list = []
            it["after"] = after_list
        by_skill.setdefault(skill_id, []).append(
            (iid, {s for s in after_list if isinstance(s, str)}, it)
        )

    for a_skill, b_skill in hard:
        for a_id, _a_after, _a_obj in by_skill.get(a_skill, []):
            for b_id, b_after, b_obj in by_skill.get(b_skill, []):
                if a_id == b_id or a_id in b_after:
                    continue
                # Respect an explicit contradicting order (b before a): don't
                # force the safety edge if it would create a cycle.
                if b_id in _intent_after_set(_a_obj):
                    notes.append(
                        f"hard_ordering_skipped_conflict: {a_skill}({a_id}) before "
                        f"{b_skill}({b_id}) contradicts planner order"
                    )
                    continue
                b_obj_after = b_obj.get("after")
                if isinstance(b_obj_after, list):
                    b_obj_after.append(a_id)
                else:
                    b_obj["after"] = [a_id]
                b_after.add(a_id)
                notes.append(
                    f"hard_ordering_enforced: added {a_id!r} ({a_skill}) before "
                    f"{b_id!r} ({b_skill})"
                )

    return notes


def _intent_after_set(intent: dict[str, Any]) -> set[str]:
    after = intent.get("after")
    if isinstance(after, list):
        return {s for s in after if isinstance(s, str)}
    return set()


def _empty_plan(selected_refs: list[str], skip_reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "plan_output_v1",
        "selected_refs": list(selected_refs),
        "content_intents": [],
        "visual_intent": {"enabled": False, "requirements_text": ""},
        "skip_reasons": list(skip_reasons),
    }


def plan_step(
    *,
    user_request: str,
    selected_refs: list[str],
    understand_output: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
) -> tuple[dict[str, Any], list[str], str]:
    """Step 2A: produce a PlanOutputV1 from a natural-language request."""
    warnings: list[str] = []

    text_id_set = {
        str(t.get("id") or "")
        for t in (understand_output.get("texts") or [])
        if isinstance(t, dict)
    }
    filtered_selected_refs = [r for r in (selected_refs or []) if isinstance(r, str) and r in text_id_set]
    dropped_refs = [r for r in (selected_refs or []) if r not in filtered_selected_refs]
    if dropped_refs:
        warnings.append(f"plan_dropped_unsupported_selected_refs: {dropped_refs}")
    has_selection = bool(filtered_selected_refs)

    if dry_run:
        plan_out = _empty_plan(filtered_selected_refs, ["dry_run_stub_plan"])
        plan_out["visual_intent"] = {"enabled": True, "requirements_text": user_request}
        warnings.append("dry_run_skipped_planner_llm")
        warnings.extend(_validate_plan_output(plan_out, understand_output))
        return plan_out, warnings, ""

    user_text = _build_plan_user_text(
        user_request=user_request,
        selected_refs=filtered_selected_refs,
        understand_output=understand_output,
    )
    raw, obj, err = _call_claude_json(
        model=model,
        system_prompt=_build_plan_system_prompt(has_selection=has_selection),
        user_text=user_text,
        max_output_tokens=4096,
        tag="step2.plan",
    )
    if err:
        warnings.append(err)
    if not isinstance(obj, dict):
        warnings.append("plan_invalid_json")
        plan_out = _empty_plan(filtered_selected_refs, ["plan_invalid_json"])
        plan_out["fatal_errors"] = ["plan_invalid_json"]
        return plan_out, warnings, raw

    obj.setdefault("skip_reasons", [])
    obj.setdefault("content_intents", [])
    obj["selected_refs"] = list(filtered_selected_refs)
    obj["schema_version"] = "plan_output_v1"
    warnings.extend(_normalize_visual_intent(obj))
    warnings.extend(_repair_plan_params(obj))
    warnings.extend(_enforce_hard_ordering(obj))
    warnings.extend(_validate_plan_output(obj, understand_output))
    return obj, warnings, raw


# Coarse execution phases, in the order the compiler forces them to run
# (Scheme A, changelog 2026-07-21_03): all "text" skills finish before any
# "table" skill starts, so table skills always see the FINAL text nodes and
# stable ids. Lower rank = runs earlier. Unknown phases sort last.
# "image" runs last: image add/delete/replace only touch `state["images"]`,
# never text/table nodes, so their relative order to text/table is irrelevant
# for correctness; pinning them last keeps a stable, predictable sequence.
_PHASE_RANK: dict[str, int] = {"text": 0, "table": 1, "image": 2}


def _intent_phase_rank(intent: dict[str, Any]) -> int:
    skill = SKILLS.get(str(intent.get("skill") or ""))
    phase = getattr(skill, "phase", "text") if skill is not None else "text"
    return _PHASE_RANK.get(str(phase), len(_PHASE_RANK))


def _topo_sort_intents(intents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Kahn's algorithm with a PHASE-aware ready-queue.

    Scheme A is enforced here: among intents that are simultaneously ready (all
    their `after` deps satisfied), we always pop a "text"-phase intent before a
    "table"-phase one. Because the planner is told never to make a text intent
    depend on a table intent, this yields "all text edits, then all table edits"
    without needing explicit `after` edges — while still honoring every real
    dependency the planner did declare. Returns (ordered_intents, warnings).
    """
    warnings: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for it in intents:
        if isinstance(it, dict):
            iid = str(it.get("id") or "")
            if iid:
                by_id[iid] = it

    indeg: dict[str, int] = {iid: 0 for iid in by_id}
    children: dict[str, list[str]] = {iid: [] for iid in by_id}
    for iid, it in by_id.items():
        for dep in (it.get("after") or []):
            if isinstance(dep, str) and dep in by_id:
                indeg[iid] += 1
                children[dep].append(iid)

    def _sort_key(iid: str) -> tuple[int, str]:
        return (_intent_phase_rank(by_id[iid]), iid)

    ready = [iid for iid, d in indeg.items() if d == 0]
    out_order: list[str] = []
    while ready:
        ready.sort(key=_sort_key)
        cur = ready.pop(0)
        out_order.append(cur)
        for ch in children[cur]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)

    if len(out_order) < len(by_id):
        leftover = [iid for iid in by_id if iid not in out_order]
        warnings.append(f"compile_cycle_or_unresolved_deps: {leftover}")
        out_order.extend(leftover)

    # Backstop: if a planner-declared `after` forced a table intent ahead of a
    # later text intent (violating Scheme A), surface it. The dependency is still
    # honored above; this only flags the unusual ordering for review.
    last_table_pos = -1
    for pos, iid in enumerate(out_order):
        rank = _intent_phase_rank(by_id[iid])
        if rank >= _PHASE_RANK["table"]:
            last_table_pos = pos
        elif rank == _PHASE_RANK["text"] and last_table_pos >= 0:
            warnings.append(
                f"compile_phase_order_violation: text intent {iid!r} runs after a "
                f"table intent (planner `after` forced it); expected text-before-table"
            )
            break

    return [by_id[i] for i in out_order if i in by_id], warnings


def _clone_understand(understand_output: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(understand_output, ensure_ascii=False))


def _prune_dangling_text_refs(state: dict[str, Any]) -> list[str]:
    """Clear reference fields that point at text ids no longer present.

    A skill that REMOVES a text node (e.g. `text.delete`) can leave OTHER nodes
    with a `translation_of` / `derived_from` / `merged_into` pointing at the gone
    id, or a table cell `ref` pointing at it. Per the Path B decision (changelog
    2026-08-04_01, clarification 2) we do NOT cascade-delete the referring content
    — that text is kept and simply demoted to a plain node by dropping the now-
    dangling link. Table cells whose sole ref vanished are dropped (and empty
    tables removed), same as `_compact_state_text_ids` does after renumbering.

    Runs BEFORE `_compact_state_text_ids` each turn so renumbering only ever sees
    valid links. Returns human-readable notes for the compile warnings log.
    """
    notes: list[str] = []
    texts = state.get("texts")
    if not isinstance(texts, list):
        return notes
    live_ids = {
        str(t.get("id") or "")
        for t in texts
        if isinstance(t, dict) and str(t.get("id") or "")
    }

    for t in texts:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "")
        for field in ("translation_of", "merged_into"):
            val = t.get(field)
            if isinstance(val, str) and val and val not in live_ids:
                t.pop(field, None)
                notes.append(f"pruned_dangling_{field}[{tid}]: {val}")
        df = t.get("derived_from")
        if isinstance(df, list) and df:
            kept = [x for x in df if isinstance(x, str) and x in live_ids]
            if len(kept) != len([x for x in df if isinstance(x, str)]):
                if kept:
                    t["derived_from"] = kept
                else:
                    t.pop("derived_from", None)
                    t.pop("derived_op", None)
                notes.append(f"pruned_dangling_derived_from[{tid}]")

    tables = state.get("tables")
    if isinstance(tables, list):
        surviving_tables: list[dict[str, Any]] = []
        for tbl in tables:
            if not isinstance(tbl, dict):
                continue
            cells = tbl.get("cells")
            if not isinstance(cells, list):
                surviving_tables.append(tbl)
                continue
            new_cells = [
                c for c in cells
                if isinstance(c, dict) and str(c.get("ref") or "") in live_ids
            ]
            if len(new_cells) != len(cells):
                notes.append(f"pruned_dangling_table_cells: {len(cells) - len(new_cells)}")
            if new_cells:
                tbl["cells"] = new_cells
                surviving_tables.append(tbl)
            else:
                notes.append("pruned_empty_table_after_delete")
        state["tables"] = surviving_tables

    return notes


def _compact_state_text_ids(state: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    """Drop placeholder text fragments and renumber remaining t-ids to t0..tN.

    A placeholder is a text fragment with `text == ""` AND `merged_into` set
    (produced e.g. by the translate subagent when it merges fragments).

    Returns (changed, id_map) where:
      - changed: True if anything was dropped or renumbered.
      - id_map:  old_id -> new_id (only for ids that survived). Dropped ids are
                 absent from this map.

    Side-effects on state:
      - state["texts"] is rewritten to the surviving fragments with new ids.
      - For surviving fragments, `merged_into` (if it pointed to a dropped id)
        is removed; otherwise it is rewritten via id_map.
    """
    texts = state.get("texts")
    if not isinstance(texts, list):
        return False, {}

    survivors: list[dict[str, Any]] = []
    dropped_ids: set[str] = set()

    for t in texts:
        if not isinstance(t, dict):
            continue
        old_id = str(t.get("id") or "")
        text_val = t.get("text")
        is_placeholder = (
            isinstance(text_val, str) and text_val == "" and bool(t.get("merged_into"))
        )
        if is_placeholder:
            if old_id:
                dropped_ids.add(old_id)
            continue
        survivors.append(t)

    id_map: dict[str, str] = {}
    for new_idx, t in enumerate(survivors):
        old_id = str(t.get("id") or "")
        new_id = f"t{new_idx}"
        if old_id and old_id != new_id:
            id_map[old_id] = new_id
        elif old_id:
            id_map[old_id] = old_id
        t["id"] = new_id

    changed = bool(dropped_ids) or any(k != v for k, v in id_map.items())
    if not changed:
        return False, {old: old for old in id_map}

    for t in survivors:
        mi = t.get("merged_into")
        if isinstance(mi, str) and mi:
            if mi in dropped_ids:
                t.pop("merged_into", None)
            elif mi in id_map:
                t["merged_into"] = id_map[mi]
        tof = t.get("translation_of")
        if isinstance(tof, str) and tof:
            if tof in dropped_ids:
                t.pop("translation_of", None)
            elif tof in id_map:
                t["translation_of"] = id_map[tof]
        df = t.get("derived_from")
        if isinstance(df, list) and df:
            remapped = [
                id_map[x] for x in df
                if isinstance(x, str) and x not in dropped_ids and x in id_map
            ]
            if remapped:
                t["derived_from"] = remapped
            else:
                t.pop("derived_from", None)

    state["texts"] = survivors

    # Table cell refs point at text ids; remap/prune them so tables stay valid
    # after renumbering. Drop cells whose ref was removed, and drop tables that
    # end up empty.
    tables = state.get("tables")
    if isinstance(tables, list):
        surviving_tables: list[dict[str, Any]] = []
        for tbl in tables:
            if not isinstance(tbl, dict):
                continue
            cells = tbl.get("cells")
            if not isinstance(cells, list):
                continue
            new_cells: list[dict[str, Any]] = []
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                ref = str(cell.get("ref") or "")
                if ref in dropped_ids:
                    continue
                if ref in id_map:
                    cell = {**cell, "ref": id_map[ref]}
                new_cells.append(cell)
            if new_cells:
                tbl["cells"] = new_cells
                surviving_tables.append(tbl)
        state["tables"] = surviving_tables

    return True, id_map


def compile_step(
    *,
    plan_output: dict[str, Any],
    understand_output: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
    user_request: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Step 2B: execute content_intents serially against a cloned state.

    Also folds the plan's `visual_intent` together with any skill that
    requested a re-flow at runtime into a single resolved `visual_intent` in
    the output, so downstream steps read ONE signal.
    """
    warnings: list[str] = []
    intents = plan_output.get("content_intents") or []
    if not isinstance(intents, list):
        warnings.append("compile_content_intents_not_a_list")
        intents = []

    plan_visual = plan_output.get("visual_intent")
    if not isinstance(plan_visual, dict):
        plan_visual = {"enabled": False, "requirements_text": ""}
    visual_enabled = bool(plan_visual.get("enabled", False))
    visual_requirements = str(plan_visual.get("requirements_text") or "")

    state = _clone_understand(understand_output)

    ordered, w = _topo_sort_intents([it for it in intents if isinstance(it, dict)])
    warnings.extend(w)

    executed_ids: list[str] = []
    skill_triggered_visual = False
    collected_visual_defaults: list[dict[str, str]] = []
    for intent in ordered:
        skill_id = str(intent.get("skill") or "")
        skill = SKILLS.get(skill_id)
        if skill is None:
            warnings.append(f"compile_no_skill_for_id: {skill_id}")
            continue
        try:
            _t0 = time.monotonic()
            result = skill.execute(
                intent=intent,
                state=state,
                api_key=api_key,
                model=model,
                dry_run=dry_run,
                user_request=user_request,
            )
            _elapsed = time.monotonic() - _t0
            # Temporary instrumentation (mirrors [llm_timing]): show how long each
            # skill's execute took, tagged by phase, so we can see whether table
            # skills (pure Python, no LLM) or text skills (one LLM call each) are
            # what makes step2.compile slow. Grep `[skill_timing]` out of the log.
            print(
                f"[skill_timing] skill={skill_id} phase={getattr(skill, 'phase', '?')} "
                f"intent={intent.get('id')} elapsed={_elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            if not isinstance(result, SkillResult):
                warnings.append(f"compile_skill_bad_result[{intent.get('id')}]: {skill_id}")
                result = SkillResult()
            if result.warnings:
                warnings.extend(result.warnings)
            if result.triggered_visual:
                skill_triggered_visual = True
                warnings.append(f"compile_skill_requested_visual[{intent.get('id')}]: {skill_id}")
            # Adopt the skill's default visual layout ONLY when the user gave no
            # visual requirement for THIS intent. If they did
            # (visual_detail_provided=true), their requirement is already in
            # visual_intent.requirements_text and wins — inject nothing.
            vd = result.default_visual_detail
            if vd is not None and not bool(intent.get("visual_detail_provided", False)):
                collected_visual_defaults.append(
                    {"key": vd.key, "image_text": vd.image_text, "step3_text": vd.step3_text}
                )
                warnings.append(
                    f"compile_injected_visual_default[{intent.get('id')}]: {vd.key}"
                )
            executed_ids.append(str(intent.get("id") or ""))
            prune_notes = _prune_dangling_text_refs(state)
            if prune_notes:
                warnings.extend(f"compile_prune_after[{intent.get('id')}]: {n}" for n in prune_notes)
            changed, id_map = _compact_state_text_ids(state)
            if changed:
                renames = {k: v for k, v in id_map.items() if k != v}
                if renames:
                    warnings.append(f"compile_compacted_after[{intent.get('id')}]: renamed={renames}")
                else:
                    warnings.append(f"compile_compacted_after[{intent.get('id')}]: dropped placeholders")
        except Exception as e:
            warnings.append(f"compile_skill_exception[{intent.get('id')}]: {type(e).__name__}: {e}")

    resolved_visual = {
        "enabled": bool(visual_enabled or skill_triggered_visual),
        "requirements_text": visual_requirements,
        # Built-in default layout details injected by skills when the user gave
        # no visual requirement for that intent (each has image_text/step3_text).
        "default_details": collected_visual_defaults,
        "requested_by_plan": visual_enabled,
        "requested_by_skill": skill_triggered_visual,
    }

    out: dict[str, Any] = {
        "schema_version": "compile_output_v1",
        "executed_intent_ids": executed_ids,
        "visual_intent": resolved_visual,
        "understand_modified": state,
    }
    return out, warnings


def step2_run(
    *,
    request_obj: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    """End-to-end Step 2 (Plan + Compile) for ONE page."""
    _t_step2_start = time.monotonic()
    user_request = str(request_obj.get("user_request") or "").strip()
    if not user_request:
        raise ValueError("request.user_request is required")

    selected_refs_raw = request_obj.get("selected_refs") or []
    selected_refs = [s for s in selected_refs_raw if isinstance(s, str)]

    understand_output_inline = request_obj.get("understand_output")
    understand_output_path = str(request_obj.get("understand_output_path") or "").strip()
    if isinstance(understand_output_inline, dict):
        understand_output = understand_output_inline
    elif understand_output_path:
        understand_output = _read_json(Path(understand_output_path).expanduser().resolve())
    else:
        raise ValueError("request must provide either understand_output or understand_output_path")

    plan_out, plan_warnings, planner_raw = plan_step(
        user_request=user_request,
        selected_refs=selected_refs,
        understand_output=understand_output,
        api_key=api_key,
        model=model,
        dry_run=dry_run,
    )
    _t_plan = time.monotonic()
    plan_fatal_errors = [
        str(e)
        for e in (plan_out.get("fatal_errors") or [])
        if str(e or "").strip()
    ]
    if plan_fatal_errors:
        return {
            "schema_version": "step2_output_v1",
            "user_request": user_request,
            "selected_refs": selected_refs,
            "visual_intent": {"enabled": False, "requirements_text": "", "default_details": []},
            "plan": plan_out,
            "compile": {
                "understand_modified": understand_output,
                "visual_intent": {"enabled": False, "requirements_text": "", "default_details": []},
                "skipped": True,
                "reason": "plan_fatal_error",
            },
            "warnings": {
                "plan": plan_warnings,
                "compile": [],
            },
            "fatal_errors": plan_fatal_errors,
            "planner_raw_response": planner_raw,
        }
    compile_out, compile_warnings = compile_step(
        plan_output=plan_out,
        understand_output=understand_output,
        api_key=api_key,
        model=model,
        dry_run=dry_run,
        user_request=user_request,
    )
    _t_compile_end = time.monotonic()
    # Temporary instrumentation: coarse plan-vs-compile split for step2. `plan`
    # is one planner LLM call; `compile` is skill execution (text skills each do
    # one LLM call, table skills are pure Python). Grep `[step2_timing]`.
    print(
        f"[step2_timing] plan={_t_plan - _t_step2_start:.2f}s "
        f"compile={_t_compile_end - _t_plan:.2f}s "
        f"total={_t_compile_end - _t_step2_start:.2f}s "
        f"intents={len(plan_out.get('content_intents') or [])}",
        file=sys.stderr,
        flush=True,
    )

    # Resolved visual intent = plan's request OR any skill's runtime request.
    resolved_visual = compile_out.get("visual_intent")
    if not isinstance(resolved_visual, dict):
        resolved_visual = {"enabled": False, "requirements_text": "", "default_details": []}

    return {
        "schema_version": "step2_output_v1",
        "user_request": user_request,
        "selected_refs": selected_refs,
        "visual_intent": resolved_visual,
        "plan": plan_out,
        "compile": compile_out,
        "warnings": {
            "plan": plan_warnings,
            "compile": compile_warnings,
        },
        "planner_raw_response": planner_raw,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Step2 Plan+Compile MVP (single page).")
    p.add_argument(
        "--request",
        type=str,
        required=True,
        help="Path to a Step2 request JSON (user_request, selected_refs, understand_output_path).",
    )
    p.add_argument("--out", type=str, required=True, help="Path to write Step2 output JSON.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Unused; the model is called via the OpenAI-compatible proxy set by PPT_LLM_BASE_URL + PPT_LLM_API_KEY.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model calls; emit deterministic stub plan and stub subagent effects.",
    )
    args = p.parse_args()

    in_path = Path(args.request).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"request not found: {in_path}")

    request_obj = _read_json(in_path)
    out_obj = step2_run(
        request_obj=request_obj,
        api_key=(args.api_key or None),
        model=str(args.model),
        dry_run=bool(args.dry_run),
    )
    _write_json(out_path, out_obj)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
