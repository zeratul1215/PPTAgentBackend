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
from copy import deepcopy
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
from ....table_spec import slim_table_for_model


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
2) `understand_output`: structured page representation produced by the page-understanding service (texts[], images[], native tables[], original_layout_description_en, ...).
3) `previous_html_available`: whether a successful final HTML rendering still matches the current PPTist page.
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

  C) `reassembly_strategy`: decide whether a valid previous final HTML can be
     used as a layout/style reference for this complete HTML generation.
     Use {"mode":"reuse_previous_html|rebuild", "change_scope":"local|section|global",
     "reason":"short internal reason"}. Choose reuse only when the main page
     structure, grouping, reading order, and regions remain suitable. Choose
     rebuild when the page needs a new composition, new major regions, changed
     columns, changed component types, or a different reading order. This is a
     generation strategy, never an HTML diff or DOM patch.

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
      coordinates. Describe the target by its visible content, semantic role, or
      position.
    * Be SELF-CONTAINED: state every part of the work this skill must do, because
      the executor sees only this one `objective` (plus the user's original
      wording for context) and the full page — not your other intents or your
      reasoning.
    * Fold ALL work for one skill into its SINGLE objective (see the one-intent
      rule below).

- ONE INTENT PER SKILL (max). Do NOT emit two intents with the same `skill`.
  If one skill must affect several targets, describe all of them in that skill's
  single objective. Each executor scans the whole page once.

- `visual_detail_provided` (bool): set `true` ONLY when the user gave a specific
  visual or placement requirement governing how this intent's result is laid
  out. Otherwise set `false`.
  This tells downstream code whether to inject a built-in default layout for this
  intent (only when `false`).
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
    * When the user does NOT specify an order, infer the data dependencies required
      by the requested final state and the chosen skills. Use each skill's usual
      ordering note as soft guidance, not as a mandatory workflow.
    * For genuinely unrelated intents, leave `after` empty. Do NOT add fake
      dependencies.
    * Note: a few safety/correctness ordering constraints are enforced downstream
      even if omitted. You still SHOULD encode the order required by this request;
      code enforcement is only a backstop.
--- AVAILABLE SKILLS ---
{SKILL_DOCS}

============================ PART B: visual_intent ============================

- `visual_intent` is a single object: {"enabled": <bool>, "requirements_text": "<string>"}.

- Set `visual_intent.enabled = true` when the requested final state includes a
  visual change to layout, composition, placement, alignment, spacing, emphasis,
  size, color, or overall styling. This is not a skill; it is the visual target
  for downstream page reconstruction.

- `requirements_text`: copy the user's visual requirement VERBATIM (the exact words
  describing what they want visually). Do NOT distill, translate, or summarize it --
  the downstream reference-image generator interprets it itself. When
  `enabled = false`, set `requirements_text` to "".

- When `visual_intent.enabled = false`, the page keeps its current layout and only
  the content edits from Part A are applied.

- Do not infer a visual request merely from the name of a content operation.
  Decide from the requested final appearance. Runtime content changes may still
  trigger necessary reflow without changing this field. When uncertain, prefer
  `enabled = false` and record the uncertainty in `skip_reasons`.

- `skip_reasons` is a list of short English strings explaining anything you intentionally did not do.

Output STRICT JSON only. No markdown, no commentary.
"""


_PLAN_SELECTION_INPUT_DOC_A = """3) `selected_refs`: a non-empty list of text ids the user explicitly selected in
   the frontend (real ids from `understand_output.texts[].id`; image ids are
   filtered out). This tells you WHICH region the user is pointing at.

TARGETING WITH A SELECTION: you still do NOT put ids in `objective`. Instead,
look up each selected id in `understand_output.texts[]`, read its text/role, and
DESCRIBE that selected content in natural language inside the objective so the
skill's executor can re-find it on the page. Treat the
selection as the default target unless `user_request` clearly overrides it
with a different scope."""

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
    previous_html_available: bool = False,
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

    def _slim_table(table: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"id": str(table.get("id") or ""), **slim_table_for_model(table)}
        except ValueError:
            return {"id": str(table.get("id") or ""), "data": []}

    slim_tables = [
        _slim_table(table)
        for table in (understand_output.get("tables") or [])
        if isinstance(table, dict)
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
            "tables": slim_tables,
            "original_layout_description_en": understand_output.get("original_layout_description_en") or "",
        },
        "previous_html_available": bool(previous_html_available),
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


def _normalize_reassembly_strategy(
    plan_out: dict[str, Any], *, previous_html_available: bool
) -> list[str]:
    """Normalize the planner's generation strategy and apply the hard gate."""
    notes: list[str] = []
    allowed_modes = {"reuse_previous_html", "rebuild"}
    allowed_scopes = {"local", "section", "global"}
    raw = plan_out.get("reassembly_strategy")
    strategy = raw if isinstance(raw, dict) else {}
    mode = str(strategy.get("mode") or "").strip()
    scope = str(strategy.get("change_scope") or "").strip()
    reason = str(strategy.get("reason") or "").strip()
    if mode not in allowed_modes:
        mode = "rebuild"
        notes.append("reassembly_strategy_defaulted")
    if scope not in allowed_scopes:
        scope = "global" if mode == "rebuild" else "section"
        notes.append("reassembly_change_scope_defaulted")
    if not previous_html_available and mode == "reuse_previous_html":
        mode = "rebuild"
        scope = "global"
        reason = "no_previous_html_available"
        notes.append("reassembly_reuse_blocked_no_previous_html")
    if not reason:
        reason = "planner_selected_rebuild" if mode == "rebuild" else "planner_selected_previous_html_reference"
    plan_out["reassembly_strategy"] = {
        "mode": mode,
        "change_scope": scope,
        "reason": reason[:300],
    }
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

    Cross-skill sequencing is otherwise inferred by the planner from the current
    request and the declared capabilities. We do not impose a global
    `canonical_rank` order. Here we
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
        "reassembly_strategy": {
            "mode": "rebuild",
            "change_scope": "global",
            "reason": "no_previous_html_available",
        },
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
    previous_html_available: bool = False,
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
        previous_html_available=previous_html_available,
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
    warnings.extend(_normalize_reassembly_strategy(obj, previous_html_available=previous_html_available))
    warnings.extend(_normalize_visual_intent(obj))
    warnings.extend(_repair_plan_params(obj))
    warnings.extend(_enforce_hard_ordering(obj))
    contract_errors = _validate_plan_output(obj, understand_output)
    warnings.extend(contract_errors)
    if contract_errors:
        obj["fatal_errors"] = [f"plan_contract_error: {error}" for error in contract_errors]
    return obj, warnings, raw


def _intent_phase_rank(intent: dict[str, Any]) -> int:
    skill = SKILLS.get(str(intent.get("skill") or ""))
    return int(getattr(skill, "canonical_rank", 9999)) if skill is not None else 9999


def _topo_sort_intents(intents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Topologically order intents using only declared real dependencies.

    Skill rank is only a deterministic tie-breaker for independent intents; it
    never creates a dependency or prescribes a table workflow.
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

    return [by_id[i] for i in out_order if i in by_id], warnings


def _clone_understand(understand_output: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(understand_output, ensure_ascii=False))


def _prune_dangling_text_refs(state: dict[str, Any]) -> list[str]:
    """Clear reference fields that point at text ids no longer present.

    A skill that REMOVES a text node (e.g. `text.delete`) can leave OTHER nodes
    with a `translation_of` / `derived_from` / `merged_into` pointing at the gone
    id. Native table cells do not point at text nodes and are never pruned here.
    Per the Path B decision (changelog
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

    # Native tables own their cell text and never reference state["texts"].
    # Deliberately do not prune or rewrite them here.

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

    # Native table cells are independent of compacted text ids.

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
    table_render_context: dict[str, Any] = {"built_table_ids": [], "rebuilt_tables": []}
    ordered, w = _topo_sort_intents([it for it in intents if isinstance(it, dict)])
    warnings.extend(w)

    executed_ids: list[str] = []
    fatal_errors: list[str] = []
    skill_triggered_visual = False
    collected_visual_defaults: list[dict[str, str]] = []
    for intent in ordered:
        skill_id = str(intent.get("skill") or "")
        skill = SKILLS.get(skill_id)
        if skill is None:
            warnings.append(f"compile_no_skill_for_id: {skill_id}")
            fatal_errors.append(
                f"intent {intent.get('id') or '?'} references unknown skill {skill_id or '?'}"
            )
            continue
        try:
            tables_before = {
                str(table.get("id") or ""): deepcopy(table)
                for table in state.get("tables") or []
                if isinstance(table, dict) and str(table.get("id") or "")
            }
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
                result = SkillResult(status="failed")
            # Older skills used warnings as their only error channel.  Treat
            # contract/target failures as fatal even if such a skill forgot to
            # set the new explicit status field.
            if result.status == "applied" and any(
                token in str(w).lower()
                for w in result.warnings
                for token in ("error", "invalid", "no_target", "not_found", "selected_nothing", "forbidden", "bad_")
            ):
                result.status = "failed"
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
            if result.status not in {"applied", "already_satisfied"}:
                warnings.append(f"compile_skill_failed[{intent.get('id')}]: {skill_id}")
                warnings.extend(result.warnings or [f"skill outcome={result.status}"])
                fatal_errors.append(f"intent {intent.get('id') or '?'} ({skill_id}) failed")
                continue
            tables_after = {
                str(table.get("id") or ""): table
                for table in state.get("tables") or []
                if isinstance(table, dict) and str(table.get("id") or "")
            }
            if skill_id == "table.rebuild":
                # Rebuild owns matrix content/structure only. Restore the
                # authoritative style snapshot before checking invariants so a
                # missing-vs-empty style representation cannot become a false
                # fatal error.
                params = intent.get("params") if isinstance(intent.get("params"), dict) else {}
                target_id = str(params.get("table_id") or "")
                if not target_id and len(tables_before) == 1:
                    target_id = next(iter(tables_before))
                before_table, after_table = tables_before.get(target_id), tables_after.get(target_id)
                if before_table is not None and after_table is not None:
                    for field in ("outline", "theme", "cellMinHeight"):
                        if field in before_table:
                            after_table[field] = deepcopy(before_table[field])
                    before_cells = {
                        str(cell.get("id")): cell.get("style") or {}
                        for row in before_table.get("data") or []
                        for cell in row if isinstance(row, list) and isinstance(cell, dict)
                    }
                    for row in after_table.get("data") or []:
                        for cell in row if isinstance(row, list) else []:
                            if isinstance(cell, dict) and str(cell.get("id")) in before_cells:
                                cell["style"] = deepcopy(before_cells[str(cell.get("id"))])
                    before_style = {
                        "outline": before_table.get("outline"), "theme": before_table.get("theme"),
                        "cellMinHeight": before_table.get("cellMinHeight"),
                    }
                    after_style = {
                        "outline": after_table.get("outline"), "theme": after_table.get("theme"),
                        "cellMinHeight": after_table.get("cellMinHeight"),
                    }
                    after_cells = {
                        str(cell.get("id")): cell.get("style") or {}
                        for row in after_table.get("data") or []
                        for cell in row if isinstance(row, list) and isinstance(cell, dict)
                    }
                    if before_style != after_style:
                        fatal_errors.append(f"intent {intent.get('id') or '?'} (table.rebuild) changed table style")
                        warnings.append("compile_table_rebuild_style_changed")
                        continue
                    if any(after_cells.get(cell_id) != style for cell_id, style in before_cells.items() if cell_id in after_cells):
                        fatal_errors.append(f"intent {intent.get('id') or '?'} (table.rebuild) changed existing cell style")
                        warnings.append("compile_table_rebuild_cell_style_changed")
                        continue
            if skill_id == "table.build":
                table_render_context["built_table_ids"].extend(
                    table_id for table_id in tables_after
                    if table_id not in tables_before and table_id not in table_render_context["built_table_ids"]
                )
            elif skill_id == "table.rebuild":
                params = intent.get("params") if isinstance(intent.get("params"), dict) else {}
                target_id = str(params.get("table_id") or "")
                if not target_id and len(tables_before) == 1:
                    target_id = next(iter(tables_before))
                original_table = tables_before.get(target_id)
                if original_table is not None and target_id in tables_after:
                    table_render_context["rebuilt_tables"].append({
                        "table_id": target_id,
                        "original_style": {
                            "outline": deepcopy(original_table.get("outline") or {}),
                            "theme": deepcopy(original_table.get("theme") or {}),
                            "cellMinHeight": original_table.get("cellMinHeight"),
                            "cell_styles": [
                                {"id": str(cell.get("id") or ""), "style": deepcopy(cell.get("style") or {})}
                                for row in original_table.get("data") or []
                                for cell in row if isinstance(row, list) and isinstance(cell, dict)
                            ],
                        },
                    })
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
            fatal_errors.append(
                f"intent {intent.get('id') or '?'} ({skill_id or '?'}) raised an exception"
            )

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
        "schema_version": "compile_output_v2",
        "executed_intent_ids": executed_ids,
        "fatal_errors": fatal_errors,
        "visual_intent": resolved_visual,
        "understand_modified": state,
        "table_render_context": table_render_context,
    }
    return out, warnings


def step2_run(
    *,
    request_obj: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
    previous_html_available: bool = False,
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
        previous_html_available=previous_html_available,
    )
    _t_plan = time.monotonic()
    plan_fatal_errors = [
        str(e)
        for e in (plan_out.get("fatal_errors") or [])
        if str(e or "").strip()
    ]
    if plan_fatal_errors:
        return {
            "schema_version": "step2_output_v2",
            "user_request": user_request,
            "selected_refs": selected_refs,
            "visual_intent": {"enabled": False, "requirements_text": "", "default_details": []},
            "plan": plan_out,
            "reassembly_strategy": {
                "mode": "rebuild",
                "change_scope": "global",
                "reason": "step2_plan_failed",
            },
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
        "schema_version": "step2_output_v2",
        "user_request": user_request,
        "selected_refs": selected_refs,
        "visual_intent": resolved_visual,
        "reassembly_strategy": dict(plan_out.get("reassembly_strategy") or {
            "mode": "rebuild",
            "change_scope": "global",
            "reason": "strategy_missing_after_plan",
        }),
        "plan": plan_out,
        "compile": compile_out,
        "warnings": {
            "plan": plan_warnings,
            "compile": compile_warnings,
        },
        "fatal_errors": [str(e) for e in (compile_out.get("fatal_errors") or []) if str(e).strip()],
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
