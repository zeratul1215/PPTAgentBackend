"""LangGraph definition for the single-page processing pipeline.

The graph is one-shot (no checkpointer): each invocation handles a
single `(project_id, page_num, demand)` task. State lives on disk in
the per-project workspace; the graph itself only carries transient
artifacts and references.
"""

from __future__ import annotations

import operator
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_backend.workspace.paths import (
    WorkspacePaths,
    next_turn_dir,
    read_json,
    workspace_for,
    write_json,
    write_text,
)
from agent_backend.workspace.html_lineage import current_candidate, write_lineage
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.reread import _materialize_images
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.steps import (
    run_step2,
    run_step4_qa,
)
from agent_backend.agent.tools.pipeline_without_reference_image.no_reference_step3 import (
    run_step3_without_reference,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _merge_dicts(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer used for state fields written from parallel branches."""
    out: dict[str, Any] = {}
    if isinstance(a, dict):
        out.update(a)
    if isinstance(b, dict):
        out.update(b)
    return out


class PipelineState(TypedDict, total=False):
    # Inputs
    project_id: str
    page_num: int
    demand: str
    model: str
    dry_run: bool
    title: str

    # Resolved at runtime
    workspace_root: str
    turn_dir: str
    started_at: float
    finished_at: float

    # Progress reporting: the display position (1-based, what the user sees) and
    # the batch index, threaded through so per-step SSE events can be attributed
    # to the right page/task without the frontend having to map slot->position.
    display_page: int
    batch_index: int
    agent_run_id: str

    # State files
    current_page_state: dict[str, Any]
    baseline_ran: bool
    understanding_status: str
    deck_style: dict[str, Any]
    deck_style_revision: int
    html_lineage_candidate: dict[str, Any]

    # Per-step artefacts
    step2_output: dict[str, Any]
    step3_result: dict[str, Any]
    step4_result: dict[str, Any]

    # Parallel-branch results
    commit_result: dict[str, Any]
    reread_result: dict[str, Any]
    branch_artefacts: Annotated[dict[str, Any], _merge_dicts]

    errors: Annotated[list[str], operator.add]


def _paths_from_state(state: PipelineState) -> WorkspacePaths:
    return workspace_for(state["project_id"])  # results_root = default


# ---------------------------------------------------------------------------
# Per-step progress. Each node announces the phase it's ENTERING so the
# frontend can show "正在理解页面 / 正在设计版式 …" instead of a blank spinner.
# Best-effort and fire-and-forget: a publishing failure must never break a turn
# (see context.emit). `stage` is a stable machine key; `label` is the Chinese
# copy the UI shows verbatim so the wording lives in one place (the backend).
# ---------------------------------------------------------------------------

_STAGE_LABELS: dict[str, str] = {
    "understand": "正在理解页面",
    "plan": "正在规划改动",
    "reassemble": "正在生成页面",
    "visual_check": "正在核对页面效果",
    "qa": "正在检查排版",
    "commit": "正在应用结果",
}


def _emit_progress(state: PipelineState, stage: str) -> None:
    try:
        from agent_backend.agent.tools.context import emit
    except Exception:
        return
    try:
        page = int(state.get("display_page") or state.get("page_num") or 0)
    except Exception:
        page = 0
    try:
        index = int(state.get("batch_index") or 0)
    except Exception:
        index = 0
    emit(
        str(state.get("project_id") or ""),
        {
            "type": "task_progress",
            "page": page,
            "slot": int(state.get("page_num") or 0),
            "index": index,
            "agent_run_id": str(state.get("agent_run_id") or ""),
            "stage": stage,
            "label": _STAGE_LABELS.get(stage, stage),
        },
    )


def _turn_index(turn_dir: Path) -> int:
    return int(re.sub(r"\D", "", turn_dir.name) or 0)


def _step2_output_is_valid(step2_output: Any) -> bool:
    """A minimal structural check: step3 consumes texts/images out of
    compile.understand_modified, so that block must be present."""
    if not isinstance(step2_output, dict):
        return False
    compile_obj = step2_output.get("compile")
    if not isinstance(compile_obj, dict):
        return False
    um = compile_obj.get("understand_modified")
    if not isinstance(um, dict):
        return False
    # texts is the primary payload; images may legitimately be empty.
    return isinstance(um.get("texts"), list)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def node_load_state(state: PipelineState) -> PipelineState:
    paths = _paths_from_state(state)
    started = time.time()
    turn_dir = next_turn_dir(paths, state["page_num"])

    patch: PipelineState = {
        "started_at": started,
        "turn_dir": str(turn_dir),
        "workspace_root": str(paths.root),
        "baseline_ran": False,
        "errors": [],
        "branch_artefacts": {},
    }

    candidate = current_candidate(paths, int(state["page_num"]))
    patch["html_lineage_candidate"] = candidate

    # Reference-free runs do not reuse reference-image turns. This keeps a
    # failed old-route run from silently crossing the route boundary.
    return patch


def _baseline_understand_input(paths: WorkspacePaths, page_num: int, turn_dir: Path) -> dict[str, Any]:
    """The step1 input for a page's first (baseline) understanding.

    Prefer the page's PPTist slide JSON when present: for pptx/ppt decks that
    slide is PPTist's native parse, carrying whole paragraphs. The legacy
    ``step1_input.json`` is built from the LibreOffice-PDF text extraction, which
    shatters CJK lines into per-glyph runs — never feed that to step1 when a
    clean slide exists. Only the shattered TEXT is replaced; the page image stays
    the bootstrap-rendered PNG (LibreOffice's sole remaining job here).
    """
    slide_path = paths.pptist_slide_json(page_num)
    if not slide_path.exists():
        raise RuntimeError(f"project is not PPTist-initialized; slide JSON missing for page {page_num}")

    slide = read_json(slide_path)
    if not (isinstance(slide, dict) and isinstance(slide.get("elements"), list)):
        raise RuntimeError(f"invalid PPTist slide JSON for page {page_num}")

    from agent_backend.agent.tools.html_to_pptist import slide_to_plan_page

    plan_page = slide_to_plan_page(slide, page_id=f"page{int(page_num) - 1}")

    # Slide image srcs are self-contained data URIs; write them to the page's
    # STABLE asset dir (not a per-turn dir) so (a) step1 can read the bytes for
    # its per-image descriptions and (b) `bundle_dir` stays alive across turns
    # for the step2 image skills and for chat-staged uploads dropped here by the
    # prestage tool. `_materialize_images` uses content-hash names, so repeated
    # rereads do not create duplicate stable image files.
    from agent_backend.workspace.assets import materialize_pptist_slide_assets, page_asset_source_dir

    materialize_pptist_slide_assets(paths, page_num, slide)
    asset_dir = page_asset_source_dir(paths, page_num)
    _materialize_images(plan_page, asset_dir)

    page_size_pt: dict[str, Any] = {"w": None, "h": None}
    manifest_path = paths.project_manifest_json()
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except Exception:  # noqa: BLE001
            manifest = None
        if isinstance(manifest, dict) and isinstance(manifest.get("page_size_pt"), dict):
            page_size_pt = manifest["page_size_pt"]

    page_png = paths.reread_page_png(page_num) if paths.reread_page_png(page_num).exists() else paths.page_png(page_num)

    return {
        "schema_version": "understand_input_v1",
        "page_num": int(page_num),
        "page_size_pt": page_size_pt,
        "bundle_dir": str(asset_dir),
        "plan_page": plan_page,
        "page_png_path": str(page_png),
        "options": {
            "need_image_descriptions": True,
            "need_original_layout_description": True,
        },
    }


def node_step2(state: PipelineState) -> PipelineState:
    paths = _paths_from_state(state)
    demand = str(state.get("demand") or "").strip()
    if not demand:
        # Empty demand is a contract violation: filtering no-op requests is
        # the outer agent's responsibility. Fail fast rather than fabricating
        # a "keep everything" demand, which would silently re-render the page.
        raise ValueError(
            "node_step2: demand is empty; the caller must filter no-op "
            "requests before invoking the pipeline"
        )

    _emit_progress(state, "plan")
    turn_dir = Path(state["turn_dir"])
    if not isinstance(state.get("current_page_state"), dict):
        raise RuntimeError("node_step2: missing shared page understanding core")
    write_json(turn_dir / "baseline_understand_output.json", state["current_page_state"])
    write_json(
        turn_dir / "step2_request.json",
        {
            "user_request": demand,
            "selected_refs": [],
            "understand_output_path": str(paths.page_understanding_json(state["page_num"])),
        },
    )
    write_json(turn_dir / "html_lineage_candidate.json", state.get("html_lineage_candidate") or {})

    out = run_step2(
        user_request=demand,
        understand_output=state["current_page_state"],
        model=str(state["model"]),
        dry_run=bool(state.get("dry_run", False)),
        previous_html_available=bool((state.get("html_lineage_candidate") or {}).get("available")),
    )
    write_json(turn_dir / "step2_output.json", out)
    fatal_errors = [str(e) for e in (out.get("fatal_errors") or []) if str(e or "").strip()] if isinstance(out, dict) else []
    if fatal_errors:
        write_json(turn_dir / "pipeline_error.json", {"stage": "step2", "fatal_errors": fatal_errors})
        raise RuntimeError(f"step2 failed: {', '.join(fatal_errors)}")
    return {"step2_output": out}


def _image_intent_without_content(step2_output: dict[str, Any]) -> bool:
    """True when the plan requested adding/replacing an image but the compiled
    state carries no real content image.

    This catches the case where an `image.add`/`image.replace` intent no-opped
    (e.g. the staged upload never made it into `state["images"]`), leaving the
    image model with nothing real to place. Generating anyway makes it fabricate
    a picture, so the caller skips beautify in this case.
    """
    try:
        plan = step2_output.get("plan") if isinstance(step2_output.get("plan"), dict) else {}
        intents = plan.get("content_intents") or []
        wants_image = any(
            isinstance(it, dict) and str(it.get("skill") or "").strip() in {"image.add", "image.replace"}
            for it in intents
        )
        if not wants_image:
            return False
        compile_obj = step2_output.get("compile") if isinstance(step2_output.get("compile"), dict) else {}
        um = compile_obj.get("understand_modified") if isinstance(compile_obj.get("understand_modified"), dict) else {}
        images = [im for im in (um.get("images") or []) if isinstance(im, dict)]
        return len(images) == 0
    except Exception:
        return False


def node_beautify_image(state: PipelineState) -> PipelineState:
    """Step 2.5: when the resolved visual_intent is enabled, generate a
    high-fidelity reference image that step3 will replicate. On any failure we
    degrade gracefully (no reference image -> step3 falls back to the original
    page render), recording the reason instead of failing the turn."""
    step2_output = state.get("step2_output") or {}
    turn_dir = Path(state["turn_dir"])

    if bool(state.get("dry_run", False)):
        write_json(turn_dir / "beautify_image_result.json", {"skipped": True, "reason": "dry_run"})
        return {"beautify_reference_image_path": ""}

    # Resume-from-checkpoint: the previous turn already produced (or legitimately
    # skipped) the reference image for this same demand. Reuse it instead of
    # burning another image-generation call.
    if state.get("resumed_step2_output"):
        reused_ref = str(state.get("resumed_beautify_reference_image_path") or "").strip()
        if reused_ref and Path(reused_ref).exists():
            try:
                shutil.copyfile(reused_ref, turn_dir / "beautify_reference.png")
            except Exception:
                pass
            write_json(
                turn_dir / "beautify_image_result.json",
                {"ok": True, "reused": True, "reason": "resumed_from_prev_turn", "reference_image_path": reused_ref},
            )
            return {"beautify_reference_image_path": reused_ref}
        write_json(
            turn_dir / "beautify_image_result.json",
            {"skipped": True, "reused": True, "reason": "resumed_from_prev_turn_no_reference"},
        )
        return {"beautify_reference_image_path": ""}

    try:
        if not state.get("deck_style") and not step2_has_layout_intent(step2_output):
            write_json(turn_dir / "beautify_image_result.json", {"skipped": True, "reason": "no_layout_intent"})
            return {"beautify_reference_image_path": ""}
    except Exception as e:
        write_json(
            turn_dir / "beautify_image_result.json",
            {"skipped": True, "reason": "has_layout_intent_check_failed", "error": f"{type(e).__name__}: {e}"},
        )
        return {"beautify_reference_image_path": ""}

    # Anti-hallucination guardrail: if the plan asked to add/replace an image but
    # no real content image survived compilation (e.g. the staged upload never
    # reached state["images"]), skip generation. Feeding the image model a slide
    # with "(no content images)" while the request says "add my image" forces it
    # to invent a fake picture. Skipping degrades to the original page render as
    # step3's reference, which never fabricates a photo.
    if _image_intent_without_content(step2_output):
        write_json(
            turn_dir / "beautify_image_result.json",
            {"skipped": True, "reason": "image_intent_without_content"},
        )
        return {"beautify_reference_image_path": ""}

    paths = _paths_from_state(state)
    out_path = paths.beautify_reference_png(int(state["page_num"]))
    _emit_progress(state, "beautify_image")
    try:
        original_understanding = state.get("current_page_state")
        meta = run_beautify_reference_image(
            step2_output=step2_output,
            out_path=out_path,
            original_understanding=original_understanding if isinstance(original_understanding, dict) else None,
            deck_style=state.get("deck_style") if isinstance(state.get("deck_style"), dict) else None,
        )
        # Keep a copy in the turn dir for inspection.
        try:
            shutil.copyfile(out_path, turn_dir / "beautify_reference.png")
        except Exception:
            pass
        write_json(
            turn_dir / "beautify_image_result.json",
            {
                "ok": True,
                "reference_image_path": meta.get("reference_image_path"),
                "attached_images": meta.get("attached_images") or [],
                "generation_mode": meta.get("generation_mode") or "",
                "notes": meta.get("notes") or [],
            },
        )
        write_text(turn_dir / "beautify_image_prompt.txt", str(meta.get("prompt") or ""))
        return {"beautify_reference_image_path": str(out_path)}
    except Exception as e:
        # Degrade gracefully: step3 falls back to the original page render as its
        # reference and still produces a valid result. Beautify is a non-critical
        # enhancement step, so its failure must NOT mark the turn failed — doing
        # so would trigger a full agent-driven re-run that re-plans from scratch
        # and loses one-shot state (e.g. a consumed pending-uploads manifest). We
        # record the reason for inspection but do not append to state["errors"].
        write_json(
            turn_dir / "beautify_image_result.json",
            {"ok": False, "reason": "generation_failed", "error": f"{type(e).__name__}: {e}"},
        )
        if state.get("deck_style"):
            raise
        return {"beautify_reference_image_path": ""}


def node_step3(state: PipelineState) -> PipelineState:
    paths = _paths_from_state(state)
    _emit_progress(state, "reassemble")
    original_png = ""
    current_state = state.get("current_page_state")
    if isinstance(current_state, dict):
        original_png = str(current_state.get("page_png_path") or "").strip()
    if not original_png:
        candidate = paths.reread_page_png(int(state["page_num"]))
        if candidate.exists():
            original_png = str(candidate)
    if not original_png:
        candidate = paths.page_png(int(state["page_num"]))
        if candidate.exists():
            original_png = str(candidate)
    frozen_png: Path | None = None
    if original_png and Path(original_png).exists():
        frozen_png = Path(state["turn_dir"]) / "original_page_context.png"
        shutil.copy2(original_png, frozen_png)

    strategy = state.get("step2_output", {}).get("reassembly_strategy") or {}
    reuse_previous = (
        strategy.get("mode") == "reuse_previous_html"
        and bool((state.get("html_lineage_candidate") or {}).get("available"))
    )
    previous_html: Path | None = None
    if reuse_previous:
        previous_html = Path(str((state["html_lineage_candidate"].get("html_path") or "")))

    # Persist the decision before Step3 so a failed generation still explains
    # whether a prior HTML candidate was selected and why.
    turn_dir = Path(state["turn_dir"])
    write_json(
        turn_dir / "reassembly_strategy.json",
        {
            "requested": strategy,
            "candidate_available": bool((state.get("html_lineage_candidate") or {}).get("available")),
            "used_previous_html": bool(previous_html),
        },
    )

    result = run_step3_without_reference(
        step2_output=state["step2_output"],
        paths=paths,
        page_num=int(state["page_num"]),
        model=str(state["model"]),
        dry_run=bool(state.get("dry_run", False)),
        turn_dir=Path(state["turn_dir"]),
        deck_style=state.get("deck_style") if isinstance(state.get("deck_style"), dict) else None,
        original_page_png=frozen_png,
        mode="edit",
        previous_html=previous_html,
    )
    visual_check = result.get("visual_self_check") or {}
    if visual_check.get("status") == "failed" and visual_check.get("retryable") is False:
        raise RuntimeError("step3 visual self-check did not produce an acceptable page")
    if visual_check.get("verdict") == "revise" and not visual_check.get("repair_applied"):
        raise RuntimeError("step3 visual self-check found an unrepaired major issue")
    write_json(turn_dir / "step3_result.json", {
        "chunk_path": result["chunk_path"],
        "prep_warnings": result.get("prep_warnings") or [],
        "soft_warnings": result.get("soft_warnings") or [],
        "has_layout_intent": bool(result.get("has_layout_intent")),
        "visual_self_check": result.get("visual_self_check") or {},
    })
    # Save the rendered page_block alongside the turn for quick inspection.
    try:
        chunk_path = Path(result["chunk_path"])
        (turn_dir / "chunk_after_step3.html").write_text(chunk_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    return {"step3_result": result}


def node_step4_qa(state: PipelineState) -> PipelineState:
    if bool(state.get("dry_run", False)):
        # Dry-run is meant to validate wiring without heavyweight runtime deps
        # like Playwright browsers.
        result = {"skipped": True, "reason": "dry_run"}
        turn_dir = Path(state["turn_dir"])
        write_json(turn_dir / "qa_result.json", result)
        return {"step4_result": result}

    paths = _paths_from_state(state)
    _emit_progress(state, "qa")
    # QA/autoshrink is a post-step3 polish pass over an already-committed preview.
    # It must never fail the turn: step3 already wrote the chunk and rebuilt
    # index.html, so a QA crash should degrade to "the un-shrunk step3 result"
    # rather than trigger a full re-run.
    try:
        result = run_step4_qa(
            paths=paths,
            page_num=int(state["page_num"]),
            title=str(state.get("title") or "PPTAgent"),
            bundle_dir=Path(state["step3_result"]["bundle_dir"]),
            chunk_path=Path(state["step3_result"]["chunk_path"]),
        )
    except Exception as e:
        result = {"skipped": True, "reason": "qa_failed", "error": f"{type(e).__name__}: {e}"}
    turn_dir = Path(state["turn_dir"])
    write_json(turn_dir / "qa_result.json", result)
    try:
        chunk_path = Path(state.get("step3_result", {}).get("chunk_path") or "")
        if chunk_path.exists():
            (turn_dir / "chunk_after_step4.html").write_text(chunk_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    return {"step4_result": result}


# ---------------------------------------------------------------------------
# Parallel branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NOTE on reread: reread is NO LONGER a graph node. It renders the committed
# preview and re-runs step1 to refresh `current_page_state.json` (the baseline
# for the NEXT turn) — it does not affect what the user sees this turn. Keeping
# it on the critical path made users wait for a full Playwright render + LLM
# call before getting their result. It now runs in the background after the
# pipeline returns (see `run_reread_background` in edit.py / context.py); the
# next edit to the same page waits for any in-flight reread first.
# ---------------------------------------------------------------------------


def _render_html_to_png(*, html_path: Path, out_png: Path, dpi: float = 150.0) -> None:
    from playwright.sync_api import sync_playwright
    from agent_backend.agent.tools.heavy_tool_impl._shared import api_script_dir

    tmp_pdf = out_png.with_suffix(".tmp.pdf")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            pw_page = browser.new_page(viewport={"width": 1280, "height": 720})
            pw_page.goto(html_path.as_uri(), wait_until="load")
            pw_page.evaluate("() => (document.fonts ? document.fonts.ready : true)")
            pw_page.pdf(
                path=str(tmp_pdf),
                print_background=True,
                prefer_css_page_size=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
        finally:
            browser.close()
    tmp_dir = out_png.parent / "_render_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                str(api_script_dir() / "render_pages_png.py"),
                str(tmp_pdf),
                "--out",
                str(tmp_dir),
                "--dpi",
                str(float(dpi)),
            ],
            check=True,
        )
        rendered = tmp_dir / "page_001.png"
        if not rendered.exists():
            raise RuntimeError(f"HTML render produced no PNG: {rendered}")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered, out_png)
    finally:
        try:
            tmp_pdf.unlink()
        except OSError:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


def commit_turn_html_to_pptist(
    *,
    paths: WorkspacePaths,
    page_num: int,
    turn_dir: Path,
    step3_result: dict[str, Any],
    title: str = "PPTAgent",
) -> dict[str, Any]:
    """Commit a turn-local HTML runtime into authoritative PPTist JSON + PNG.

    The PPTist JSON and PNG are both prepared under state/staged first. Only
    after both succeed do we atomically promote them to the page's source of
    truth, preventing a failed render/conversion from leaving a half-written
    page.
    """
    from agent_backend.agent.tools.html_to_pptist import convert_html
    from agent_backend.workspace.assets import (
        clear_staged_candidates,
        materialize_pptist_slide_assets,
    )

    page_num = int(page_num)
    turn_dir = Path(turn_dir)
    html_path = turn_dir / "chunk_after_step4.html"
    if not html_path.exists():
        html_path = Path((step3_result or {}).get("chunk_path") or "")
    if not html_path.exists():
        raise RuntimeError("commit: no final HTML found for this turn")

    bundle_dir = Path((step3_result or {}).get("bundle_dir") or html_path.parent)
    index_html_value = str((step3_result or {}).get("index_html") or "").strip()
    render_html = Path(index_html_value) if index_html_value else Path()
    if not render_html.is_file():
        render_html = html_path

    doc = convert_html(render_html, title=str(title or "PPTAgent"), assets_dir=bundle_dir)
    slides = doc.get("slides") or []
    if not slides:
        raise RuntimeError("commit: HTML->PPTist conversion produced no slide")
    slide = slides[0]
    invalid_images = [
        str(el.get("id") or "")
        for el in (slide.get("elements") or [])
        if isinstance(el, dict)
        and el.get("type") == "image"
        and (float(el.get("width") or 0) <= 0 or float(el.get("height") or 0) <= 0)
    ]
    if invalid_images:
        raise RuntimeError(
            "commit: HTML->PPTist conversion produced zero-size image elements: "
            + ", ".join(invalid_images)
        )

    staged_dir = paths.page_state_dir(page_num) / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_slide = staged_dir / f"commit_slide_{turn_dir.name}.json"
    staged_png = staged_dir / f"commit_reread_{turn_dir.name}.png"
    write_json(staged_slide, slide)
    _render_html_to_png(html_path=render_html, out_png=staged_png)

    os.replace(staged_slide, paths.pptist_slide_json(page_num))
    os.replace(staged_png, paths.reread_page_png(page_num))
    materialize_pptist_slide_assets(paths, page_num, slide)
    clear_staged_candidates(paths, page_num)

    commit = {
        "pptist_slide_json": str(paths.pptist_slide_json(page_num)),
        "reread_page_png": str(paths.reread_page_png(page_num)),
        "final_html": str(html_path),
        "render_html": str(render_html),
        "page_num": page_num,
        "at": time.time(),
    }
    write_json(turn_dir / "commit_result.json", commit)
    try:
        write_lineage(
            paths, page_num, source_turn=turn_dir.name,
            final_html_path=str(html_path), slide=slide,
        )
    except Exception:
        pass
    return commit


def node_commit_to_preview(state: PipelineState) -> PipelineState:
    """Commit turn-local HTML into PPTist JSON + current page PNG."""
    _emit_progress(state, "commit")
    paths = _paths_from_state(state)

    turn_dir = Path(state["turn_dir"])
    commit = commit_turn_html_to_pptist(
        paths=paths,
        page_num=int(state["page_num"]),
        turn_dir=turn_dir,
        step3_result=state.get("step3_result") or {},
        title=str(state.get("title") or "PPTAgent"),
    )
    if _step2_consumed_uploads(state.get("step2_output")):
        from agent_backend.workspace.assets import consume_pending_uploads_for_run

        consume_pending_uploads_for_run(
            paths,
            int(state["page_num"]),
            str(state.get("agent_run_id") or ""),
        )
    return {
        "commit_result": commit,
        "branch_artefacts": {"commit": commit},
    }


def _step2_consumed_uploads(step2_output: Any) -> bool:
    if not isinstance(step2_output, dict):
        return False
    plan = step2_output.get("plan")
    compile_out = step2_output.get("compile")
    if not isinstance(plan, dict) or not isinstance(compile_out, dict):
        return False
    executed = {str(value) for value in compile_out.get("executed_intent_ids") or []}
    return any(
        isinstance(intent, dict)
        and str(intent.get("id") or "") in executed
        and str(intent.get("skill") or "") in {"image.add", "image.replace"}
        for intent in plan.get("content_intents") or []
    )


def node_finalize(state: PipelineState) -> PipelineState:
    finished = time.time()
    turn_dir = Path(state["turn_dir"])

    manifest = {
        "project_id": state["project_id"],
        "page_num": int(state["page_num"]),
        "demand": str(state.get("demand") or ""),
        "model": str(state["model"]),
        "dry_run": bool(state.get("dry_run", False)),
        "started_at": float(state.get("started_at") or 0.0),
        "finished_at": float(finished),
        "duration_seconds": float(finished - float(state.get("started_at") or finished)),
        "baseline_ran": bool(state.get("baseline_ran")),
        "understanding_status": str(state.get("understanding_status") or "cached"),
        "deck_style_revision": int(state.get("deck_style_revision") or 0),
        "used_deck_style": bool(state.get("deck_style")),
        "reassembly_strategy": (state.get("step2_output") or {}).get("reassembly_strategy") or {},
        "html_lineage_candidate": state.get("html_lineage_candidate") or {},
        "commit_result": state.get("commit_result") or {},
        "reread_result": state.get("reread_result") or {},
        "step3_chunk_path": str((state.get("step3_result") or {}).get("chunk_path") or ""),
        "html_runtime_index": str(Path(state.get("step3_result", {}).get("index_html") or "")),
        "errors": list(state.get("errors") or []),
    }
    write_json(turn_dir / "manifest.json", manifest)
    return {"finished_at": finished}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def build_pipeline_graph():
    g: StateGraph = StateGraph(PipelineState)
    g.add_node("load_state", node_load_state)
    g.add_node("step2", node_step2)
    g.add_node("step3", node_step3)
    g.add_node("step4_qa", node_step4_qa)
    g.add_node("commit_to_preview", node_commit_to_preview)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "load_state")
    g.add_edge("load_state", "step2")
    g.add_edge("step2", "step3")
    g.add_edge("step3", "step4_qa")
    g.add_edge("step4_qa", "commit_to_preview")
    g.add_edge("commit_to_preview", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def get_pipeline_graph():
    """Lazy singleton so repeated invocations reuse the compiled graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_pipeline_graph()
    return _GRAPH


def run_pipeline_once(
    *,
    project_id: str,
    page_num: int,
    demand: str,
    model: str = "claude-opus-4-8",
    dry_run: bool = False,
    title: str = "PPTAgent",
    display_page: int | None = None,
    batch_index: int = 0,
    current_page_state: dict[str, Any] | None = None,
    understanding_status: str = "cached",
    deck_style: dict[str, Any] | None = None,
    deck_style_revision: int = 0,
    agent_run_id: str = "",
) -> dict[str, Any]:
    """Run one task through the pipeline and return the final state.

    `display_page`/`batch_index` are used only for progress reporting (SSE
    `task_progress` events); they default to the slot / 0 when unset.
    """
    graph = get_pipeline_graph()
    init: PipelineState = {
        "project_id": project_id,
        "page_num": int(page_num),
        "demand": demand,
        "model": model,
        "dry_run": bool(dry_run),
        "title": title,
        "display_page": int(display_page if display_page is not None else page_num),
        "batch_index": int(batch_index),
        "agent_run_id": str(agent_run_id or ""),
        "current_page_state": current_page_state or {},
        "understanding_status": str(understanding_status or "cached"),
        "deck_style": deck_style or {},
        "deck_style_revision": int(deck_style_revision or 0),
        "errors": [],
        "branch_artefacts": {},
    }
    return graph.invoke(init)
