"""Step3 implementation for the reference-image-disabled pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from agent_backend.agent.tools.heavy_tool_impl._shared import api_script_dir
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition import steps
from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.step_3_reassemble import (
    step3_reassemble_mvp as s3,
)


def _normalize_visual_self_check(value: Any) -> dict[str, Any]:
    """Normalize the current visual-critic contract conservatively."""
    if not isinstance(value, dict):
        return {"status": "unavailable", "verdict": "unavailable", "severity": "none", "issues": []}
    result = dict(value)
    if not isinstance(result.get("issues"), list):
        return {"status": "unavailable", "verdict": "unavailable", "severity": "none", "issues": []}
    issues = result["issues"]
    verdict = str(result.get("verdict") or "").strip().lower()
    severity = str(result.get("severity") or "").strip().lower()
    nested = [str(item.get("severity") or "").strip().lower() for item in issues if isinstance(item, dict)]
    if not severity:
        severity = "major" if any(item in {"critical", "high", "major"} for item in nested) else ("minor" if any(item in {"minor", "low"} for item in nested) else "none")
    if severity in {"critical", "high"}:
        severity = "major"
    if verdict == "pass" and severity == "major":
        verdict = "revise"
    if verdict not in {"pass", "revise"}:
        verdict = "unavailable"
    result.update({"verdict": verdict, "severity": severity, "issues": issues})
    result["status"] = "unavailable" if verdict == "unavailable" else ("passed" if verdict == "pass" else "needs_revision")
    return result


def _disabled_visual_self_check() -> dict[str, Any]:
    return {"status": "skipped", "reason": "disabled_by_config", "verdict": "not_run", "severity": "none", "repair_applied": False}


def _system_prompt(mode: str) -> str:
    prompt = s3._SYSTEM_PROMPT_BASE
    prompt = re.sub(
        r"- `reference_image` \(image\):[\s\S]*?exact rules\.\n",
        "",
        prompt,
        count=1,
    )
    replacements = {
        "- Generate the HTML for THIS page only, reproducing `reference_image` as faithfully and beautifully as possible.":
            "- Generate the HTML for THIS page only from the authoritative PageSpec and visual brief.",
        "match the reference image's look as closely as possible":
            "create a coherent, polished page that follows the supplied design brief",
        "when the reference clearly shows a gradient/shadow":
            "when a restrained gradient/shadow materially supports the design",
        "follow the reference image and content needs when they differ":
            "follow the content and design brief when they differ",
        "reference image's size and the space it fills":
            "PageSpec image usage, aspect ratio, and the space it should fill",
        "from `reference_image` or from images":
            "from content images",
        "the visual density of `reference_image`":
            "the intended visual density of the page",
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    # Keep the old shared contract's image-specific wording out of this route.
    # Existing-page mode may still receive a BEFORE screenshot, but it is only
    # context and must never be described as the target to reproduce.
    prompt = prompt.replace("reference_image", "before_page_context")
    prompt = prompt.replace("reference image", "before-page context")
    if mode == "create":
        prompt = "\n".join(
            line for line in prompt.splitlines()
            if "before_page_context" not in line and "before-page context" not in line
        )
    else:
        prompt = prompt.replace(
            "- `before_page_context` (image): the VISUAL TARGET for this page.",
            "- `before_page_context` (image): an ORIGINAL-page context image only; it is not a target to copy.",
        )

    if mode == "create":
        mode_text = """
Creation mode:
- There is no original-page screenshot. The page starts on a blank canvas, but
  the PageSpec already contains the final content, semantic roles, image/table
  structure, and whole-page visual organization. Do not invent missing content.
- `visual_intent.requirements_text` is a final-page design brief, not a delta.
- Use `texts[].kind` to establish hierarchy. Use the full canvas deliberately;
  avoid both cramped content and accidental large unused regions.
"""
    else:
        mode_text = """
Existing-page mode:
- The attached image is the ORIGINAL page before the edit. It is context, not a
  target to copy pixel-for-pixel.
- If `previous_final_html` is supplied, use it as a structure and style
  reference, but always output a complete newly generated HTML page.
- Never output an HTML diff, DOM patch, or instructions for another model.
- `visual_change_request` describes the requested change. Implement it even if
  the changed region must have a different arrangement.
- Preserve visual traits and regions that the user did not ask to change, while
  allowing necessary reflow. User requirements and the final PageSpec win over
  original coordinates.
- Never OCR the original image. Final text and content images come from the
  PageSpec only.
"""
    prompt = prompt.replace("reference_image", "design_brief").replace("reference image", "design brief")
    return prompt.rstrip() + "\n\n" + mode_text.strip() + "\n"


def _user_prompt(
    *, step2: dict[str, Any], page_state: dict[str, Any], css_doc: str,
    mode: str, previous_html: str = "",
) -> str:
    state = dict(page_state)
    original_layout = str(state.pop("original_layout_description_en", "") or "").strip()
    user_request = str(state.pop("user_request", "") or "").strip()
    defaults = state.pop("default_layout_details", None) or []
    vi = step2.get("visual_intent") if isinstance(step2.get("visual_intent"), dict) else {}
    visual = str(vi.get("requirements_text") or "").strip()
    strategy = step2.get("reassembly_strategy") if isinstance(step2.get("reassembly_strategy"), dict) else {}
    compile_obj = step2.get("compile") if isinstance(step2.get("compile"), dict) else {}
    table_render_context = (
        step2.get("table_render_context")
        if isinstance(step2.get("table_render_context"), dict)
        else (compile_obj.get("table_render_context") if isinstance(compile_obj.get("table_render_context"), dict) else {})
    )
    if mode == "create":
        brief = visual or user_request
    else:
        brief = visual or user_request
    notes = ""
    if original_layout:
        notes += "\noriginal_page_layout_notes:\n" + original_layout
    if defaults:
        notes += "\ndefault_layout_details:\n" + "\n".join(f"- {x}" for x in defaults if str(x).strip())
    previous_section = ""
    if previous_html:
        previous_section = (
            "\n\nprevious_final_html (reference only; output a complete new HTML document):\n"
            + previous_html
        )
    return (
        "mode:\n" + mode
        + "\n\npage_state (authoritative final content):\n"
        + json.dumps(state, ensure_ascii=False, indent=2)
        + "\n\nrequired_refs:\n"
        + json.dumps(state.get("required_refs") or {"all": []}, ensure_ascii=False, indent=2)
        + "\n\nvisual_change_request_or_final_design:\n"
        + brief
        + "\n\nreassembly_strategy:\n"
        + json.dumps(strategy, ensure_ascii=False)
        + "\n\ntable_render_context (internal rendering guidance; not page content):\n"
        + json.dumps(table_render_context, ensure_ascii=False)
        + "\n\nfull_user_request:\n"
        + user_request
        + notes
        + "\n\ncss_library_doc:\n"
        + css_doc
        + "\n"
        + previous_section
    )


def _render(html_path: Path, png_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1400, "height": 850}, device_scale_factor=1)
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts ? document.fonts.ready : Promise.resolve()")
            page.wait_for_timeout(250)
            slide = page.query_selector(".page")
            if slide is None:
                raise RuntimeError("generated HTML has no .page element")
            slide.screenshot(path=str(png_path))
        finally:
            browser.close()


def _comparison(before: Path, current: Path, output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    left = Image.open(before).convert("RGB")
    right = Image.open(current).convert("RGB")
    width = 1100
    height = round(width * left.height / left.width)
    left = left.resize((width, height))
    right = right.resize((width, height))
    canvas = Image.new("RGB", (width * 2, height + 42), "#eeeeee")
    canvas.paste(left, (0, 42))
    canvas.paste(right, (width, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 14), "BEFORE (context)", fill="#111111", font=font)
    draw.text((width + 16, 14), "CURRENT", fill="#111111", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


_CRITIC_SYSTEM_PROMPT = """You are a strict but practical visual QA critic.
Judge the supplied rendered presentation page and return only the requested JSON.
Do not write HTML, explanations, or markdown.
"""


_CRITIC_PROMPT = """You are a visual QA critic for one presentation page.
Return JSON only:
{"verdict":"pass|revise","severity":"none|minor|major","issues":[
 {"region":"...","category":"requirement|content|style|geometry|readability|image|color|whitespace",
  "observed":"...","expected":"...","repair":"smallest useful HTML/CSS repair"}]}

Judge only material problems. Check final content coverage, explicit visual
requirements, readability, overflow, clipping, overlap, broken images, contrast,
alignment, balance, editable HTML, and whether content is unnecessarily tiny or
clustered in a small area leaving a large accidental blank region. Intentional
whitespace, minimalist slides, and user-requested blank space are valid.
For an existing-page edit, the left BEFORE image is context rather than a pixel
target: requested changes and necessary reflow are correct. Penalize only
unjustified loss of important style/regions or failure to implement the request.
Use revise/major only for a defect a user would immediately notice. Minor polish
does not trigger repair. Return at most five issues.
"""


def _critic_prompt(step2: dict[str, Any], page_state: dict[str, Any], html: str, mode: str) -> str:
    compact = {
        "mode": mode,
        "reassembly_strategy": step2.get("reassembly_strategy") or {},
        "user_request": step2.get("user_request") or "",
        "visual_intent": step2.get("visual_intent") or {},
        "page_state": {
            "palette": page_state.get("palette") or {},
            "texts": page_state.get("texts") or [],
            "images": page_state.get("images") or [],
            "tables": page_state.get("tables") or [],
        },
    }
    return (
        "Evaluate the rendered page using the supplied visual evidence and contract.\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
        + "\n\nWhen reassembly_strategy.mode is reuse_previous_html, also check that regions not targeted by the user did not drift unnecessarily.\n"
        + "\n\ncurrent_html:\n"
        + re.sub(r"\s+", " ", html)[:16000]
        + "\n\nReturn only JSON."
    )


def _repair_prompt(original: str, page_state: dict[str, Any], html: str, issues: list[Any]) -> str:
    return (
        original
        + "\n\nVISUAL SELF-CHECK REPAIR\n"
        + "The attached image shows the current rendered page (and, for an edit, BEFORE on the left). "
        + "Repair only the listed major issues. Preserve correct content, requested changes, image sources, "
        + "data-ref values, and editable structure. Do not add content or redesign unrelated regions. Output "
        + "the complete HTML between PAGE_START and PAGE_END markers.\n\nissues:\n"
        + json.dumps(issues[:5], ensure_ascii=False, indent=2)
        + "\n\nfinal_page_state:\n"
        + json.dumps(page_state, ensure_ascii=False)
        + "\n\ncurrent_html:\n"
        + html
    )


def _write_runtime(*, step2: dict[str, Any], page_block: str, paths: Any, page_num: int, turn_dir: Path) -> tuple[Path, Path]:
    runtime = turn_dir / "html_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    css_src = api_script_dir() / "css_library.css"
    if css_src.exists():
        shutil.copy2(css_src, runtime / "css_library.css")
    (runtime / "styles.css").write_text("", encoding="utf-8")
    chunk_dir = runtime / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{int(page_num):03d}_{int(page_num):03d}.html"
    html = steps._build_chunk_html(
        page_block=page_block,
        base_href="../",
        title="PPTAgent",
        s3_mod=s3,
    )
    html = steps._publish_referenced_assets(
        step2_output=step2,
        chunk_html=html,
        paths=paths,
        page_num=page_num,
        bundle_dir=runtime,
    )
    chunk_path.write_text(html, encoding="utf-8")
    steps._reassemble_runtime_index_html(bundle_dir=runtime, title="PPTAgent")
    return chunk_path, runtime / "index.html"


def run_step3_without_reference(
    *, step2_output: dict[str, Any], paths: Any, page_num: int, model: str,
    dry_run: bool, turn_dir: Path, original_page_png: Path | None = None,
    deck_style: dict[str, Any] | None = None, mode: str | None = None,
    previous_html: Path | None = None,
) -> dict[str, Any]:
    page_state = s3._extract_page_state(step2_output)
    mode = mode or ("create" if bool(step2_output.get("fill_mode")) else "edit")
    system = _system_prompt(mode)
    previous_html_text = ""
    if previous_html and previous_html.exists():
        previous_html_text = previous_html.read_text(encoding="utf-8", errors="replace")
        (turn_dir / "previous_final_html.html").write_text(previous_html_text, encoding="utf-8")
    user = _user_prompt(
        step2=step2_output,
        page_state=page_state,
        css_doc=steps._css_library_doc(paths),
        mode=mode,
        previous_html=previous_html_text,
    )
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "step3_system_prompt.txt").write_text(system, encoding="utf-8")
    (turn_dir / "step3_user_prompt.txt").write_text(user, encoding="utf-8")
    if dry_run:
        page_block = steps._dry_run_page_block(page_state=page_state, page_num=page_num)
        debug: dict[str, Any] = {"raw": "", "hard_errors": [], "soft_warnings": []}
    else:
        base_url, api_key = steps._resolve_llm_backend()
        page_block, debug = s3._generate_page_block_with_repair(
            base_url=base_url, api_key=api_key, model=model,
            max_output_tokens=8192, temperature=0.0, retries=2,
            retry_base_seconds=2.0, retry_max_seconds=60.0, retry_backoff=2.0,
            system_prompt=system, user_prompt=user,
            page_png_bytes=original_page_png.read_bytes() if original_page_png and original_page_png.exists() else None,
            page_state=page_state,
        )
    def write_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    write_text(turn_dir / "step3_raw_response.txt", str(debug.get("raw") or ""))

    def write_json(path: Path, value: Any) -> None:
        write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write_json(turn_dir / "step3_validation.json", {"hard_errors": debug.get("hard_errors") or [], "soft_warnings": debug.get("soft_warnings") or []})
    if not page_block:
        raise RuntimeError(f"step3 without reference failed: {debug.get('hard_errors') or 'empty output'}")

    chunk, index = _write_runtime(step2=step2_output, page_block=page_block, paths=paths, page_num=page_num, turn_dir=turn_dir)
    current_png = turn_dir / "step3_visual_current.png"
    _render(index, current_png)
    enabled = os.environ.get("PPT_STEP3_VISUAL_SELF_CHECK", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        check = _disabled_visual_self_check()
        write_json(turn_dir / "step3_visual_check.json", check)
        return {
            "chunk_path": str(chunk),
            "bundle_dir": str(index.parent),
            "index_html": str(index),
            "page_state": page_state,
            "has_layout_intent": bool(step2_output.get("visual_intent")),
            "visual_self_check": check,
            "prep_warnings": debug.get("soft_warnings") or [],
            "soft_warnings": debug.get("soft_warnings") or [],
        }
    compare_path: Path | None = None
    critic_image = current_png
    if original_page_png and original_page_png.exists():
        compare_path = turn_dir / "step3_before_current_comparison.png"
        _comparison(original_page_png, current_png, compare_path)
        critic_image = compare_path
    prompt = _critic_prompt(step2_output, page_state, chunk.read_text(encoding="utf-8"), mode)
    write_text(turn_dir / "step3_visual_check_prompt.txt", prompt)
    if dry_run:
        critic = {"verdict": "pass", "severity": "none", "issues": [], "status": "dry_run"}
    else:
        try:
            base_url, api_key = steps._resolve_llm_backend()
            raw = s3._openai_chat_completion(
                base_url=base_url, api_key=api_key, model=model,
                system_prompt=_CRITIC_SYSTEM_PROMPT, user_text=prompt,
                image_bytes=critic_image.read_bytes(), max_tokens=1800,
                temperature=0.0, retries=1, retry_base_seconds=1.0,
                retry_max_seconds=8.0, retry_backoff=2.0,
                tag="step3.without_reference.visual_critic",
            )
            write_text(turn_dir / "step3_visual_check_raw.txt", raw or "")
            critic = json.loads(re.sub(r"^```json\s*|\s*```$", "", (raw or "").strip(), flags=re.I))
        except Exception as exc:
            write_text(turn_dir / "step3_visual_check_error.txt", f"{type(exc).__name__}: {exc}")
            critic = {"verdict": "unavailable", "severity": "none", "issues": [], "status": "unavailable"}
    critic = _normalize_visual_self_check(critic)
    verdict = critic.get("verdict")
    severity = critic.get("severity")
    write_json(turn_dir / "step3_visual_check.json", critic)
    repaired = False
    if critic.get("verdict") == "revise" and critic.get("severity") == "major" and not dry_run:
        repair = _repair_prompt(user, page_state, chunk.read_text(encoding="utf-8"), critic.get("issues") or [])
        write_text(turn_dir / "step3_visual_repair_prompt.txt", repair)
        base_url, api_key = steps._resolve_llm_backend()
        raw = s3._openai_chat_completion(
            base_url=base_url, api_key=api_key, model=model,
            system_prompt=system, user_text=repair,
            image_bytes=critic_image.read_bytes(), max_tokens=8192,
            temperature=0.0, retries=1, retry_base_seconds=1.0,
            retry_max_seconds=8.0, retry_backoff=2.0,
            tag="step3.without_reference.visual_repair",
        )
        write_text(turn_dir / "step3_visual_repair_raw.txt", raw or "")
        candidate = s3._extract_page_markup(raw)
        if candidate:
            candidate, hard, soft = s3._validate_and_normalize_page_block(page_state=page_state, page_block=candidate)
            write_json(turn_dir / "step3_visual_repair_validation.json", {"hard_errors": hard, "soft_warnings": soft})
            if not hard:
                page_block = candidate
                _write_runtime(step2=step2_output, page_block=page_block, paths=paths, page_num=page_num, turn_dir=turn_dir)
                repaired = True
    final_status = "repaired" if repaired else ("passed" if verdict == "pass" else ("unavailable" if verdict == "unavailable" else "failed"))
    return {
        "page_block": page_block,
        "chunk_path": str(chunk),
        "bundle_dir": str(turn_dir / "html_runtime"),
        "index_html": str(index),
        "has_layout_intent": bool(s3._has_layout_intent(step2_output)),
        "retryable": False if final_status == "failed" else True,
        "visual_self_check": {
            "status": final_status,
            "repair_applied": repaired,
            "verdict": verdict,
            "severity": severity,
            "retryable": False if final_status == "failed" else True,
        },
    }
