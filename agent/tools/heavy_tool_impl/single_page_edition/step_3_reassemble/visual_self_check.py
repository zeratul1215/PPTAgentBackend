"""One-round visual self-check for Step 3 HTML output.

This module deliberately treats the rendered HTML as the current truth.  It
never compares a PPTist rendering and it fails open: a critic or repair
failure keeps the already validated Step 3 result.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


CRITIC_SYSTEM_PROMPT = """You are a visual QA critic for one editable presentation page.
The comparison image has TARGET on the left and CURRENT on the right. Judge
only material visual problems; small font rasterization, spacing, and color
differences are acceptable.

Return JSON only:
{"verdict":"pass|revise","severity":"none|minor|major","issues":[
 {"region":"...","category":"content|structure|geometry|readability|image|color|polish",
  "observed":"...","expected":"...","repair":"..."}]}

Use revise/major only for a missing or extra region, wrong column or hierarchy,
large geometry/color mismatch, missing or misplaced image, unreadable text, or
another clearly serious problem. Do not require pixel-perfect matching. Return
at most five issues ordered by impact.
"""


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_critic(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    verdict = str(value.get("verdict") or "").strip().lower()
    severity = str(value.get("severity") or "").strip().lower()
    if verdict not in {"pass", "revise"} or severity not in {"none", "minor", "major"}:
        return None
    issues = value.get("issues") if isinstance(value.get("issues"), list) else []
    return {"verdict": verdict, "severity": severity, "issues": issues[:5]}


def render_html_page(html_path: Path, png_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
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


def make_comparison_image(target_path: Path, current_path: Path, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    target = Image.open(target_path).convert("RGB")
    current = Image.open(current_path).convert("RGB")
    width = 1100
    height = round(width * target.height / target.width)
    target = target.resize((width, height))
    current = current.resize((width, height))
    canvas = Image.new("RGB", (width * 2, height + 42), "#eeeeee")
    canvas.paste(target, (0, 42))
    canvas.paste(current, (width, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 14), "TARGET", fill="#111111", font=font)
    draw.text((width + 16, 14), "CURRENT", fill="#111111", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _critic_prompt(*, page_state: dict[str, Any], current_html: str) -> str:
    # Keep the model's structural context compact. The image is the primary
    # evidence; full HTML is useful for locating a repair but needlessly large
    # for the critic.
    compact = {
        "texts": page_state.get("texts") or [],
        "images": page_state.get("images") or [],
        "tables": page_state.get("tables") or [],
    }
    return (
        "Compare TARGET (left) and CURRENT (right).\n\n"
        "Authoritative content (do not judge OCR from the image):\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\n\nCurrent HTML structural summary:\n"
        + re.sub(r"\s+", " ", current_html)[:12000]
        + "\n\nReturn only the requested JSON."
    )


def _repair_prompt(*, original_user_prompt: str, page_state: dict[str, Any], current_html: str, issues: list[Any]) -> str:
    return (
        "VISUAL SELF-CHECK REPAIR\n"
        "The comparison image has TARGET on the left and CURRENT on the right.\n"
        "Repair only the material issues listed below. Preserve correct layout, exact text, image refs, "
        "data-ref values, and editable structure. Do not add content. Return the complete HTML page between "
        "PAGE_START and PAGE_END markers.\n\n"
        "Original request:\n" + original_user_prompt + "\n\n"
        "Authoritative page content:\n" + json.dumps(page_state, ensure_ascii=False) + "\n\n"
        "Issues:\n" + json.dumps(issues[:5], ensure_ascii=False) + "\n\n"
        "Current HTML:\n" + current_html
    )


def run_visual_self_check(
    *,
    step3_module: Any,
    page_block: str,
    page_state: dict[str, Any],
    system_prompt: str,
    original_user_prompt: str,
    reference_path: Path,
    current_html_path: Path,
    turn_dir: Path,
    model: str,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retry_backoff: float,
) -> dict[str, Any]:
    """Critique and, at most once, repair a rendered Step3 page."""
    result: dict[str, Any] = {
        "status": "error",
        "severity": "none",
        "repair_applied": False,
    }
    initial_html = current_html_path.read_text(encoding="utf-8", errors="replace")
    _write_text(turn_dir / "chunk_after_step3_initial.html", initial_html)
    current_png = turn_dir / "step3_visual_current.png"
    comparison_png = turn_dir / "step3_visual_comparison.png"
    try:
        render_html_page(current_html_path, current_png)
        make_comparison_image(reference_path, current_png, comparison_png)
        prompt = _critic_prompt(page_state=page_state, current_html=initial_html)
        _write_text(turn_dir / "step3_visual_check_prompt.txt", prompt)
        base_url, api_key = step3_module._resolve_backend(api_key_arg="")
        started = time.monotonic()
        raw = step3_module._openai_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_text=prompt,
            image_bytes=comparison_png.read_bytes(),
            max_tokens=1800,
            temperature=0.0,
            retries=1,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            retry_backoff=retry_backoff,
            tag="step3.visual_critic",
        )
        _write_text(turn_dir / "step3_visual_check_raw.txt", raw or "")
        critic = _normalize_critic(_extract_json(raw))
        if critic is None:
            result.update({"status": "error", "reason": "invalid_critic_json"})
            _write_json(turn_dir / "step3_visual_check.json", result)
            return {"page_block": page_block, "visual_self_check": result}
        _write_json(turn_dir / "step3_visual_check.json", critic)
        result.update({"status": "passed", "severity": critic["severity"], "verdict": critic["verdict"]})
        if critic["verdict"] != "revise" or critic["severity"] != "major":
            return {"page_block": page_block, "visual_self_check": result}

        repair_prompt = _repair_prompt(
            original_user_prompt=original_user_prompt,
            page_state=page_state,
            current_html=initial_html,
            issues=critic.get("issues") or [],
        )
        _write_text(turn_dir / "step3_visual_repair_prompt.txt", repair_prompt)
        repair_raw = step3_module._openai_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_text=repair_prompt,
            image_bytes=comparison_png.read_bytes(),
            max_tokens=8192,
            temperature=0.0,
            retries=1,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            retry_backoff=retry_backoff,
            tag="step3.visual_repair",
        )
        _write_text(turn_dir / "step3_visual_repair_raw.txt", repair_raw or "")
        candidate = step3_module._extract_page_markup(repair_raw)
        if not candidate:
            result.update({"status": "repair_rejected", "severity": "major", "reason": "missing_page_markers"})
            _write_json(turn_dir / "step3_visual_check.json", result)
            return {"page_block": page_block, "visual_self_check": result}
        normalized, hard, soft = step3_module._validate_and_normalize_page_block(
            page_state=page_state, page_block=candidate
        )
        _write_json(turn_dir / "step3_visual_repair_validation.json", {"hard_errors": hard, "soft_warnings": soft})
        if hard:
            result.update({"status": "repair_rejected", "severity": "major", "reason": "repair_validation_failed"})
            _write_json(turn_dir / "step3_visual_check.json", result)
            return {"page_block": page_block, "visual_self_check": result}
        _write_text(turn_dir / "chunk_after_step3_repaired.html", normalized)
        result.update({"status": "repaired", "severity": "major", "repair_applied": True,
                       "critic_elapsed_s": round(time.monotonic() - started, 2)})
        return {"page_block": normalized, "visual_self_check": result}
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        _write_json(turn_dir / "step3_visual_check.json", result)
        return {"page_block": page_block, "visual_self_check": result}
