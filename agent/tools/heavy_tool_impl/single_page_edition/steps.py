"""Thin wrappers around the vendored `agent_backend/step_*` modules.

Each wrapper exposes a single pure function the LangGraph nodes can call
without going through subprocess + JSON files. The wrappers do not own
any disk state; the LangGraph nodes are responsible for persistence.
"""

from __future__ import annotations

import html
import importlib
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from agent_backend.agent.tools.heavy_tool_impl._shared import api_script_dir
from agent_backend.workspace.paths import WorkspacePaths, read_json, write_json, write_text


def _resolve_llm_backend() -> tuple[str, str]:
    """Resolve the OpenAI-compatible backend used by step1/2/3.

    We intentionally raise RuntimeError (not SystemExit) because this module is
    used inside a server / LangGraph runner, not a CLI.
    """
    base_url = (os.environ.get("PPT_LLM_BASE_URL") or "").strip()
    api_key = (os.environ.get("PPT_LLM_API_KEY") or "").strip()
    if not base_url:
        raise RuntimeError(
            "Missing PPT_LLM_BASE_URL (OpenAI-compatible endpoint, e.g. https://eoeo.xyz/v1)."
        )
    if not api_key:
        raise RuntimeError("Missing PPT_LLM_API_KEY.")
    return base_url, api_key


def _load_step_modules():
    base = "agent_backend.agent.tools.heavy_tool_impl.single_page_edition"
    s1 = importlib.import_module(f"{base}.step_1_understand.step1_understand_recompose_mvp")
    s2 = importlib.import_module(f"{base}.step_2_plan.step2_plan_compile_mvp")
    s3 = importlib.import_module(f"{base}.step_3_reassemble.step3_reassemble_mvp")
    s4 = importlib.import_module(f"{base}.step_4_qa.step4_qa_mvp")
    return s1, s2, s3, s4


# ---------------------------------------------------------------------------
# Step 1 (merged understand + recompose)
# ---------------------------------------------------------------------------


def run_step1(
    *, understand_input: dict[str, Any], model: str, dry_run: bool,
    focus_requests: list[str] | None = None,
) -> dict[str, Any]:
    s1, *_ = _load_step_modules()
    return s1.understand_step(
        input_obj=understand_input,
        api_key=None,
        model=model,
        dry_run=dry_run,
        focus_requests=focus_requests,
    )


# ---------------------------------------------------------------------------
# Step 2 (plan + compile, single page)
# ---------------------------------------------------------------------------


def run_step2(
    *,
    user_request: str,
    understand_output: dict[str, Any],
    selected_refs: list[str] | None = None,
    previous_html_available: bool = False,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    _, s2, *_ = _load_step_modules()
    request_obj = {
        "user_request": str(user_request or ""),
        "selected_refs": list(selected_refs or []),
        "understand_output": understand_output,
        "previous_html_available": bool(previous_html_available),
    }
    return s2.step2_run(
        request_obj=request_obj,
        api_key=None,
        model=model,
        dry_run=dry_run,
        previous_html_available=bool(previous_html_available),
    )


# ---------------------------------------------------------------------------
# Step 3 (reassemble, single page) — we reuse the private helpers because
# step3 ships a `main()` only.
# ---------------------------------------------------------------------------


def _build_chunk_html(*, page_block: str, base_href: str, title: str, s3_mod: Any) -> str:
    return s3_mod._wrap_chunk_html(page_block=page_block, base_href=base_href, title=title)


def run_step3_single_page(
    *,
    step2_output: dict[str, Any],
    paths: WorkspacePaths,
    page_num: int,
    model: str,
    dry_run: bool,
    reference_image_path: str = "",
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 2,
    retry_base_seconds: float = 2.0,
    retry_max_seconds: float = 60.0,
    retry_backoff: float = 2.0,
    title: str = "PPTAgent",
    turn_dir: Path | None = None,
    deck_style: dict[str, Any] | None = None,
    on_visual_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run Step 3 against a single page in a turn-local HTML runtime.

    `reference_image_path`, when provided, is a beautify reference image that
    becomes the layout target. Otherwise step3 falls back to the original page
    render. Either way step3 runs a single (unified) prompt.

    Returns a dict with `chunk_path`, `page_block`, and any soft warnings.
    """
    _, _, s3, _ = _load_step_modules()

    page_state = s3._extract_page_state(step2_output)
    prep_warnings: list[str] = []
    has_layout = s3._has_layout_intent(step2_output)
    system_prompt = s3._SYSTEM_PROMPT
    user_prompt = s3._build_user_prompt(
        page_state=page_state,
        css_library_doc=_css_library_doc(paths),
        has_layout_intent=has_layout,
        deck_style=deck_style if isinstance(deck_style, dict) else None,
    )

    # Reference image resolution order:
    #   1) explicit beautify reference image (from step 2.5),
    #   2) the original page render recorded in step2 output,
    #   3) the workspace page render.
    page_png_bytes: bytes | None = None
    used_beautify_reference = False
    ref_s = str(reference_image_path or "").strip()
    if ref_s:
        rp = Path(ref_s).expanduser()
        if rp.exists():
            try:
                page_png_bytes = rp.read_bytes()
                used_beautify_reference = True
            except Exception:
                page_png_bytes = None

    if page_png_bytes is None:
        try:
            compile_obj = step2_output.get("compile") if isinstance(step2_output.get("compile"), dict) else {}
            um = compile_obj.get("understand_modified") if isinstance(compile_obj.get("understand_modified"), dict) else {}
            png_s = str(um.get("page_png_path") or "").strip()
            if png_s:
                p = Path(png_s).expanduser().resolve()
                if p.exists():
                    page_png_bytes = p.read_bytes()
        except Exception:
            page_png_bytes = None

    if page_png_bytes is None:
        png_path = paths.page_png(page_num)
        if png_path.exists():
            page_png_bytes = png_path.read_bytes()

    debug: dict[str, Any] = {}
    if dry_run:
        page_block = _dry_run_page_block(page_state=page_state, page_num=page_num)
    else:
        base_url, api_key = _resolve_llm_backend()
        page_block, debug = s3._generate_page_block_with_repair(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_output_tokens=int(max_output_tokens),
            temperature=float(temperature),
            retries=int(retries),
            retry_base_seconds=float(retry_base_seconds),
            retry_max_seconds=float(retry_max_seconds),
            retry_backoff=float(retry_backoff),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            page_png_bytes=page_png_bytes,
            page_state=page_state,
        )
        if not page_block:
            hard = debug.get("hard_errors") if isinstance(debug.get("hard_errors"), list) else []
            raise RuntimeError(f"step3 failed: {hard}")

    runtime_dir = Path(turn_dir) / "html_runtime" if turn_dir is not None else paths.root / "_tmp_html_runtime"
    bundle_dir = runtime_dir
    chunks_dir = bundle_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    css = api_script_dir() / "css_library.css"
    if css.exists():
        css_text = css.read_text(encoding="utf-8", errors="replace")
        fonts_src = Path(__file__).resolve().parents[5] / "agent_frontend" / "pptist" / "src" / "assets" / "fonts"
        font_faces: list[str] = []
        if fonts_src.exists():
            fonts_dst = bundle_dir / "fonts"
            fonts_dst.mkdir(parents=True, exist_ok=True)
            for font_path in sorted(fonts_src.glob("*.woff2")):
                shutil.copy2(font_path, fonts_dst / font_path.name)
                family = font_path.stem
                font_faces.append(
                    f"@font-face {{ font-display: swap; font-family: '{family}'; src: url('./fonts/{font_path.name}') format('woff2'); }}"
                )
        if font_faces:
            css_text = "\n".join(font_faces) + "\n\n" + css_text
        write_text(bundle_dir / "css_library.css", css_text)
    # Some wrappers still reference styles.css; keep it harmless and local.
    if not (bundle_dir / "styles.css").exists():
        write_text(bundle_dir / "styles.css", "")

    base_href = "../"
    chunk_path = chunks_dir / f"chunk_{int(page_num):03d}_{int(page_num):03d}.html"
    chunk_html = _build_chunk_html(page_block=page_block, base_href=base_href, title=title, s3_mod=s3)

    # Copy any per-turn image assets the chunk references (e.g. a content-hash
    # slide image or a user-uploaded `图片 1.png`) into this turn's runtime.
    # The chunk uses <base href="../">, so rewritten `images/page{slot}/...`
    # srcs resolve from `html_runtime/` for HTML->PPTist conversion and the
    # final reread PNG render.
    chunk_html = _publish_referenced_assets(
        step2_output=step2_output,
        chunk_html=chunk_html,
        paths=paths,
        page_num=page_num,
        bundle_dir=bundle_dir,
    )
    write_text(chunk_path, chunk_html)

    _reassemble_runtime_index_html(bundle_dir=bundle_dir, title=title)

    visual_self_check: dict[str, Any]
    enabled = str(os.environ.get("PPT_STEP3_VISUAL_SELF_CHECK", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        visual_self_check = {
            "status": "skipped",
            "reason": "disabled_by_config",
            "verdict": "not_run",
            "severity": "none",
            "repair_applied": False,
        }
    elif not used_beautify_reference or not ref_s or not Path(ref_s).exists():
        visual_self_check = {
            "status": "skipped",
            "reason": "no_beautify_reference",
            "severity": "none",
            "repair_applied": False,
        }
    elif turn_dir is None:
        visual_self_check = {
            "status": "skipped",
            "reason": "no_turn_directory",
            "severity": "none",
            "repair_applied": False,
        }
    else:
        if on_visual_check is not None:
            try:
                on_visual_check()
            except Exception:
                pass
        from .step_3_reassemble.visual_self_check import run_visual_self_check

        check = run_visual_self_check(
            step3_module=s3,
            page_block=page_block,
            page_state=page_state,
            system_prompt=system_prompt,
            original_user_prompt=user_prompt,
            reference_path=Path(ref_s),
            current_html_path=chunk_path,
            turn_dir=Path(turn_dir),
            model=model,
            retry_base_seconds=max(0.5, float(retry_base_seconds)),
            retry_max_seconds=max(2.0, float(retry_max_seconds)),
            retry_backoff=max(1.0, float(retry_backoff)),
        )
        visual_self_check = check.get("visual_self_check") if isinstance(check.get("visual_self_check"), dict) else {
            "status": "error",
            "reason": "invalid_self_check_result",
            "severity": "none",
            "repair_applied": False,
        }
        if visual_self_check.get("status") in {"unavailable", "error"}:
            raise RuntimeError("step3 visual self-check unavailable")
        if (
            visual_self_check.get("verdict") == "revise"
            and visual_self_check.get("severity") == "major"
            and not visual_self_check.get("repair_applied")
        ):
            raise RuntimeError("step3 visual self-check found an unrepaired major issue")
        repaired_block = str(check.get("page_block") or page_block)
        if repaired_block and repaired_block != page_block:
            final_html = _build_chunk_html(page_block=repaired_block, base_href=base_href, title=title, s3_mod=s3)
            final_html = _publish_referenced_assets(
                step2_output=step2_output,
                chunk_html=final_html,
                paths=paths,
                page_num=page_num,
                bundle_dir=bundle_dir,
            )
            write_text(chunk_path, final_html)
            _reassemble_runtime_index_html(bundle_dir=bundle_dir, title=title)
            page_block = repaired_block

    return {
        "page_block": page_block,
        "chunk_path": str(chunk_path),
        "bundle_dir": str(bundle_dir),
        "index_html": str(bundle_dir / "index.html"),
        "prep_warnings": prep_warnings,
        "soft_warnings": debug.get("soft_warnings") if isinstance(debug, dict) else [],
        "has_layout_intent": bool(has_layout),
        "used_beautify_reference": bool(used_beautify_reference),
        "visual_self_check": visual_self_check,
    }


_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)


def _page_images_slot(page_num: int) -> str:
    """The per-page image folder name (mirrors the baseline `images/pageK/`
    convention and the chunk's `page{N-1}` element-id prefix)."""
    return f"page{int(page_num) - 1}"


def _ensure_writable_page_images_dir(*, bundle_dir: Path, slot: str) -> Path:
    """Return a REAL (writable) `bundle/images/{slot}/` directory into which the
    edited page's newly-referenced assets can be published, without touching the
    pristine baseline.

    At bootstrap `bundle/images` is a symlink to `baseline/ref_html/images`, so
    writing through it would mutate the baseline (the source of truth for
    unedited pages and re-bootstrap). We lazily de-symlink: replace the symlink
    with a real dir that re-symlinks every baseline page folder back into place,
    so unedited pages still resolve. Then materialise the edited page's own slot
    as a real dir, seeded with any baseline images that page retains, so both
    kept-baseline srcs (`images/{slot}/NN.png`) and new uploads resolve."""
    images_dir = bundle_dir / "images"
    if images_dir.is_symlink():
        baseline_images = Path(os.readlink(images_dir))
        if not baseline_images.is_absolute():
            baseline_images = (images_dir.parent / baseline_images).resolve()
        images_dir.unlink()
        images_dir.mkdir(parents=True, exist_ok=True)
        if baseline_images.is_dir():
            for child in baseline_images.iterdir():
                dst = images_dir / child.name
                if dst.exists() or dst.is_symlink():
                    continue
                try:
                    os.symlink(child, dst, target_is_directory=child.is_dir())
                except OSError:
                    if child.is_dir():
                        shutil.copytree(child, dst)
                    else:
                        shutil.copyfile(child, dst)
    images_dir.mkdir(parents=True, exist_ok=True)

    slot_dir = images_dir / slot
    if slot_dir.is_symlink():
        # Seeded above as a symlink back to baseline; promote to a real dir so we
        # can add the edited page's assets without polluting the baseline.
        baseline_slot = slot_dir.resolve()
        slot_dir.unlink()
        slot_dir.mkdir(parents=True, exist_ok=True)
        if baseline_slot.is_dir():
            for f in baseline_slot.iterdir():
                if f.is_file():
                    try:
                        shutil.copyfile(f, slot_dir / f.name)
                    except OSError:
                        continue
    else:
        slot_dir.mkdir(parents=True, exist_ok=True)
    return slot_dir


def _publish_referenced_assets(
    *, step2_output: Any, chunk_html: str, paths: WorkspacePaths, page_num: int, bundle_dir: Path | None = None
) -> str:
    """Publish image assets referenced by the freshly generated chunk into the
    turn runtime's per-page `images/{slot}/` folder and rewrite the chunk's
    bare-name `<img src>` to point there. Returns the (possibly) rewritten HTML.

    Safe relative filenames and run-isolated upload paths are handled (data:
    URIs, absolute/remote URLs, and srcs already under `images/...` are left
    untouched). Missing sources are skipped silently; required content is
    validated by the surrounding pipeline."""
    try:
        um = (step2_output or {}).get("compile", {}).get("understand_modified", {})
        bundle_src = str(um.get("bundle_dir") or "").strip()
    except Exception:  # noqa: BLE001
        bundle_src = ""
    if not bundle_src:
        return chunk_html
    src_dir = Path(bundle_src).expanduser()
    if not src_dir.is_dir():
        return chunk_html

    slot = _page_images_slot(page_num)
    slot_dir: Path | None = None

    def _rewrite(m: re.Match[str]) -> str:
        nonlocal slot_dir
        prefix, src, suffix = m.group(1), m.group(2).strip(), m.group(3)
        # Leave data URIs, remote/absolute URLs, and already-published srcs alone.
        if (
            not src
            or src.startswith("data:")
            or "://" in src
            or src.startswith("/")
            or src.startswith("images/")
        ):
            return m.group(0)
        rel = Path(src)
        if ".." in rel.parts:
            return m.group(0)
        name = rel.name
        candidates = [(src_dir / rel).resolve(), (src_dir / name).resolve()]
        if src_dir.name == "source":
            candidates.append((src_dir.parent / "uploads" / rel).resolve())
        src_file = next((candidate for candidate in candidates if candidate.is_file()), None)
        if src_file is None:
            return m.group(0)
        if slot_dir is None:
            slot_dir = _ensure_writable_page_images_dir(
                bundle_dir=bundle_dir or (paths.root / "_tmp_html_runtime"), slot=slot
            )
        dst_file = slot_dir / name
        if not dst_file.exists():
            try:
                shutil.copyfile(src_file, dst_file)
            except OSError:
                return m.group(0)
        return f"{prefix}images/{slot}/{name}{suffix}"

    return _IMG_SRC_RE.sub(_rewrite, chunk_html)


def _css_library_doc(paths: WorkspacePaths) -> str:
    p = api_script_dir() / "css_library.md"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    # Fallback: pass the CSS itself (better than nothing).
    p2 = paths.root / "_tmp_html_runtime" / "css_library.css"
    if p2.exists():
        return p2.read_text(encoding="utf-8", errors="replace")
    return ""


def _dry_run_page_block(*, page_state: dict[str, Any], page_num: int) -> str:
    pid = f"page{int(page_num) - 1}"
    body_parts: list[str] = []
    for t in page_state.get("texts") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "").strip()
        txt = str(t.get("text") or "")
        if not tid:
            continue
        body_parts.append(f'<div data-ref="{html.escape(tid)}">{html.escape(txt)}</div>')
    for im in page_state.get("images") or []:
        if not isinstance(im, dict):
            continue
        iid = str(im.get("id") or "").strip()
        src = str(im.get("src") or "")
        if not iid or not src:
            continue
        body_parts.append(f'<img data-ref="{html.escape(iid)}" src="{html.escape(src)}" />')
    return (
        "<!-- PAGE_START -->\n"
        f'<div class="page" id="{pid}">\n'
        + "\n".join(body_parts)
        + "\n</div>\n<!-- PAGE_END -->\n"
    )


_PAGE_DIV_RE = re.compile(r'<div\s+class="page"[^>]*>.*?</div>', re.DOTALL | re.IGNORECASE)


def _reassemble_runtime_index_html(*, bundle_dir: Path, title: str) -> None:
    """Build a single-page index.html inside a turn-local runtime."""
    chunks_dir = bundle_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.html"))
    blocks: list[str] = []
    for cf in chunk_files:
        txt = cf.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"<!--\s*PAGE_START\s*-->(.*?)<!--\s*PAGE_END\s*-->", txt, re.DOTALL)
        if m:
            blocks.append(m.group(1).strip())
            continue
        m2 = _PAGE_DIV_RE.search(txt)
        if m2:
            blocks.append(m2.group(0).strip())

    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh">',
            "<head>",
            '<base href="./">',
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"<title>{title}</title>",
            '<link rel="stylesheet" href="styles.css" />',
            '<link rel="stylesheet" href="css_library.css" />',
            "</head>",
            "<body>",
            "\n\n".join(blocks),
            "</body>",
            "</html>",
            "",
        ]
    )
    write_text(bundle_dir / "index.html", html)


# ---------------------------------------------------------------------------
# Step 4 (QA + autoshrink) — runs on the turn-local runtime, but we
# only care about the result for `page_num` and read its serialised
# page back out of the indexed result list.
# ---------------------------------------------------------------------------


def run_step4_qa(
    *,
    paths: WorkspacePaths,
    page_num: int,
    title: str = "PPTAgent",
    no_shrink: bool = False,
    min_font_pt: float = 8.0,
    min_scale: float = 0.5,
    max_iters: int = 8,
    min_line_height_mul: float = 1.15,
    bundle_dir: Path | None = None,
    chunk_path: Path | None = None,
) -> dict[str, Any]:
    """Run step4 against the turn-local runtime and return per-page QA outcome
    for `page_num` (and whether the chunk was rewritten)."""
    _, _, s3, s4 = _load_step_modules()

    from playwright.sync_api import sync_playwright  # local import; preview-only dep

    bundle_dir = Path(bundle_dir) if bundle_dir is not None else paths.root / "_tmp_html_runtime"
    css_path = bundle_dir / "css_library.css"
    if not css_path.exists():
        raise RuntimeError(f"css_library.css missing in {bundle_dir}")

    index_pages = s4._index_pages_in_bundle(bundle_dir)
    autoshrink_opts: dict[str, Any] | None = None
    if not no_shrink:
        autoshrink_opts = {
            "minFontPt": float(min_font_pt),
            "minScale": float(min_scale),
            "maxIters": int(max_iters),
            "minLineHeightMul": float(min_line_height_mul),
        }

    index_html = bundle_dir / "index.html"
    if not index_html.exists():
        _reassemble_runtime_index_html(bundle_dir=bundle_dir, title=title)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                pw_page = browser.new_page(viewport={"width": 1280, "height": 720})
                qa = s4._run_qa_on_index(
                    pw_page=pw_page,
                    index_html_path=index_html,
                    autoshrink_opts=autoshrink_opts,
                )
            finally:
                browser.close()
    except Exception as e:
        # Production-friendly: if Playwright browsers are missing, we still
        # return a structured result so the pipeline can proceed with step3 output.
        return {
            "skipped": True,
            "reason": "playwright_unavailable",
            "error": f"{type(e).__name__}: {e}",
            "rewrote_chunk": False,
            "before": [],
            "after": [],
            "shrink_info": {},
            "target_idx": -1,
        }

    wrote = False
    target_idx = next((i for i, (pno, _cf) in enumerate(index_pages) if int(pno) == int(page_num)), -1)

    if autoshrink_opts is not None and target_idx >= 0:
        serialised: list[str] = qa.get("serialised_pages") or []
        shrink_info = qa.get("shrink_info") or {}
        per_page = shrink_info.get("perPage") if isinstance(shrink_info.get("perPage"), list) else []
        adjusted = {
            int(x["idx"]): bool(x.get("adjusted"))
            for x in per_page
            if isinstance(x, dict) and isinstance(x.get("idx"), int)
        }
        if adjusted.get(target_idx) and len(serialised) > target_idx:
            new_outer = serialised[target_idx]
            chunk_path = Path(chunk_path) if chunk_path is not None else paths.chunk_path(page_num)
            try:
                original = chunk_path.read_text(encoding="utf-8", errors="replace")
                new_html = s4._replace_page_block_in_chunk_html(
                    chunk_html=original, new_page_div_outer=new_outer
                )
                write_text(chunk_path, new_html)
                wrote = True
            except Exception:
                wrote = False

    if wrote:
        _reassemble_runtime_index_html(bundle_dir=bundle_dir, title=title)

    return {
        "before": qa.get("before") or [],
        "after": qa.get("after") or [],
        "shrink_info": qa.get("shrink_info") or {},
        "target_idx": int(target_idx),
        "rewrote_chunk": bool(wrote),
    }
