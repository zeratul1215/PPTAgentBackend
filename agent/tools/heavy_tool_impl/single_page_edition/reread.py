"""Single-page reread from PPTist JSON + current page PNG."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from agent_backend.workspace.paths import WorkspacePaths, read_json, write_json
from agent_backend.workspace.assets import materialize_pptist_slide_assets, page_asset_source_dir
from .steps import run_step1

_DATA_URI_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)


def _image_ext(mime: str) -> str:
    ext = mimetypes.guess_extension(mime) or ".png"
    if ext == ".jpe":
        return ".jpg"
    return ext


def _load_slide_for_reread(paths: WorkspacePaths, page_num: int) -> dict[str, Any]:
    saved = paths.pptist_slide_json(page_num)
    if not saved.exists():
        raise RuntimeError(f"reread: slide JSON missing for page {page_num}: {saved}")
    slide = read_json(saved)
    if isinstance(slide, dict) and isinstance(slide.get("elements"), list):
        return slide
    raise RuntimeError(f"reread: invalid PPTist slide JSON for page {page_num}")


def _materialize_images(plan_page: dict[str, Any], bundle_dir: Path) -> None:
    """Mirror data-URI image sources under ``bundle_dir`` and rewrite ``src`` to
    the relative content-hash filename used by the page's stable assets."""
    imgs = plan_page.get("images")
    if not isinstance(imgs, list):
        return
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for idx, im in enumerate(imgs):
        if not isinstance(im, dict):
            continue
        src = str(im.get("src") or "")
        m = _DATA_URI_RE.match(src)
        if not m:
            continue
        mime, b64 = m.group(1), m.group(2)
        try:
            raw = base64.b64decode(b64, validate=False)
            if not raw:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            name = f"{digest[:16]}{_image_ext(mime)}"
            dst = bundle_dir / name
            if not dst.exists():
                dst.write_bytes(raw)
            im["src"] = name
        except Exception:  # noqa: BLE001
            # Leave the data URI in place; step1 will just skip this asset.
            continue


def reread_page(
    *,
    paths: WorkspacePaths,
    page_num: int,
    model: str,
    dry_run: bool,
    dpi: float = 150.0,
    keep_temp: bool = False,
    page_png_override: Path | None = None,
) -> dict[str, Any]:
    """PPTist slide JSON + page PNG → adapter → step1 → current_page_state.

    Returns the new `understand_output_v1` dict and writes it into the shared
    `page_understanding.json` wrapper for callers that still use this legacy
    reread entry point.

    ``page_png_override`` (Path A): when a page was manually edited, the HTML
    chunk no longer matches what the user sees, so rendering it would give step1
    a stale image. The frontend instead uploads a PNG rendered from the current
    slide JSON; when that file is passed here it is used verbatim as the page
    image and the chunk render is skipped.
    """
    from agent_backend.agent.tools.html_to_pptist import slide_to_plan_page

    page_idx0 = int(page_num) - 1

    tmp_root = Path(tempfile.mkdtemp(prefix=f"reread_p{int(page_num):03d}_"))
    try:
        # Persist the page image step1 looks at under the workspace so downstream
        # steps (step2 beautify multimodal context, step3 page_png reference) can
        # load it too.
        stable_png = paths.reread_page_png(page_num)
        stable_png.parent.mkdir(parents=True, exist_ok=True)

        if page_png_override is not None and Path(page_png_override).exists():
            if Path(page_png_override).resolve() != stable_png.resolve():
                shutil.copyfile(page_png_override, stable_png)
        elif not stable_png.exists():
            baseline = paths.page_png(page_num)
            if baseline.exists():
                shutil.copyfile(baseline, stable_png)
            else:
                raise RuntimeError(f"reread: no current or baseline PNG for page {page_num}")

        # 2) Structure: from the page's current PPTist slide (synced edits win),
        # extracted by the adapter into a step1 plan_page.
        slide = _load_slide_for_reread(paths, page_num)
        new_plan_page = slide_to_plan_page(slide, page_id=f"page{page_idx0}")

        # Image srcs on the slide are self-contained data URIs; write them to
        # the page's STABLE asset dir (its bundle_dir) so step1 can read the
        # bytes AND so the dir survives past this reread — the next turn's image
        # skills resolve `state["bundle_dir"]` against it.
        materialize_pptist_slide_assets(paths, page_num, slide)
        asset_dir = page_asset_source_dir(paths, page_num)
        _materialize_images(new_plan_page, asset_dir)

        # 3) page_size_pt: carry forward from the page's previous understanding
        # (reread does not change the page's physical dimensions).
        page_size_pt: dict[str, Any] = {"w": None, "h": None}
        prev_path = paths.page_understanding_json(page_num)
        if prev_path.exists():
            try:
                prev_wrapper = read_json(prev_path)
                prev_state = prev_wrapper.get("core") if isinstance(prev_wrapper, dict) else None
            except Exception:  # noqa: BLE001
                prev_state = None
            if isinstance(prev_state, dict):
                ps = prev_state.get("page_size_pt")
                if isinstance(ps, dict):
                    page_size_pt = {"w": ps.get("w"), "h": ps.get("h")}

        understand_input = {
            "schema_version": "understand_input_v1",
            "page_num": int(page_num),
            "page_size_pt": page_size_pt,
            "bundle_dir": str(asset_dir),
            "plan_page": new_plan_page,
            "page_png_path": str(stable_png),
            "options": {
                "need_image_descriptions": True,
                "need_original_layout_description": True,
            },
        }

        new_understand_output = run_step1(
            understand_input=understand_input,
            model=model,
            dry_run=dry_run,
        )
        # Keep page identifiers meaningful relative to the project.
        new_understand_output["page_num"] = int(page_num)
        new_understand_output["page_id"] = f"page{int(page_num) - 1}"

        write_json(
            paths.page_understanding_json(page_num),
            {
                "schema_version": "page_understanding_v2",
                "core": new_understand_output,
                "common": {},
                "focused": {"entries": []},
            },
        )
        return new_understand_output
    finally:
        if not keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
