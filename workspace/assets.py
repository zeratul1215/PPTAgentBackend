"""PPTist-first per-page asset materialization.

Committed ``pptist_slide.json`` stays self-contained for the frontend, but the
agent stack also needs real files for Step1 image descriptions, image skills,
HTML generation, and HTML->PPTist conversion. This module mirrors data-URI
images from a slide into ``pages/page_XXX/assets/source`` using content hashes
and writes a small manifest.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any

from .paths import WorkspacePaths, read_json, write_json


DATA_URI_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)


def page_asset_source_dir(paths: WorkspacePaths, slot: int) -> Path:
    return paths.page_assets_dir(int(slot)) / "source"


def page_asset_uploads_dir(paths: WorkspacePaths, slot: int) -> Path:
    return paths.page_assets_dir(int(slot)) / "uploads"


def page_asset_generated_dir(paths: WorkspacePaths, slot: int) -> Path:
    return paths.page_assets_dir(int(slot)) / "generated"


def page_asset_manifest(paths: WorkspacePaths, slot: int) -> Path:
    return paths.page_assets_dir(int(slot)) / "manifest.json"


def _image_ext(mime: str) -> str:
    ext = mimetypes.guess_extension(mime) or ".png"
    if ext == ".jpe":
        return ".jpg"
    return ext


def _iter_elements(slide: dict[str, Any]) -> list[dict[str, Any]]:
    elements = slide.get("elements")
    if not isinstance(elements, list):
        return []
    return [e for e in elements if isinstance(e, dict)]


def materialize_pptist_slide_assets(
    paths: WorkspacePaths,
    slot: int,
    slide: dict[str, Any],
) -> dict[str, Any]:
    """Mirror data-URI images from ``slide`` into ``assets/source``.

    The slide is not mutated. The manifest maps PPTist element ids to the file
    path produced for their current image source.
    """
    source_dir = page_asset_source_dir(paths, int(slot))
    uploads_dir = page_asset_uploads_dir(paths, int(slot))
    generated_dir = page_asset_generated_dir(paths, int(slot))
    source_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    for idx, el in enumerate(_iter_elements(slide)):
        src = el.get("src")
        if not isinstance(src, str):
            continue
        m = DATA_URI_RE.match(src)
        if not m:
            continue
        mime, b64 = m.group(1), m.group(2)
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            continue
        if not raw:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        name = f"{digest[:16]}{_image_ext(mime)}"
        dst = source_dir / name
        if not dst.exists():
            try:
                dst.write_bytes(raw)
            except OSError:
                continue
        images.append(
            {
                "element_id": el.get("id") or f"image_{idx}",
                "sha256": digest,
                "mime": mime,
                "filename": name,
                "path": str(dst),
                "source": "pptist_json",
            }
        )

    manifest = {
        "schema_version": "page_assets_v1",
        "slot": int(slot),
        "source_dir": str(source_dir),
        "uploads_dir": str(uploads_dir),
        "generated_dir": str(generated_dir),
        "images": images,
    }
    write_json(page_asset_manifest(paths, int(slot)), manifest)
    return manifest


def clear_staged_candidates(paths: WorkspacePaths, slot: int) -> None:
    staged = paths.page_state_dir(int(slot)) / "staged"
    if not staged.exists():
        return
    for child in staged.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
        except OSError:
            continue


def consume_pending_uploads_for_run(
    paths: WorkspacePaths,
    slot: int,
    run_id: str,
) -> None:
    """Remove only one successfully committed run's pending upload records."""
    run_id = str(run_id or "").strip()
    if not run_id:
        return
    manifest = paths.pending_uploads_json(int(slot))
    if manifest.exists():
        try:
            payload = read_json(manifest)
            uploads = payload.get("uploads") if isinstance(payload, dict) else []
            remaining = [
                item
                for item in (uploads if isinstance(uploads, list) else [])
                if not (
                    isinstance(item, dict)
                    and str(item.get("run_id") or "") == run_id
                )
            ]
            write_json(
                manifest,
                {"schema_version": "pending_uploads_v1", "uploads": remaining},
            )
        except Exception:
            return
    shutil.rmtree(page_asset_uploads_dir(paths, int(slot)) / run_id, ignore_errors=True)
