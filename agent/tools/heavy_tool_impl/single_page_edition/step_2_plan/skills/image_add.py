"""Skill: image.add — add user-uploaded picture(s) to the page's content list.

Scope (this phase): ONLY user-uploaded images (chat-dropped / manually placed);
no AI image generation. The external ``stage_page_asset`` tool has staged the
upload(s) for this page and recorded them in the pending-uploads manifest before
the pipeline runs.

The executor therefore does not move files or need a page number. It:
1. reads the pending-uploads manifest for this page,
2. for each staged file, generates an English ``description_en`` (vision model,
   steered by the user's optional note),
3. reads the file's pixel aspect ratio into ``display_w_pt/h_pt`` as a SOFT
   reference (portrait vs landscape) — downstream re-layout rescales freely,
4. appends ``{id: add_i{n}, src, description_en, display_w_pt/h_pt}`` to
   ``state["images"]``,
5. leaves the pending manifest untouched until the page commit succeeds.

Adding an element changes the element set → ``triggered_visual=True`` so the
downstream visual path re-flows the page and places the new picture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    Skill,
    SkillResult,
    _bundle_dir_from_state,
    _gen_image_description_en,
    _next_add_image_id,
    _read_image_pixels,
    _read_pending_uploads,
    _resolve_pending_upload_file,
)


def _repair_image_add_params(params: dict[str, Any]) -> list[str]:
    # Fully objective/upload-driven: no structured params to repair.
    return []


def _run_image_add(
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

    bundle_dir = _bundle_dir_from_state(state)
    if bundle_dir is None:
        warnings.append(f"image_add_no_bundle_dir[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    pending = _read_pending_uploads(bundle_dir)
    if not pending:
        warnings.append(f"image_add_no_pending_uploads[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    prepared: list[tuple[dict[str, Any], str, Path, bytes]] = []
    for up in pending:
        filename = str(up.get("filename") or "").strip()
        if not filename:
            return SkillResult(
                warnings=warnings + [f"image_add_filename_missing[{iid}]"],
                status="failed",
                triggered_visual=False,
            )
        fpath = _resolve_pending_upload_file(bundle_dir, filename)
        if not fpath.is_file():
            return SkillResult(
                warnings=warnings + [f"image_add_file_missing[{iid}]: {filename}"],
                status="failed",
                triggered_visual=False,
            )

        try:
            image_bytes = fpath.read_bytes()
        except OSError as e:
            return SkillResult(
                warnings=warnings + [f"image_add_read_error[{iid}]: {filename}: {e}"],
                status="failed",
                triggered_visual=False,
            )
        prepared.append((up, filename, fpath, image_bytes))

    images = state.get("images")
    if not isinstance(images, list):
        images = []
        state["images"] = images

    added = 0
    for up, filename, fpath, image_bytes in prepared:
        user_note = str(up.get("user_note") or "").strip() or objective

        desc, w = _gen_image_description_en(
            image_bytes=image_bytes,
            model=model,
            user_note=user_note,
            dry_run=dry_run,
        )
        warnings.extend(f"[{iid}] {m}" for m in w)

        entry: dict[str, Any] = {
            "id": _next_add_image_id(state),
            "src": _stable_src(bundle_dir, fpath, filename),
            "description_en": desc,
        }
        px = _read_image_pixels(fpath)
        if px is not None:
            # Store raw px as a SOFT aspect-ratio reference (no DPI→pt conv).
            entry["display_w_pt"] = float(px[0])
            entry["display_h_pt"] = float(px[1])
        images.append(entry)
        added += 1

    if added == 0:
        warnings.append(f"image_add_added_nothing[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    warnings.append(f"image_add_applied[{iid}]: {added} image(s)")
    return SkillResult(warnings=warnings, status="applied", triggered_visual=True)


def _stable_src(bundle_dir: Any, resolved_path: Path, fallback: str) -> str:
    """Return the content-list path relative to the stable source bundle."""
    try:
        return resolved_path.resolve().relative_to(Path(bundle_dir).resolve()).as_posix()
    except (OSError, ValueError):
        return Path(fallback).name


_IMAGE_ADD_PLAN_DOC = """  Capability: register staged user-uploaded pictures as
  new page content. The natural-language objective may state each image's role or
  intended use. It accepts only actual staged uploads and adding images requires
  visual re-layout."""


SKILL = Skill(
    id="image.add",
    canonical_rank=40,
    summary="add user-uploaded image(s) to the page (triggers re-layout)",
    plan_doc=_IMAGE_ADD_PLAN_DOC,
    repair=_repair_image_add_params,
    execute=_run_image_add,
    ordering_note="Independent of text and table content unless the requested final composition creates an explicit dependency.",
    phase="image",
)
