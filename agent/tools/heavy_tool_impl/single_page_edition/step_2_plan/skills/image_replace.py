"""Skill: image.replace — swap page picture(s) for user-uploaded replacement(s).

The user names which picture to replace ("把这张团队照换成新的") and drops the
replacement image in chat; the external ``stage_page_asset`` tool has already
staged the new file into the page bundle + pending-uploads manifest.

The executor:
1. semantically matches the OLD image(s) to replace (via ``description_en``),
2. takes the staged replacement file(s) from the pending manifest,
3. pairs them in order (target[i] ← upload[i]),
4. for each pair: points the existing entry's ``src`` at the new file and
   regenerates ``description_en``, but KEEPS the id and the old
   ``display_w_pt/h_pt`` (position + footprint unchanged; deformation is
   accepted),
5. leaves old assets and the pending manifest untouched until commit succeeds.

Because the element set and every coordinate/footprint stay identical, replace
does NOT trigger a re-layout (``triggered_visual=False``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    Skill,
    SkillResult,
    _bundle_dir_from_state,
    _gen_image_description_en,
    _match_image_ids,
    _read_pending_uploads,
    _resolve_pending_upload_file,
)


def _repair_image_replace_params(params: dict[str, Any]) -> list[str]:
    return []


def _run_image_replace(
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

    images = state.get("images")
    if not isinstance(images, list) or not images:
        warnings.append(f"image_replace_no_images[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    bundle_dir = _bundle_dir_from_state(state)
    if bundle_dir is None:
        warnings.append(f"image_replace_no_bundle_dir[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    uploads = _read_pending_uploads(bundle_dir)
    if not uploads:
        warnings.append(f"image_replace_no_pending_uploads[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    target_ids, w = _match_image_ids(
        intent=intent,
        state=state,
        model=model,
        user_request=user_request,
        dry_run=dry_run,
    )
    warnings.extend(f"[{iid}] {m}" for m in w)
    if not target_ids:
        warnings.append(f"image_replace_matched_nothing[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    by_id: dict[str, dict[str, Any]] = {
        str(im.get("id") or ""): im for im in images if isinstance(im, dict)
    }

    # Replacement is atomic: do not consume only part of the staged set.
    if len(target_ids) != len(uploads):
        warnings.append(
            f"image_replace_count_mismatch[{iid}]: targets={len(target_ids)} "
            f"uploads={len(uploads)}"
        )
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)
    prepared: list[tuple[dict[str, Any], dict[str, Any], str, Path, bytes]] = []
    for index, target_id in enumerate(target_ids):
        old = by_id.get(target_id)
        if not isinstance(old, dict):
            return SkillResult(
                warnings=warnings + [f"image_replace_target_missing[{iid}]: {target_id}"],
                status="failed",
                triggered_visual=False,
            )
        up = uploads[index]
        new_name = str(up.get("filename") or "").strip()
        if not new_name:
            return SkillResult(
                warnings=warnings + [f"image_replace_filename_missing[{iid}]: {index}"],
                status="failed",
                triggered_visual=False,
            )
        new_path = _resolve_pending_upload_file(bundle_dir, new_name)
        if not new_path.is_file():
            return SkillResult(
                warnings=warnings + [f"image_replace_new_file_missing[{iid}]: {new_name}"],
                status="failed",
                triggered_visual=False,
            )
        try:
            image_bytes = new_path.read_bytes()
        except OSError as e:
            return SkillResult(
                warnings=warnings + [f"image_replace_read_error[{iid}]: {new_name}: {e}"],
                status="failed",
                triggered_visual=False,
            )
        prepared.append((old, up, new_name, new_path, image_bytes))

    replaced = 0
    for old, up, new_name, new_path, image_bytes in prepared:
        desc, dw = _gen_image_description_en(
            image_bytes=image_bytes,
            model=model,
            user_note=str(up.get("user_note") or ""),
            dry_run=dry_run,
        )
        warnings.extend(f"[{iid}] {m}" for m in dw)

        try:
            old["src"] = new_path.resolve().relative_to(Path(bundle_dir).resolve()).as_posix()
        except (OSError, ValueError):
            old["src"] = Path(new_name).name
        old["description_en"] = desc
        # id, display_w_pt/h_pt intentionally preserved (footprint unchanged;
        # deformation of the new image is accepted per decision).

        replaced += 1

    if replaced == 0:
        warnings.append(f"image_replace_replaced_nothing[{iid}]")
        return SkillResult(warnings=warnings, status="failed", triggered_visual=False)

    warnings.append(f"image_replace_applied[{iid}]: {replaced} image(s)")
    # Footprint + element set unchanged → no re-layout needed.
    return SkillResult(warnings=warnings, status="applied", triggered_visual=False)


_IMAGE_REPLACE_PLAN_DOC = """  Capability: replace existing pictures with staged
  user uploads while retaining each target's current position and size. The
  natural-language objective must identify the target image by visible content or
  semantic role. Ambiguous targets are left unchanged. This capability does not
  change page geometry."""


SKILL = Skill(
    id="image.replace",
    canonical_rank=42,
    summary="replace image(s) with uploaded one(s), same slot/size (no re-layout)",
    plan_doc=_IMAGE_REPLACE_PLAN_DOC,
    repair=_repair_image_replace_params,
    execute=_run_image_replace,
    ordering_note="Independent of text and table content unless the requested final result creates an explicit dependency.",
    phase="image",
)
