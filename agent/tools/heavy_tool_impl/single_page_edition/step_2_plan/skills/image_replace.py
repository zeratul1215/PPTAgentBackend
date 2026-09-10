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
   accepted), then deletes the OLD file from the bundle,
5. clears the manifest.

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
    _clear_pending_uploads,
    _delete_bundle_file,
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
        return SkillResult(warnings=warnings, triggered_visual=False)

    bundle_dir = _bundle_dir_from_state(state)
    if bundle_dir is None:
        warnings.append(f"image_replace_no_bundle_dir[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    uploads = _read_pending_uploads(bundle_dir)
    if not uploads:
        warnings.append(f"image_replace_no_pending_uploads[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

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
        return SkillResult(warnings=warnings, triggered_visual=False)

    by_id: dict[str, dict[str, Any]] = {
        str(im.get("id") or ""): im for im in images if isinstance(im, dict)
    }

    # Pair each matched target with a staged upload, in order. Extra targets or
    # extra uploads (count mismatch) are surfaced and left untouched.
    pairs = min(len(target_ids), len(uploads))
    if len(target_ids) != len(uploads):
        warnings.append(
            f"image_replace_count_mismatch[{iid}]: targets={len(target_ids)} "
            f"uploads={len(uploads)} (using {pairs})"
        )

    replaced = 0
    for i in range(pairs):
        old = by_id.get(target_ids[i])
        if not isinstance(old, dict):
            continue
        up = uploads[i]
        new_name = str(up.get("filename") or "").strip()
        if not new_name:
            continue
        new_path = _resolve_pending_upload_file(bundle_dir, new_name)
        if not new_path.is_file():
            warnings.append(f"image_replace_new_file_missing[{iid}]: {new_name}")
            continue

        try:
            image_bytes = new_path.read_bytes()
        except OSError as e:
            warnings.append(f"image_replace_read_error[{iid}]: {new_name}: {e}")
            image_bytes = b""

        desc, dw = _gen_image_description_en(
            image_bytes=image_bytes,
            model=model,
            user_note=str(up.get("user_note") or ""),
            dry_run=dry_run,
        )
        warnings.extend(f"[{iid}] {m}" for m in dw)

        old_src = str(old.get("src") or "")
        try:
            old["src"] = new_path.resolve().relative_to(Path(bundle_dir).resolve()).as_posix()
        except (OSError, ValueError):
            old["src"] = Path(new_name).name
        old["description_en"] = desc
        # id, display_w_pt/h_pt intentionally preserved (footprint unchanged;
        # deformation of the new image is accepted per decision).

        # Delete the old file only if it isn't the same name as the new one.
        if old_src and Path(old_src).name != Path(str(old.get("src") or "")).name:
            if _delete_bundle_file(bundle_dir, old_src):
                warnings.append(f"image_replace_removed_old_file[{iid}]: {old_src}")
        replaced += 1

    if replaced == 0:
        warnings.append(f"image_replace_replaced_nothing[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    _clear_pending_uploads(bundle_dir)
    warnings.append(f"image_replace_applied[{iid}]: {replaced} image(s)")
    # Footprint + element set unchanged → no re-layout needed.
    return SkillResult(warnings=warnings, triggered_visual=False)


_IMAGE_REPLACE_PLAN_DOC = """  Replace existing picture(s) with user-uploaded image(s), keeping the SAME
  position and size. Use for "把这张图换成我新传的这张/replace the photo with this".
  The replacement upload is already staged into the page. `objective` (natural
  language) MUST name WHICH image to replace (by content/role). No `params` — the
  executor matches the old image by description and pairs it with the staged
  upload. Keeps the old element's slot/size (deformation accepted), so it does
  NOT trigger a visual re-layout. If it can't tell which image is meant it
  replaces nothing (safe)."""


_IMAGE_REPLACE_ORDERING_NOTE = (
    "Operates only on the image list and never changes geometry; independent of "
    "text/table edits. Runs in the image phase (after text/table)."
)


SKILL = Skill(
    id="image.replace",
    canonical_rank=42,
    summary="replace image(s) with uploaded one(s), same slot/size (no re-layout)",
    plan_doc=_IMAGE_REPLACE_PLAN_DOC,
    repair=_repair_image_replace_params,
    execute=_run_image_replace,
    ordering_note=_IMAGE_REPLACE_ORDERING_NOTE,
    phase="image",
)
