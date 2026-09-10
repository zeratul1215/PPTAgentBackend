"""Skill: image.delete — remove picture(s) from the page's content list.

The user names which picture to drop ("删掉右上角的 logo / remove the team
photo"). The executor semantically matches that against each image's
``description_en`` (mirrors the text skills' locate step), removes the matched
entries from ``state["images"]``, and deletes the corresponding files from the
page's bundle dir (executor deletes directly; no rollback — accepted risk).

Removing an element changes the element set → ``triggered_visual=True`` so the
page re-flows and no hole is left where the picture was.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Skill,
    SkillResult,
    _bundle_dir_from_state,
    _delete_bundle_file,
    _match_image_ids,
)


def _repair_image_delete_params(params: dict[str, Any]) -> list[str]:
    # Objective-driven; no structured params.
    return []


def _run_image_delete(
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
        warnings.append(f"image_delete_no_images[{iid}]")
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
        warnings.append(f"image_delete_matched_nothing[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    bundle_dir = _bundle_dir_from_state(state)
    target_set = set(target_ids)
    survivors: list[dict[str, Any]] = []
    removed = 0
    for im in images:
        if not isinstance(im, dict):
            survivors.append(im)
            continue
        if str(im.get("id") or "") in target_set:
            src = str(im.get("src") or "")
            if bundle_dir is not None and _delete_bundle_file(bundle_dir, src):
                warnings.append(f"image_delete_removed_file[{iid}]: {src}")
            removed += 1
        else:
            survivors.append(im)

    if removed == 0:
        warnings.append(f"image_delete_removed_nothing[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    state["images"] = survivors
    warnings.append(f"image_delete_applied[{iid}]: {removed} image(s)")
    return SkillResult(warnings=warnings, triggered_visual=True)


_IMAGE_DELETE_PLAN_DOC = """  Remove picture(s) from the page. Use for "删掉那张图/remove the logo/去掉这张照片".
  `objective` (natural language) MUST name WHICH image to remove (by content or
  role, e.g. "the logo in the top-right", "the chart"). No `params` — the
  executor matches the description to the page's images. Removes elements, so it
  triggers a visual re-layout. If it can't tell which image is meant it deletes
  nothing (safe)."""


_IMAGE_DELETE_ORDERING_NOTE = (
    "Operates only on the image list; independent of text/table edits. Runs in "
    "the image phase (after text/table)."
)


SKILL = Skill(
    id="image.delete",
    canonical_rank=41,
    summary="remove image(s) from the page (triggers re-layout)",
    plan_doc=_IMAGE_DELETE_PLAN_DOC,
    repair=_repair_image_delete_params,
    execute=_run_image_delete,
    ordering_note=_IMAGE_DELETE_ORDERING_NOTE,
    phase="image",
)
