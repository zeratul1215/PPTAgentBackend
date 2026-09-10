"""Tool: stage_page_asset — bind a chat Artifact image to a target page.

The user uploads images through the Artifact API before a chat run starts. Once
the agent knows WHICH page the picture belongs to, this tool copies or hardlinks
the Artifact into that page's run-isolated upload area and appends a pending
manifest entry for image.add / image.replace.

The ``image.add`` / ``image.replace`` skills then find the file already in place
(the "file already staged" precondition) and consume the manifest. This keeps
the skills free of page numbers and file plumbing — they only edit the content
list. Page numbers here are 1-based display positions, resolved to a stable slot
so add/delete/reorder stay consistent.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from agent_backend.agent.tools.context import (
    project_lock,
    require_agent_run_id,
    require_project_id,
    require_session_id,
    require_user_id,
    workspace_for,
)
from agent_backend.workspace import repo
from agent_backend.workspace import pageorder
from agent_backend.workspace.paths import read_json, write_json


def _uniquify(dst_dir: Path, name: str) -> str:
    """Return a filename under ``dst_dir`` that doesn't clobber an existing one,
    appending ``_1``, ``_2``, ... to the stem on collision."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = name
    i = 0
    while (dst_dir / candidate).exists():
        i += 1
        candidate = f"{stem}_{i}{suffix}"
    return candidate


def _append_pending(paths, slot: int, entry: dict[str, str]) -> None:
    """Append one {filename, user_note} to the page's pending-uploads manifest."""
    p = paths.pending_uploads_json(slot)
    existing: list[dict[str, Any]] = []
    if p.exists():
        try:
            data = read_json(p)
            items = data.get("uploads") if isinstance(data, dict) else data
            if isinstance(items, list):
                existing = [it for it in items if isinstance(it, dict)]
        except Exception:
            existing = []
    existing.append(entry)
    write_json(p, {"schema_version": "pending_uploads_v1", "uploads": existing})


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@tool
def stage_page_asset(
    page_ref: str,
    artifact_ref: str,
    runtime: ToolRuntime,
    user_note: str = "",
) -> dict[str, Any]:
    """Place a chat-uploaded image onto a page, ready for image.add/replace.

    Call this BEFORE editing a page when the user attached an image they want on
    that page. `page_ref` is the stable page ref returned by outline/locate.
    `artifact_ref` is one of the current message's uploaded Artifact refs.
    `user_note` is the user's own words about the image.

    This copies/links the file into the page's asset bundle and records it as pending.
    Follow up on the SAME page: use `patch_pages(images_involved=True)` for an
    in-place replacement, `fill_empty_pages` for a blank page, or `edit_pages`
    when a nonblank page's composition must change.
    """
    pid = require_project_id(runtime)
    sid = require_session_id(runtime)
    uid = require_user_id(runtime)
    run_id = require_agent_run_id(runtime)
    paths = workspace_for(pid)
    lock = project_lock(pid)

    ref = str(artifact_ref or "").strip()
    if not ref:
        return {"project_id": pid, "ok": False, "error": "artifact_ref is required"}
    artifact = repo.get_artifact(artifact_ref=ref, session_id=sid, user_id=uid)
    if not artifact or artifact.get("status") == "deleted":
        return {"project_id": pid, "ok": False, "error": "artifact_not_found"}
    src = Path(str(artifact.get("storage_path") or ""))
    if not src.is_file():
        return {"project_id": pid, "ok": False, "error": "artifact_file_missing"}

    with lock:
        slot = pageorder.slot_for_page_ref(paths, str(page_ref))
        if slot is None:
            return {"project_id": pid, "ok": False, "error": "page_not_found"}
        page = pageorder.position_for_slot(paths, int(slot))

        sha = str(artifact.get("sha256") or "")
        suffix = Path(str(artifact.get("filename") or src.name)).suffix or src.suffix or ".png"
        fname = f"{sha[:16] or ref}{suffix}"
        rel_name = f"{run_id}/{fname}"
        dst_dir = paths.page_assets_dir(int(slot)) / "uploads" / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            _copy_or_link(src, dst_dir / fname)
        except OSError as e:
            return {"project_id": pid, "ok": False, "error": f"stage failed: {e}"}

        _append_pending(
            paths,
            int(slot),
            {
                "filename": rel_name,
                "user_note": str(user_note or "").strip(),
                "artifact_ref": ref,
                "run_id": run_id,
                "original_filename": str(artifact.get("filename") or ""),
                "status": "pending",
            },
        )

    return {
        "project_id": pid,
        "ok": True,
        "page": int(page or 0),
        "page_ref": pageorder.page_ref_for_slot(int(slot)),
        "artifact_ref": ref,
        "filename": rel_name,
    }


__all__ = ["stage_page_asset"]
