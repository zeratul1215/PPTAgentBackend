"""Tools for staging Session Resources into a target page.

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
    page_lock,
    require_agent_run_id,
    require_project_id,
    require_session_id,
    require_user_id,
    workspace_for,
)
from agent_backend.workspace import repo
from agent_backend.workspace import pageorder
from agent_backend.workspace.paths import read_json, write_json
from agent_backend.workspace.session_resources import resource_path


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


def _message_context(runtime: ToolRuntime, session_id: str, user_id: str) -> tuple[str | None, int | None]:
    context = runtime.context or {}
    message_id = context.get("current_message_id") if isinstance(context, dict) else getattr(context, "current_message_id", None)
    message_id = str(message_id or "") or None
    return message_id, repo.get_message_seq(message_id=message_id or "", session_id=session_id, user_id=user_id)


def _append_pending_resource(paths, slot: int, entry: dict[str, Any]) -> None:
    p = paths.pending_resources_json(slot)
    existing: list[dict[str, Any]] = []
    if p.exists():
        try:
            data = read_json(p)
            if isinstance(data, dict) and isinstance(data.get("resources"), list):
                existing = [it for it in data["resources"] if isinstance(it, dict)]
        except Exception:
            pass
    existing.append(entry)
    write_json(p, {"schema_version": "pending_resources_v1", "resources": existing})


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@tool
def stage_page_resource(
    page_ref: str,
    resource_ref: str,
    runtime: ToolRuntime,
    user_note: str = "",
) -> dict[str, Any]:
    """Stage an immutable Session Resource for a later operation on one page.

    `page_ref` is the stable page ref returned by outline/locate.
    `resource_ref` identifies a resource in the current session.
    `user_note` is the user's own words about the image.

    This copies/links the file into the page's asset bundle and records it as
    pending for a subsequent page operation in the same run.
    """
    pid = require_project_id(runtime)
    sid = require_session_id(runtime)
    uid = require_user_id(runtime)
    run_id = require_agent_run_id(runtime)
    message_id, message_seq = _message_context(runtime, sid, uid)
    paths = workspace_for(pid)
    ref = str(resource_ref or "").strip()
    if not ref:
        return {"project_id": pid, "ok": False, "error": "resource_ref is required"}
    resource = repo.get_session_resource(resource_ref=ref, session_id=sid, user_id=uid)
    if not resource or resource.get("status") == "deleted":
        return {"project_id": pid, "ok": False, "error": "resource_not_found"}

    slot = pageorder.slot_for_page_ref(paths, str(page_ref))
    if slot is None:
        return {"project_id": pid, "ok": False, "error": "page_not_found"}
    with page_lock(pid, int(slot)):
        if pageorder.entry_for_slot(paths, int(slot)) is None:
            return {"project_id": pid, "ok": False, "error": "page_not_found"}
        page = pageorder.position_for_slot(paths, int(slot))

        dst_dir = paths.page_assets_dir(int(slot)) / "uploads" / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        staged: list[str] = []
        try:
            for item in resource.get("files") or []:
                role = str(item.get("role") or "original")
                source = resource_path(uid, sid, ref, str(item.get("relative_path") or ""))
                if not source.is_file():
                    continue
                fname = f"{ref}_{role}{source.suffix or '.bin'}"
                _copy_or_link(source, dst_dir / fname)
                staged.append(fname)
        except OSError as e:
            return {"project_id": pid, "ok": False, "error": f"stage failed: {e}"}
        if not staged:
            return {"project_id": pid, "ok": False, "error": "resource_file_missing"}

        original = next((f for f in resource.get("files") or [] if f.get("role") == "original"), None)
        original_stage = next((name for name in staged if name.startswith(f"{ref}_original")), None)
        if original and original_stage:
            original_name = Path(str(original.get("relative_path") or original_stage)).name
            _append_pending(paths, int(slot), {
                "filename": f"{run_id}/{original_stage}",
                "user_note": str(user_note or "").strip(),
                "resource_ref": ref,
                "artifact_ref": ref,
                "run_id": run_id,
                "original_filename": original_name,
                "resource_kind": resource.get("kind"),
                "status": "pending",
            })
        else:
            _append_pending_resource(paths, int(slot), {
                "resource_ref": ref,
                "run_id": run_id,
                "user_note": str(user_note or "").strip(),
                "resource_kind": resource.get("kind"),
                "files": [f"{run_id}/{name}" for name in staged],
                "status": "pending",
            })
        repo.touch_session_resources(
            resource_refs=[ref], session_id=sid, user_id=uid,
            relation="staged", agent_run_id=run_id, message_id=message_id, message_seq=message_seq,
            details={"target_project_id": pid, "target_page_ref": pageorder.page_ref_for_slot(int(slot)), "user_note": str(user_note or "").strip()},
        )

    return {
        "project_id": pid,
        "ok": True,
        "page": int(page or 0),
        "page_ref": pageorder.page_ref_for_slot(int(slot)),
        "resource_ref": ref,
        "artifact_ref": ref,
        "filename": f"{run_id}/{original_stage}" if original_stage else None,
        "kind": resource.get("kind"),
    }


stage_page_asset = stage_page_resource

__all__ = ["stage_page_resource", "stage_page_asset"]
