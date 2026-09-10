"""Filesystem layout helpers for a per-project workspace.

New workspaces are PPTist-first: each page's ``state/pptist_slide.json`` is the
only persisted page source of truth. Full-pipeline HTML is a turn-local
intermediate, while baseline page PNGs and per-page assets are derived support
artifacts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# Project root is the `agent_backend/` directory itself.
# This keeps the subproject runnable when vendored or split into its own repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "result"


def _safe_project_id(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip()) or "project"
    return s


def make_project_id(source_path: Path) -> str:
    """Build a stable-ish but unique id from the upload stem + a timestamp."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_safe_project_id(source_path.stem)}_{stamp}"


@dataclass(frozen=True)
class WorkspacePaths:
    """All on-disk paths derived from a project_id."""

    project_id: str
    root: Path

    @property
    def baseline_dir(self) -> Path:
        return self.root / "baseline"

    @property
    def pages_png_dir(self) -> Path:
        return self.baseline_dir / "pages_png"

    @property
    def baseline_manifest_json(self) -> Path:
        return self.baseline_dir / "manifest.json"

    def source_original(self, suffix: str) -> Path:
        """The verbatim original PPT/PPTX upload.

        Kept so the embedded PPTist editor can natively parse the original
        presentation into editable slides. ``suffix`` includes the leading dot
        (e.g. ``.pptx``); an empty/plain suffix falls back to ``.bin``."""
        s = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ".bin"
        return self.baseline_dir / f"source_original{s}"

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    def page_dir(self, page_num: int) -> Path:
        return self.pages_dir / f"page_{int(page_num):03d}"

    def page_state_dir(self, page_num: int) -> Path:
        return self.page_dir(page_num) / "state"

    def page_assets_dir(self, page_num: int) -> Path:
        """Stable per-page asset root.

        Subdirectories:
        - ``source``: materialized image bytes from committed PPTist JSON.
        - ``uploads``: user-uploaded images staged for an edit but not consumed.
        - ``generated``: future generated assets.
        """
        return self.page_dir(page_num) / "assets"

    @property
    def session_uploads_dir(self) -> Path:
        """Landing zone for chat-dropped attachments (images) BEFORE the agent
        has decided which page they belong to.

        ``/chat`` writes each raw attachment here and injects an "available
        uploads" manifest into the agent context. When the agent knows the
        target page it calls the ``stage_page_asset`` tool, which MOVES the file
        from here into that page's stable ``page_assets_dir`` and records it in
        the page's pending-uploads manifest for ``image.add`` to consume."""
        return self.root / "_session_uploads"

    def pending_uploads_json(self, page_num: int) -> Path:
        """Per-page manifest of user uploads staged for this page but not yet
        consumed. ``stage_page_asset`` appends ``{filename, user_note}`` entries;
        the ``image.add`` skill reads them, adds each to ``state["images"]``, and
        clears the manifest so a later turn doesn't re-add them.

        Lives INSIDE ``page_assets_dir`` (the page's ``bundle_dir``) so the skill
        can locate it from ``state["bundle_dir"]`` alone — skills receive only the
        cloned state, never ``WorkspacePaths``. The leading underscore keeps it
        out of the way of the ``upload_*`` / ``reread_img_*`` image files."""
        return self.page_assets_dir(page_num) / "uploads" / "_pending_uploads.json"

    def page_understanding_json(self, page_num: int) -> Path:
        """Shared page understanding wrapper.

        ``core`` inside this file is the exact historical Step1
        ``understand_output_v1`` consumed by the Full Pipeline.
        """
        return self.page_state_dir(page_num) / "page_understanding.json"

    def pptist_slide_json(self, page_num: int) -> Path:
        """The page's authoritative PPTist slide JSON, keyed by stable slot."""
        return self.page_state_dir(page_num) / "pptist_slide.json"

    def staged_pptist_slide_json(self, page_num: int, content_hash: str) -> Path:
        """Candidate PPTist slide JSON for a manual edit that is not yet ready.

        Manual edits are promoted only after their matching frontend-rendered
        PNG reaches the backend. Keeping candidates under a hash prevents a new
        JSON payload from being paired with an old image.
        """
        safe = re.sub(r"[^A-Fa-f0-9_.-]+", "_", str(content_hash))[:128] or "unknown"
        return self.page_state_dir(page_num) / "staged" / f"pptist_slide_{safe}.json"

    def beautify_reference_png(self, page_num: int) -> Path:
        return self.page_state_dir(page_num) / "beautify_reference.png"

    def page_pending_reread_marker(self, page_num: int) -> Path:
        """Per-page JSON sidecar describing why ``current_page_state`` is stale.

        It is created only after the page's authoritative PPTist JSON and its
        matching current PNG are both ready for a future understanding refresh.
        """
        return self.page_state_dir(page_num) / "pending_reread.json"

    def reread_page_png(self, page_num: int) -> Path:
        """The page image step1 looks at during reread. Normally written by
        reread itself (chunk HTML -> PDF -> PNG); for a manually-edited (dirty)
        page the frontend renders the current slide JSON to PNG and uploads it
        here first, so the reread sees exactly what the user sees."""
        return self.page_state_dir(page_num) / "reread_page.png"

    def staged_reread_page_png(self, page_num: int, content_hash: str) -> Path:
        """Candidate reread PNG paired with a staged manual-edit slide JSON."""
        safe = re.sub(r"[^A-Fa-f0-9_.-]+", "_", str(content_hash))[:128] or "unknown"
        return self.page_state_dir(page_num) / "staged" / f"reread_page_{safe}.png"

    def turns_dir(self, page_num: int) -> Path:
        return self.page_dir(page_num) / "turns"

    def chunk_path(self, page_num: int) -> Path:
        # Fallback path for non-turn callers. Normal full-pipeline runs write
        # HTML under the current turn's html_runtime directory.
        p = f"{int(page_num):03d}"
        return self.root / "_tmp_html_runtime" / "chunks" / f"chunk_{p}_{p}.html"

    def page_png(self, page_num: int) -> Path:
        return self.pages_png_dir / f"page_{int(page_num):03d}.png"

    def project_manifest_json(self) -> Path:
        return self.root / "project.json"


def workspace_for(project_id: str, *, results_root: Path | None = None) -> WorkspacePaths:
    root = (Path(results_root).expanduser().resolve() if results_root else DEFAULT_RESULTS_ROOT) / project_id
    return WorkspacePaths(project_id=project_id, root=root)


def list_projects(*, results_root: Path | None = None) -> list[WorkspacePaths]:
    """Enumerate existing project workspaces (newest first by mtime)."""
    root = (Path(results_root).expanduser().resolve() if results_root else DEFAULT_RESULTS_ROOT)
    if not root.exists():
        return []
    ws: list[WorkspacePaths] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "project.json").exists():
            ws.append(workspace_for(child.name, results_root=root))
    ws.sort(key=lambda w: w.project_manifest_json().stat().st_mtime if w.project_manifest_json().exists() else 0, reverse=True)
    return ws


def next_turn_dir(paths: WorkspacePaths, page_num: int) -> Path:
    """Return a fresh turns/turn_K/ directory for `page_num`."""
    base = paths.turns_dir(page_num)
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(int(re.sub(r"\D", "", p.name) or 0) for p in base.glob("turn_*") if p.is_dir())
    k = (existing[-1] if existing else 0) + 1
    out = base / f"turn_{k:04d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
