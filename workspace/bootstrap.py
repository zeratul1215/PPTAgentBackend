"""Workspace bootstrap for PPTist-first projects.

Upload bootstrap only creates deterministic visual baseline artifacts. PPTist
slide JSON arrives later from the frontend's native parser via the project
initialization endpoint; that JSON becomes the sole page source of truth.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .paths import WorkspacePaths, workspace_for, write_json
from ._imports import project_root
from agent_backend.agent.tools.heavy_tool_impl._shared import api_script_dir


__all__ = ["bootstrap_workspace_from_upload"]


# ---------------------------------------------------------------------------
# Step 0 (deterministic) — render baseline PNGs only.
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def _bootstrap_workspace_from_pdf_render(
    *,
    pdf_path: Path,
    project_id: str,
    results_root: Path | None = None,
    dpi: float = 150.0,
    need_image_descriptions: bool = True,
    need_original_layout_description: bool = True,
    title: str = "PPTAgent",
    user_id: str | None = None,
    source_kind: str | None = None,
    original_upload: Path | None = None,
) -> WorkspacePaths:
    """Materialise a fresh workspace from a temporary PPT/PPTX PDF render."""
    paths = workspace_for(project_id, results_root=results_root)
    if paths.root.exists():
        raise FileExistsError(f"workspace already exists: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=False)
    paths.baseline_dir.mkdir(parents=True, exist_ok=True)
    paths.pages_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    script_dir = api_script_dir()

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Keep the verbatim PPT upload only until frontend PPTist parsing and backend
    # initialization finish. The initialization endpoint deletes it afterwards.
    source_original_name: str | None = None
    if original_upload is not None:
        original_upload = original_upload.expanduser().resolve()
        if original_upload.exists():
            dst = paths.source_original(original_upload.suffix.lower())
            shutil.copyfile(original_upload, dst)
            source_original_name = dst.name

    # 1) Page PNGs.
    _run([py, str(script_dir / "render_pages_png.py"), str(pdf_path), "--out", str(paths.pages_png_dir), "--dpi", str(float(dpi))])
    n_pages = len(sorted(paths.pages_png_dir.glob("page_*.png")))
    if n_pages <= 0:
        raise RuntimeError("no page PNGs produced from uploaded deck")
    page_size_pt = {"w": None, "h": None}

    # 2) Manifests. The project is not AI-editable until PPTist JSON and initial
    # Step1 have been written by the initialize endpoint.
    write_json(
        paths.baseline_manifest_json,
        {
            "pages_png_dir": str(paths.pages_png_dir),
            "page_count": int(n_pages),
            "page_size_pt": page_size_pt,
        },
    )
    write_json(
        paths.project_manifest_json(),
        {
            "project_id": paths.project_id,
            "user_id": user_id,
            "page_count": int(n_pages),
            "page_size_pt": page_size_pt,
            "results_root": str(paths.root.parent),
            "project_root": str(project_root()),
            "title": title,
            "source_kind": source_kind,
            "source_original": source_original_name,
            "status": "pending_pptist_init",
        },
    )

    # 3) Ordered page list. Initial slots line up with baseline PNG page_NNN.
    from .pageorder import write_initial_order

    write_initial_order(paths, page_count=int(n_pages))

    # 4) Index the deck in the database (no-op when no DB is configured). The
    # on-disk project.json above stays the source of truth for the workspace;
    # this row is what GET /api/decks and the agent's deck tools read from.
    if user_id:
        from .repo import upsert_deck

        upsert_deck(
            project_id=paths.project_id,
            user_id=user_id,
            title=title,
            page_count=int(n_pages),
            page_size_pt=page_size_pt,
            source_kind=source_kind or (pdf_path.suffix.lower().lstrip(".") or None),
            source_path=str(paths.root),
            workspace_root=str(paths.root),
        )
    return paths


def bootstrap_workspace_from_upload(
    *,
    source_path: Path,
    project_id: str,
    user_id: str | None = None,
    results_root: Path | None = None,
    convert_outdir: Path | None = None,
    dpi: float = 150.0,
    need_image_descriptions: bool = True,
    need_original_layout_description: bool = True,
    title: str = "PPTAgent",
) -> WorkspacePaths:
    """Bootstrap a workspace from a supported PowerPoint upload.

    The PDF produced here is a temporary rasterization source for
    ``baseline/pages_png`` only.
    """
    from .convert import ConversionError, convert_to_pdf

    source_path = source_path.expanduser().resolve()
    if source_path.suffix.lower() not in {".ppt", ".pptx"}:
        raise ConversionError("only .ppt and .pptx uploads are supported")
    outdir = convert_outdir or source_path.parent
    pdf_path = convert_to_pdf(src=source_path, outdir=outdir)
    return _bootstrap_workspace_from_pdf_render(
        pdf_path=pdf_path,
        project_id=project_id,
        user_id=user_id,
        results_root=results_root,
        dpi=dpi,
        need_image_descriptions=need_image_descriptions,
        need_original_layout_description=need_original_layout_description,
        title=title,
        source_kind=source_path.suffix.lower().lstrip(".") or None,
        original_upload=source_path,
    )
