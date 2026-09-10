"""Locator for the vendored step-0 assets directory.

Formerly ``workspace/_imports.api_script_dir``. Moved here because the assets
are a heavy-tool concern shared across tools, not global workspace
infrastructure. ``PROJECT_ROOT`` / ``project_root`` stay in
``agent_backend.workspace._imports``.
"""

from __future__ import annotations

from pathlib import Path


# The vendored step-0 assets directory (formerly the top-level script/, then
# workspace/assets/): render_pages_png.py / css_library.*.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def api_script_dir() -> Path:
    """Directory holding render_pages_png.py / css_library.*."""
    return ASSETS_DIR
