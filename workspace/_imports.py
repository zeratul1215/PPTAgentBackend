"""Internal path helpers for the standalone subproject.

``PROJECT_ROOT`` is the ``agent_backend/`` package directory (keeps ``result/``
and ``.env`` anchored there). The deterministic step-0 assets that used to be
resolved here now live with the heavy tools that consume them; see
``agent_backend.agent.tools.heavy_tool_impl._shared.api_script_dir``.
"""

from __future__ import annotations

from pathlib import Path


# This module lives at agent_backend/workspace/_imports.py, so parents[1] is the
# agent_backend/ package root (keeps result/ and .env anchored there).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return PROJECT_ROOT
