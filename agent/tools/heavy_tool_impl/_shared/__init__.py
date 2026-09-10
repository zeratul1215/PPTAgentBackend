"""Shared step-0 tooling used by more than one heavy tool.

The deterministic step-0 helpers (PDF render / ref-HTML extraction / plan
build / CSS library) live under ``assets/`` and are consumed by both the
single-page editing engine (per-turn re-render in ``reread.py``, CSS in
step3/step4) and the PDF ingest bootstrap. They are neither global workspace
infrastructure nor owned by any single heavy tool, so they sit here.
"""

from __future__ import annotations

from .paths import ASSETS_DIR, api_script_dir

__all__ = ["ASSETS_DIR", "api_script_dir"]
