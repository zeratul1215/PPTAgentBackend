"""Provenance for successful HTML-backed page commits."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .paths import WorkspacePaths, read_json, write_json


def _without_transient_id(slide: Any) -> Any:
    if isinstance(slide, dict):
        return {k: v for k, v in slide.items() if k != "id"}
    return slide


def slide_hash(slide: Any) -> str:
    payload = json.dumps(
        _without_transient_id(slide), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def lineage_path(paths: WorkspacePaths, page_num: int) -> Path:
    return paths.page_state_dir(int(page_num)) / "html_lineage.json"


def write_lineage(
    paths: WorkspacePaths, page_num: int, *, source_turn: str,
    final_html_path: str, slide: dict[str, Any], source: str = "full_pipeline",
) -> dict[str, Any]:
    record = {
        "schema_version": "html_lineage_v1",
        "source_turn": str(source_turn),
        "final_html_path": str(final_html_path),
        "slide_hash": slide_hash(slide),
        "source": str(source),
        "created_at": time.time(),
    }
    write_json(lineage_path(paths, page_num), record)
    return record


def current_candidate(paths: WorkspacePaths, page_num: int) -> dict[str, Any]:
    marker = lineage_path(paths, page_num)
    result: dict[str, Any] = {"available": False, "reason": "missing_lineage"}
    try:
        record = read_json(marker)
        if not isinstance(record, dict):
            result["reason"] = "invalid_lineage"
            return result
        html_path = Path(str(record.get("final_html_path") or ""))
        slide_path = paths.pptist_slide_json(int(page_num))
        if not html_path.exists():
            result["reason"] = "html_missing"
            return result
        if not slide_path.exists():
            result["reason"] = "slide_missing"
            return result
        current_hash = slide_hash(read_json(slide_path))
        if current_hash != str(record.get("slide_hash") or ""):
            result["reason"] = "slide_changed"
            return result
        result.update({"available": True, "reason": "current", "record": record, "html_path": str(html_path)})
    except Exception as exc:  # conservative: stale lineage is never fatal
        result["reason"] = f"lineage_read_error:{type(exc).__name__}"
    return result

