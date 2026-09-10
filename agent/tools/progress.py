"""User-facing progress notes for a chat run.

The note is deliberately separate from assistant token streaming: tool-bearing
assistant messages can contain provider-specific call payloads, while this
tool accepts only one short, non-technical sentence and persists it as a
filtered run event.
"""

from __future__ import annotations

import re

from langchain.tools import ToolRuntime, tool

from agent_backend.agent.tools.context import require_agent_run_id, require_session_id
from agent_backend.workspace import repo


_INTERNAL_TERMS = re.compile(
    r"(?:page@|slot\b|element.?id|json\b|patch_pages|edit_pages|understand_pages|"
    r"stage_page_asset|deck-task-agent|subagent|tool\b|工具调用|子\s*agent)",
    re.IGNORECASE,
)
_FALLBACK = "我正在确认你的要求和目标页面，然后开始处理。"


@tool
def report_progress(message: str, runtime: ToolRuntime) -> dict[str, object]:
    """Show one short user-facing sentence describing the intended approach.

    Call this once before starting a slide modification. Describe what you will
    inspect or change in ordinary user language. Never mention tools, agents,
    internal identifiers, JSON, implementation details, or hidden reasoning.
    """
    run_id = require_agent_run_id(runtime)
    session_id = require_session_id(runtime)
    text = " ".join(str(message or "").split()).strip()
    if not text or _INTERNAL_TERMS.search(text):
        text = _FALLBACK
    text = text[:120].rstrip("，,；;：:")

    # A run gets one introductory note. This keeps retries or model repetition
    # from producing a stack of near-identical status lines.
    existing = repo.list_agent_events(run_id, after=0, limit=1000)
    if not any(str(event.get("type") or "") == "status_update" for event in existing):
        repo.append_agent_event(
            agent_run_id=run_id,
            session_id=session_id,
            event={"type": "status_update", "agent_run_id": run_id, "text": text},
        )
    return {"ok": True, "shown": True}


__all__ = ["report_progress"]
