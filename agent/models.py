"""Chat model wiring for the conversational agent.

The agent talks to Claude through the same OpenAI-compatible proxy the inner
pipeline uses (``PPT_LLM_BASE_URL`` / ``PPT_LLM_API_KEY``). We build a plain
``ChatOpenAI`` instance and hand it to ``create_deep_agent`` directly, so no
provider auto-resolution happens.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "claude-opus-4-8"


def _repair_tool_arguments(raw: str) -> str:
    """Repair malformed tool-call ``arguments`` returned by the proxy.

    The intermediate proxy sometimes prepends a spurious empty-object ``{}`` to
    the real JSON (e.g. ``'{}{"a": 2}'``), which is not valid JSON and makes
    LangChain drop the tool call. We defensively strip leading empty
    ``{}`` / ``[]`` fragments until the remainder parses as JSON.
    """
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return raw
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    for prefix in ("{}", "[]"):
        while s.startswith(prefix):
            candidate = s[len(prefix):].lstrip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                s = candidate
                break
        else:
            continue
    return raw


def _repair_response_tool_calls(response_dict: dict[str, Any]) -> dict[str, Any]:
    """Walk an OpenAI-shaped response dict and repair tool-call arguments."""
    try:
        for choice in response_dict.get("choices") or []:
            msg = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                    fn["arguments"] = _repair_tool_arguments(fn["arguments"])
    except Exception:
        # Never let repair break a valid response.
        return response_dict
    return response_dict


class ProxyChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that repairs malformed proxy tool-call payloads."""

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):  # type: ignore[override]
        if isinstance(response, dict):
            response = _repair_response_tool_calls(response)
        else:
            try:
                dumped = response.model_dump(
                    exclude={"choices": {"__all__": {"message": {"parsed"}}}}
                )
                response = _repair_response_tool_calls(dumped)
            except Exception:
                pass
        return super()._create_chat_result(response, generation_info)


def agent_model_name() -> str:
    """Resolve the model id the conversational agent should drive."""
    return (
        os.environ.get("PPT_AGENT_MODEL")
        or os.environ.get("PPT_PIPELINE_MODEL")
        or os.environ.get("PPT_LLM_MODEL")
        or DEFAULT_MODEL
    )


def build_chat_model() -> ChatOpenAI:
    """Construct the ChatOpenAI client pointed at the intermediate proxy.

    Raises a clear error if the required proxy env vars are missing so the
    server fails fast at startup rather than on the first chat request.
    """
    base_url = os.environ.get("PPT_LLM_BASE_URL")
    api_key = os.environ.get("PPT_LLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "PPT_LLM_BASE_URL and PPT_LLM_API_KEY must be set (see .env) "
            "to run the conversational agent."
        )

    return ProxyChatOpenAI(
        model=agent_model_name(),
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        max_retries=3,
        # The proxy blocks the default openai-python User-Agent; the inner
        # pipeline spoofs curl for the same reason. Match it here.
        default_headers={"User-Agent": "curl/8.4.0"},
    )
