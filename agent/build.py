"""Assemble the conversational deep agent.

We build a single agent instance (module-level singleton) and reuse it across
chat requests. Multi-turn memory is provided by a checkpointer keyed on
``thread_id = session_id`` (a chat thread, not a deck): a ``PostgresSaver`` when
``PPT_DATABASE_URL`` is configured (survives restarts), otherwise an
``InMemorySaver`` fallback. The session's ``session_id`` + ``user_id`` are
injected through runtime ``context``; the active deck is resolved live from the
session store, so the user can switch decks mid-conversation.

Built-in filesystem / ``execute`` tools and the default general-purpose
subagent are stripped via a registered ``HarnessProfile`` so the agent only
sees our PPT tools.
"""

from __future__ import annotations

import os
import threading

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    SubAgent,
    create_deep_agent,
    register_harness_profile,
)
from langgraph.checkpoint.memory import InMemorySaver

from agent_backend.agent.models import agent_model_name, build_chat_model
from agent_backend.agent.prompts import (
    DECK_TASK_AGENT_DESCRIPTION,
    DECK_TASK_AGENT_PROMPT,
    OUTER_DECK_AGENT_PROMPT,
    TASK_TOOL_DESCRIPTION,
)
from agent_backend.agent.tools import ALL_TOOLS, AgentContext
from agent_backend.workspace.db import database_url, get_pool


def _register_profile() -> None:
    """Strip built-in tools + the general-purpose subagent for our model.

    Registered under both the provider key and the exact provider:model key so
    the lookup matches regardless of how deepagents derives the profile key for
    a pre-built ChatOpenAI instance.
    """
    profile = HarnessProfile(
        excluded_tools=frozenset(
            {"execute", "ls", "read_file", "write_file", "edit_file", "glob", "grep"}
        ),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        tool_description_overrides={"task": TASK_TOOL_DESCRIPTION},
    )
    register_harness_profile("openai", profile)
    register_harness_profile(f"openai:{agent_model_name()}", profile)


_agent = None
_agent_lock = threading.Lock()


def _build_checkpointer():
    """Persistent checkpointer when a DB is configured, else in-memory.

    The ``PostgresSaver`` shares the app connection pool and creates its own
    checkpoint tables via ``.setup()`` (idempotent). If a database URL is
    configured, setup failure is fatal so resumable runs are never silently
    downgraded to ephemeral memory.
    """
    if not database_url():
        if os.getenv("PPT_ALLOW_EPHEMERAL_STATE") == "1":
            return InMemorySaver()
        return InMemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver

    pool = get_pool()
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def build_agent():
    """Construct the deep agent (fresh instance)."""
    _register_profile()
    model = build_chat_model()
    deck_task_agent: SubAgent = {
        "name": "deck-task-agent",
        "description": DECK_TASK_AGENT_DESCRIPTION,
        "system_prompt": DECK_TASK_AGENT_PROMPT,
        "tools": ALL_TOOLS,
        "model": model,
    }
    return create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=OUTER_DECK_AGENT_PROMPT,
        subagents=[deck_task_agent],
        context_schema=AgentContext,
        checkpointer=_build_checkpointer(),
    )


def get_agent():
    """Return the process-wide singleton agent, building it on first use."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = build_agent()
    return _agent
