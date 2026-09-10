"""Tools exposed to the conversational agent.

The agent never touches the filesystem or the pipeline directly. It works
through a small, curated toolset:

* Deck management: ``list_decks``, ``set_active_deck`` — see the user's decks
  and switch which one this session is editing.
* Inspection (read-only): ``get_deck_outline``, ``locate_pages``,
  ``understand_pages`` — understand the active deck and resolve page references.
* Editing: ``patch_pages`` for layout-preserving changes, ``edit_pages`` for
  the full HTML pipeline, and ``fill_empty_pages`` for already-inserted blank
  pages.
* Structure: ``delete_pages`` / ``add_page`` / ``move_page`` — mutate the deck's
  ordered page list (never renumber on-disk slots).

Page numbers passed to these tools are 1-based *display positions*; the tools
resolve them to stable slots via ``workspace.pageorder`` so add/delete/reorder
stay consistent.

Neither ``session_id`` nor ``project_id`` is a tool argument. The server injects
``session_id`` + ``user_id`` per request via deepagents runtime context; the
active deck (``project_id``) is resolved live from the session store (see
``context.require_project_id``), so switching decks mid-turn takes effect
immediately.
"""

from __future__ import annotations

from agent_backend.agent.tools.assets import stage_page_asset
from agent_backend.agent.tools.chat_artifacts import inspect_chat_artifacts
from agent_backend.agent.tools.context import AgentContext, set_progress_publisher
from agent_backend.agent.tools.edit import edit_pages
from agent_backend.agent.tools.fill import fill_empty_pages
from agent_backend.agent.tools.deck_style import get_deck_style
from agent_backend.agent.tools.patch import patch_pages
from agent_backend.agent.tools.inspect import (
    get_deck_outline,
    locate_pages,
)
from agent_backend.agent.tools.manage import list_decks, set_active_deck
from agent_backend.agent.tools.page_understanding import understand_pages
from agent_backend.agent.tools.pages import add_page, delete_pages, move_page
from agent_backend.agent.tools.progress import report_progress

ALL_TOOLS = [
    list_decks,
    set_active_deck,
    get_deck_outline,
    locate_pages,
    understand_pages,
    get_deck_style,
    patch_pages,
    edit_pages,
    fill_empty_pages,
    delete_pages,
    add_page,
    move_page,
    stage_page_asset,
    inspect_chat_artifacts,
    report_progress,
]

__all__ = [
    "ALL_TOOLS",
    "AgentContext",
    "set_progress_publisher",
    "list_decks",
    "set_active_deck",
    "get_deck_outline",
    "locate_pages",
    "understand_pages",
    "get_deck_style",
    "patch_pages",
    "edit_pages",
    "fill_empty_pages",
    "delete_pages",
    "add_page",
    "move_page",
    "stage_page_asset",
    "inspect_chat_artifacts",
    "report_progress",
]
