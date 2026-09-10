"""Per-page reread state for manual edits and AI page updates.

``pending_reread`` records that a page's ``current_page_state.json`` is stale
and must be refreshed from PPTist JSON plus the page's current PNG.

Lifecycle:
- SET after the frontend has both saved current PPTist JSON and uploaded its
  matching render, after Patch, or after Full Pipeline commits final PPTist JSON.
- CHECKED only when that page is about to enter Full Pipeline. Read-only tools
  and Patch use PPTist JSON directly and do not consume this marker.
- CLEARED only after that reread succeeds.
"""

from __future__ import annotations

from .paths import WorkspacePaths, read_json, write_json


def mark_pending_reread(paths: WorkspacePaths, slot: int) -> None:
    """Mark a page stale after its current PPTist JSON and PNG are ready."""
    write_json(
        paths.page_pending_reread_marker(int(slot)),
        {"schema_version": "pending_reread_v1"},
    )


def is_pending_reread(paths: WorkspacePaths, slot: int) -> bool:
    """Whether ``slot`` has a pending understanding refresh."""
    marker = paths.page_pending_reread_marker(int(slot))
    if not marker.exists():
        return False
    try:
        data = read_json(marker)
        return isinstance(data, dict) and data.get("schema_version") == "pending_reread_v1"
    except Exception:
        # A corrupt marker must still trigger a conservative refresh attempt.
        return True


def clear_pending_reread(paths: WorkspacePaths, slot: int) -> None:
    """Clear the current reread marker for this slot."""
    try:
        paths.page_pending_reread_marker(int(slot)).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
