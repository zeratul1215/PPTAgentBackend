"""Canonical PPTist-native table helpers.

Tables are rectangular logical matrices.  A merged anchor owns its span and
the cells covered by that span remain empty placeholders, matching PPTist's
native JSON contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


STYLE_KEYS = {
    "bold", "em", "underline", "strikethrough", "color", "backcolor",
    "fontsize", "fontname", "align", "vAlign",
}
OUTLINE_STYLES = {"solid", "dashed", "dotted"}


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _cell(raw: Any, *, fallback_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    style = raw.get("style") if isinstance(raw.get("style"), dict) else {}
    style = {str(k): deepcopy(v) for k, v in style.items() if str(k) in STYLE_KEYS}
    return {
        "id": str(raw.get("id") or fallback_id),
        "text": str(raw.get("text") or ""),
        "rowspan": _positive_int(raw.get("rowspan", 1), "cell.rowspan"),
        "colspan": _positive_int(raw.get("colspan", 1), "cell.colspan"),
        "style": style,
    }


def _empty_cell(cell_id: str) -> dict[str, Any]:
    return {"id": cell_id, "text": "", "rowspan": 1, "colspan": 1, "style": {}}


def validate_table_spec(table: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one full TableSpec, returning a deep copy."""
    if not isinstance(table, dict):
        raise ValueError("table must be an object")
    table_id = str(table.get("id") or "").strip()
    if not table_id:
        raise ValueError("table.id is required")
    raw_data = table.get("data")
    if not isinstance(raw_data, list) or not raw_data or not all(isinstance(r, list) for r in raw_data):
        raise ValueError("table.data must be a non-empty rectangular array")
    rows = len(raw_data)
    cols = len(raw_data[0])
    if cols < 1 or any(len(row) != cols for row in raw_data):
        raise ValueError("table.data must be rectangular")

    data = [[_cell(raw, fallback_id=f"{table_id}_r{r}_c{c}")
             for c, raw in enumerate(row)] for r, row in enumerate(raw_data)]
    seen_ids: set[str] = set()
    occupied: list[list[tuple[int, int] | None]] = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = data[r][c]
            cid = cell["id"]
            if cid in seen_ids:
                raise ValueError(f"duplicate table cell id: {cid}")
            seen_ids.add(cid)
            covered = occupied[r][c]
            if covered is not None:
                if cell["text"] or cell["rowspan"] != 1 or cell["colspan"] != 1:
                    raise ValueError(f"cell {cid} conflicts with merged cell at {covered}")
                continue
            rs, cs = cell["rowspan"], cell["colspan"]
            if r + rs > rows or c + cs > cols:
                raise ValueError(f"cell {cid} span exceeds table bounds")
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    if occupied[rr][cc] is not None:
                        raise ValueError(f"overlapping merged cells at {(rr, cc)}")
                    occupied[rr][cc] = (r, c)

    widths = table.get("colWidths")
    if not isinstance(widths, list) or len(widths) != cols:
        widths = [1.0 / cols] * cols
    else:
        try:
            widths = [float(x) for x in widths]
        except (TypeError, ValueError) as exc:
            raise ValueError("table.colWidths must contain numbers") from exc
        if any(x <= 0 for x in widths):
            raise ValueError("table.colWidths must be positive")
        total = sum(widths)
        widths = [x / total for x in widths]

    outline_raw = table.get("outline") if isinstance(table.get("outline"), dict) else {}
    outline = {
        "width": max(0.0, float(outline_raw.get("width", 1) or 0)),
        "style": str(outline_raw.get("style") or "solid") if str(outline_raw.get("style") or "solid") in OUTLINE_STYLES else "solid",
        "color": str(outline_raw.get("color") or "#eeece1"),
    }
    theme_raw = table.get("theme") if isinstance(table.get("theme"), dict) else {}
    theme = {
        "color": str(theme_raw.get("color") or "#67508F"),
        "rowHeader": bool(theme_raw.get("rowHeader", False)),
        "rowFooter": bool(theme_raw.get("rowFooter", False)),
        "colHeader": bool(theme_raw.get("colHeader", False)),
        "colFooter": bool(theme_raw.get("colFooter", False)),
    }
    try:
        min_height = max(1.0, float(table.get("cellMinHeight", 24) or 24))
    except (TypeError, ValueError) as exc:
        raise ValueError("table.cellMinHeight must be numeric") from exc
    return {
        "id": table_id,
        "colWidths": widths,
        "cellMinHeight": min_height,
        "outline": outline,
        "theme": theme,
        "data": data,
    }


def table_from_pptist(element: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    """Extract a native PPTist table without changing its semantic structure."""
    raw = dict(element) if isinstance(element, dict) else {}
    raw["id"] = str(raw.get("id") or fallback_id)
    return validate_table_spec(raw)


def slim_table_for_model(table: dict[str, Any]) -> dict[str, Any]:
    """Return only content and merge information for an understanding model."""
    spec = validate_table_spec(table)
    return {
        "data": [[
            {"text": cell["text"], "rowspan": cell["rowspan"], "colspan": cell["colspan"]}
            for cell in row
        ] for row in spec["data"]]
    }


def iter_cells(table: dict[str, Any], *, include_placeholders: bool = True):
    """Iterate normalized cell copies. Use ``iter_cell_refs`` for mutation."""
    spec = validate_table_spec(table)
    occupied: list[list[tuple[int, int] | None]] = [[None] * len(spec["data"][0]) for _ in spec["data"]]
    for r, row in enumerate(spec["data"]):
        for c, cell in enumerate(row):
            if occupied[r][c] is not None:
                continue
            for rr in range(r, r + cell["rowspan"]):
                for cc in range(c, c + cell["colspan"]):
                    occupied[rr][cc] = (r, c)
    for r, row in enumerate(spec["data"]):
        for c, cell in enumerate(row):
            is_visible = occupied[r][c] == (r, c)
            if include_placeholders or is_visible:
                yield r, c, cell


def iter_cell_refs(table: dict[str, Any], *, include_placeholders: bool = True):
    """Iterate cells from the caller's table while validating span visibility.

    ``validate_table_spec`` deliberately returns a deep copy, so callers that
    need to update ``cell.text`` must use this iterator instead of
    ``iter_cells``. Validation happens before the first reference is yielded.
    """
    spec = validate_table_spec(table)
    raw_data = table.get("data")
    if not isinstance(raw_data, list):
        return
    rows = len(spec["data"])
    cols = len(spec["data"][0])
    occupied: list[list[tuple[int, int] | None]] = [[None] * cols for _ in range(rows)]
    for r, row in enumerate(spec["data"]):
        for c, cell in enumerate(row):
            if occupied[r][c] is not None:
                continue
            for rr in range(r, r + cell["rowspan"]):
                for cc in range(c, c + cell["colspan"]):
                    occupied[rr][cc] = (r, c)
    for r, row in enumerate(raw_data):
        if not isinstance(row, list):
            continue
        for c, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            is_visible = occupied[r][c] == (r, c)
            if include_placeholders or is_visible:
                yield r, c, cell


def find_cell(table: dict[str, Any], cell_id: str) -> tuple[int, int, dict[str, Any]] | None:
    for r, c, cell in iter_cell_refs(table):
        if cell["id"] == cell_id:
            return r, c, cell
    return None


def table_from_composer(rows: Any, table_id: str, header_rows: int = 0) -> dict[str, Any]:
    """Compile Composer's simple string matrix into a native no-merge table."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        raise ValueError("composer table rows must be a non-empty matrix")
    cols = len(rows[0])
    if cols < 1 or any(len(r) != cols for r in rows):
        raise ValueError("composer table rows must be rectangular")
    data = [[{"id": f"{table_id}_r{r}_c{c}", "text": str(v or ""), "rowspan": 1,
              "colspan": 1, "style": {}} for c, v in enumerate(row)] for r, row in enumerate(rows)]
    return validate_table_spec({"id": table_id, "colWidths": [1 / cols] * cols,
        "cellMinHeight": 24, "outline": {"width": 1, "style": "solid", "color": "#eeece1"},
        "theme": {"color": "#67508F", "rowHeader": header_rows > 0}, "data": data})
