"""Deterministic table helpers: a CLOSED set of enumerated numeric operators
plus the shared grid utilities that `table.build` / `table.reshape` /
`table.compute` all rely on.

Design guardrails (see changelog 2026-07-21_03):
- **No formula-string evaluation.** The model only ever picks an enumerated
  operator name + which columns/rows to feed it; ALL arithmetic happens here in
  plain Python. There is no `eval`, no expression parser, no open-ended math.
- **Closed operator set.** `OPERATORS` below is the whole vocabulary. Adding a
  capability = adding one named function here, never accepting arbitrary code.

The grid helpers resolve a table cell to its display text using the SAME rule
step3 renders by: when one text node is a whole column (one segment per row),
the cell at that row maps to that node's row-th segment; otherwise a cell maps
to its node's full text.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .base import _get_segments


# ---------------------------------------------------------------------------
# Grid utilities (shared by build / reshape / compute)
# ---------------------------------------------------------------------------


def cells_of(table: dict[str, Any]) -> list[dict[str, Any]]:
    cells = table.get("cells")
    return [c for c in cells if isinstance(c, dict)] if isinstance(cells, list) else []


def column_cells(table: dict[str, Any], col: int) -> list[dict[str, Any]]:
    """Cells in `col`, sorted by row (row-major within the column)."""
    out = [c for c in cells_of(table) if _as_int(c.get("col")) == col]
    out.sort(key=lambda c: _as_int(c.get("row")))
    return out


def row_cells(table: dict[str, Any], row: int) -> list[dict[str, Any]]:
    out = [c for c in cells_of(table) if _as_int(c.get("row")) == row]
    out.sort(key=lambda c: _as_int(c.get("col")))
    return out


def cell_display_text(
    table: dict[str, Any],
    row: int,
    col: int,
    by_id: dict[str, dict[str, Any]],
) -> Optional[str]:
    """Resolve the visible text of cell (row, col).

    Mirrors step3's data-ref rule: if this cell's `ref` is shared by several
    cells in the SAME column (one text node spanning the column, one segment per
    row), return that node's segment at this row's position; otherwise return the
    node's whole text. Returns None when the cell or its node is missing.
    """
    target = None
    for c in cells_of(table):
        if _as_int(c.get("row")) == row and _as_int(c.get("col")) == col:
            target = c
            break
    if target is None:
        return None
    ref = str(target.get("ref") or "")
    node = by_id.get(ref)
    if not isinstance(node, dict):
        return None

    same_ref_in_col = [
        c for c in column_cells(table, col) if str(c.get("ref") or "") == ref
    ]
    if len(same_ref_in_col) > 1:
        rows_sorted = sorted({_as_int(c.get("row")) for c in same_ref_in_col})
        try:
            idx = rows_sorted.index(row)
        except ValueError:
            idx = -1
        segs = _get_segments(node)
        if 0 <= idx < len(segs):
            return segs[idx]
        return str(node.get("text") or "")
    return str(node.get("text") or "")


def _as_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Number parsing / formatting
# ---------------------------------------------------------------------------

# A leading currency symbol or a trailing percent sign is carried over to the
# formatted result when EVERY input shares it, so a column of "$1,200" sums to
# "$3,600" rather than a bare "3600".
_CURRENCY_PREFIXES = ("$", "€", "£", "¥", "￥", "₩", "₹")
_NUM_CORE_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> Optional[tuple[float, str, str, int]]:
    """Parse a numeric value out of a cell string.

    Returns (value, currency_prefix, percent_suffix, decimals) or None when no
    number is present. `decimals` is the count of fractional digits in the
    source, used to format an aggregate consistently. Thousands separators are
    stripped; a single leading currency symbol and/or trailing `%` are captured
    so the result can be re-formatted in the same style.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    prefix = ""
    for p in _CURRENCY_PREFIXES:
        if s.startswith(p):
            prefix = p
            s = s[len(p):].strip()
            break
    suffix = ""
    if s.endswith("%"):
        suffix = "%"
        s = s[:-1].strip()
    m = _NUM_CORE_RE.search(s)
    if not m:
        return None
    core = m.group(0).replace(",", "")
    try:
        val = float(core)
    except ValueError:
        return None
    decimals = 0
    if "." in core:
        decimals = len(core.split(".", 1)[1])
    return val, prefix, suffix, decimals


def format_number(
    value: float,
    *,
    currency_prefix: str = "",
    percent_suffix: str = "",
    decimals: int = 0,
    thousands: bool = True,
) -> str:
    """Format an aggregate back into a cell string, echoing the inputs' style."""
    dec = max(0, min(6, int(decimals)))
    if abs(value - round(value)) < 1e-9 and dec == 0:
        body = f"{int(round(value)):,}" if thousands else str(int(round(value)))
    else:
        body = f"{value:,.{dec}f}" if thousands else f"{value:.{dec}f}"
    return f"{currency_prefix}{body}{percent_suffix}"


# ---------------------------------------------------------------------------
# Enumerated operators (the WHOLE vocabulary; closed set)
# ---------------------------------------------------------------------------


def _op_sum(vals: list[float]) -> float:
    return float(sum(vals))


def _op_mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _op_min(vals: list[float]) -> float:
    return float(min(vals)) if vals else 0.0


def _op_max(vals: list[float]) -> float:
    return float(max(vals)) if vals else 0.0


def _op_count(vals: list[float]) -> float:
    return float(len(vals))


# name -> (reducer, needs_numeric_inputs). `count` still counts numeric cells so
# it stays consistent with the other reducers over the same parsed values.
OPERATORS: dict[str, Callable[[list[float]], float]] = {
    "sum": _op_sum,
    "mean": _op_mean,
    "min": _op_min,
    "max": _op_max,
    "count": _op_count,
}


def operator_names() -> list[str]:
    return list(OPERATORS.keys())


def reduce_values(op: str, values: list[float]) -> Optional[float]:
    fn = OPERATORS.get(op)
    if fn is None:
        return None
    return fn(values)
