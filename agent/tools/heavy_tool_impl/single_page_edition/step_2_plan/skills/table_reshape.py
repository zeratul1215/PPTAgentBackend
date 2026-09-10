"""Skill: table.reshape — deterministic structural edits on an existing table.

NO model call. The planner picks an enumerated `op` + its params; this skill
rewrites the grid in plain Python. It never edits cell TEXT (that is the text
skills' job) — it only moves/adds/removes rows and columns, transposes, or sorts
rows by a column's value.

Supported ops (closed set):
  * "add_row"    {at?: int}                 insert an empty row (default: append)
  * "add_col"    {at?: int}                 insert an empty column (default: append)
  * "delete_row" {row: int}
  * "delete_col" {col: int}
  * "swap_rows"  {a: int, b: int}
  * "swap_cols"  {a: int, b: int}
  * "move_row"   {from: int, to: int}
  * "move_col"   {from: int, to: int}
  * "transpose"  {}
  * "sort_rows_by_col" {col: int, order?: "asc"|"desc", numeric?: bool,
                        keep_header?: bool}
"""

from __future__ import annotations

from typing import Any

from .base import Skill, SkillResult
from . import table_calc as tc


_RESHAPE_PLAN_DOC = """  Restructure an EXISTING table deterministically (no text change). Use for
  "交换第 3 和第 4 列", "把最后一行删掉", "加一列", "转置这个表格",
  "按第二列从大到小排序". Does not compute values (use table.compute) and does
  not edit cell wording (use text.rewrite/translate/redact scoped to the cell).
  params:
    * "table_id": OPTIONAL id of the target table. Omit when the page has a
      single table.
    * "op": one of "add_row" | "add_col" | "delete_row" | "delete_col" |
      "swap_rows" | "swap_cols" | "move_row" | "move_col" | "transpose" |
      "sort_rows_by_col".
    * op-specific fields (all 0-based indices):
        - add_row/add_col: "at" (OPTIONAL int; default = append at the end).
        - delete_row: "row"; delete_col: "col".
        - swap_rows/swap_cols: "a", "b".
        - move_row/move_col: "from", "to".
        - transpose: no extra fields.
        - sort_rows_by_col: "col"; OPTIONAL "order" ("asc"|"desc", default asc),
          "numeric" (bool, default true — compare as numbers when possible),
          "keep_header" (bool, default true — pin row 0 in place).
  Runs in the TABLE phase (after all text edits). Requests a visual re-layout."""


_RESHAPE_ORDERING_NOTE = (
    "TABLE phase, after all text edits and usually after table.build (reshape "
    "operates on a table that already exists). Emit `after` on the relevant "
    "table.build intent when both target the same new table."
)


_OPS = {
    "add_row", "add_col", "delete_row", "delete_col",
    "swap_rows", "swap_cols", "move_row", "move_col",
    "transpose", "sort_rows_by_col",
}


def _repair_reshape_params(params: dict[str, Any]) -> list[str]:
    repairs: list[str] = []
    op = params.get("op")
    if not isinstance(op, str) or op not in _OPS:
        params["op"] = ""
        repairs.append("repaired_reshape_op_invalid")
    return repairs


def _select_table(state: dict[str, Any], table_id: str) -> dict[str, Any] | None:
    tables = state.get("tables")
    if not isinstance(tables, list) or not tables:
        return None
    if table_id:
        for t in tables:
            if isinstance(t, dict) and str(t.get("id") or "") == table_id:
                return t
        return None
    # No id given: only unambiguous when there is exactly one table.
    real = [t for t in tables if isinstance(t, dict)]
    return real[0] if len(real) == 1 else None


def _remap_cols(table: dict[str, Any], new_of_old: dict[int, int], new_cols: int) -> None:
    cells = tc.cells_of(table)
    out: list[dict[str, Any]] = []
    for cell in cells:
        oc = tc._as_int(cell.get("col"))
        if oc in new_of_old:
            out.append({**cell, "col": new_of_old[oc]})
    table["cells"] = out
    table["cols"] = new_cols


def _remap_rows(table: dict[str, Any], new_of_old: dict[int, int], new_rows: int) -> None:
    cells = tc.cells_of(table)
    out: list[dict[str, Any]] = []
    for cell in cells:
        orr = tc._as_int(cell.get("row"))
        if orr in new_of_old:
            out.append({**cell, "row": new_of_old[orr]})
    table["cells"] = out
    table["rows"] = new_rows


def _run_reshape(
    *,
    intent: dict[str, Any],
    state: dict[str, Any],
    api_key: str | None,
    model: str,
    dry_run: bool,
    user_request: str = "",
) -> SkillResult:
    warnings: list[str] = []
    iid = str(intent.get("id") or "")
    params = intent.get("params") if isinstance(intent.get("params"), dict) else {}
    op = str(params.get("op") or "")
    if op not in _OPS:
        warnings.append(f"reshape_bad_op[{iid}]: {op!r}")
        return SkillResult(warnings=warnings, triggered_visual=False)

    table = _select_table(state, str(params.get("table_id") or ""))
    if table is None:
        warnings.append(f"reshape_no_target_table[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    rows = tc._as_int(table.get("rows"))
    cols = tc._as_int(table.get("cols"))
    if rows <= 0 or cols <= 0:
        warnings.append(f"reshape_bad_table_dims[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    ok = False
    if op == "add_col":
        at = tc._as_int(params.get("at"), cols)
        at = max(0, min(at, cols))
        new_of_old = {c: (c if c < at else c + 1) for c in range(cols)}
        _remap_cols(table, new_of_old, cols + 1)
        ok = True
    elif op == "add_row":
        at = tc._as_int(params.get("at"), rows)
        at = max(0, min(at, rows))
        new_of_old = {r: (r if r < at else r + 1) for r in range(rows)}
        _remap_rows(table, new_of_old, rows + 1)
        ok = True
    elif op == "delete_col":
        c0 = tc._as_int(params.get("col"))
        if not (0 <= c0 < cols):
            warnings.append(f"reshape_col_out_of_range[{iid}]: {c0}")
        else:
            new_of_old = {c: (c if c < c0 else c - 1) for c in range(cols) if c != c0}
            _remap_cols(table, new_of_old, cols - 1)
            ok = True
    elif op == "delete_row":
        r0 = tc._as_int(params.get("row"))
        if not (0 <= r0 < rows):
            warnings.append(f"reshape_row_out_of_range[{iid}]: {r0}")
        else:
            new_of_old = {r: (r if r < r0 else r - 1) for r in range(rows) if r != r0}
            _remap_rows(table, new_of_old, rows - 1)
            ok = True
    elif op in ("swap_cols", "swap_rows"):
        a = tc._as_int(params.get("a"))
        b = tc._as_int(params.get("b"))
        n = cols if op == "swap_cols" else rows
        if not (0 <= a < n and 0 <= b < n):
            warnings.append(f"reshape_swap_out_of_range[{iid}]: {(a, b)}")
        else:
            mapping = {i: i for i in range(n)}
            mapping[a], mapping[b] = b, a
            if op == "swap_cols":
                _remap_cols(table, mapping, cols)
            else:
                _remap_rows(table, mapping, rows)
            ok = True
    elif op in ("move_col", "move_row"):
        src = tc._as_int(params.get("from"))
        dst = tc._as_int(params.get("to"))
        n = cols if op == "move_col" else rows
        if not (0 <= src < n and 0 <= dst < n):
            warnings.append(f"reshape_move_out_of_range[{iid}]: {(src, dst)}")
        else:
            order = list(range(n))
            order.remove(src)
            order.insert(dst, src)
            # order[new] = old  ->  need old -> new
            mapping = {old: new for new, old in enumerate(order)}
            if op == "move_col":
                _remap_cols(table, mapping, cols)
            else:
                _remap_rows(table, mapping, rows)
            ok = True
    elif op == "transpose":
        for cell in tc.cells_of(table):
            r = tc._as_int(cell.get("row"))
            c = tc._as_int(cell.get("col"))
            cell["row"], cell["col"] = c, r
        table["rows"], table["cols"] = cols, rows
        ok = True
    elif op == "sort_rows_by_col":
        ok = _sort_rows_by_col(table, params, state, warnings, iid)

    if not ok:
        return SkillResult(warnings=warnings, triggered_visual=False)
    warnings.append(f"reshape_applied[{iid}]: {op}")
    return SkillResult(warnings=warnings, triggered_visual=True)


def _sort_rows_by_col(
    table: dict[str, Any],
    params: dict[str, Any],
    state: dict[str, Any],
    warnings: list[str],
    iid: str,
) -> bool:
    col = tc._as_int(params.get("col"))
    rows = tc._as_int(table.get("rows"))
    cols = tc._as_int(table.get("cols"))
    if not (0 <= col < cols):
        warnings.append(f"reshape_sort_col_out_of_range[{iid}]: {col}")
        return False
    order_desc = str(params.get("order") or "asc").strip().lower() == "desc"
    numeric = params.get("numeric")
    numeric = True if not isinstance(numeric, bool) else numeric
    keep_header = params.get("keep_header")
    keep_header = True if not isinstance(keep_header, bool) else keep_header

    texts = state.get("texts") if isinstance(state.get("texts"), list) else []
    by_id = {str(t.get("id") or ""): t for t in texts if isinstance(t, dict)}

    start = 1 if (keep_header and rows > 1) else 0
    sortable = list(range(start, rows))

    def _key(r: int) -> tuple[int, Any]:
        raw = tc.cell_display_text(table, r, col, by_id) or ""
        if numeric:
            parsed = tc.parse_number(raw)
            if parsed is not None:
                return (0, parsed[0])
            return (1, raw)  # non-numeric sorts after numbers
        return (0, raw)

    sorted_rows = sorted(sortable, key=_key, reverse=order_desc)
    # old row -> new row
    mapping = {r: r for r in range(start)}
    for new_pos, old_r in enumerate(sorted_rows, start=start):
        mapping[old_r] = new_pos
    _remap_rows(table, mapping, rows)
    return True


SKILL = Skill(
    id="table.reshape",
    canonical_rank=4,
    summary="deterministic structural edits on a table (add/del/move/swap/transpose/sort)",
    plan_doc=_RESHAPE_PLAN_DOC,
    repair=_repair_reshape_params,
    execute=_run_reshape,
    ordering_note=_RESHAPE_ORDERING_NOTE,
    phase="table",
)
