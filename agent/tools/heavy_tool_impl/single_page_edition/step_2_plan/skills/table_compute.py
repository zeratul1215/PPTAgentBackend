"""Skill: table.compute — derive numeric values for a table with DETERMINISTIC
Python arithmetic.

The model NEVER does the math. At plan time it only picks an enumerated operator
(`sum`/`mean`/`min`/`max`/`count`), which columns (or rows) to aggregate, and
where to place the result. This skill then:
  1. reads the referenced cells' display text,
  2. parses numbers out of them (currency/percent/thousands aware),
  3. reduces them with the chosen operator IN PYTHON,
  4. creates a NEW derived text node per result (carrying `derived_from` +
     `derived_op` provenance), and
  5. refs that node into the target cell.

No model call, no formula strings, no `eval` — a closed operator set only.
Runs in the TABLE phase (after all text edits), so the numbers it aggregates are
the FINAL cell values.
"""

from __future__ import annotations

from typing import Any

from .base import Skill, SkillResult, _append_derived_text
from . import table_calc as tc


_COMPUTE_PLAN_DOC = """  Compute DERIVED numeric values over an existing table (totals, averages,
  counts, min/max) and write them into a target row or column. Use for
  "在最下面加一行求每列的和", "算出每一行的平均分放到最后一列",
  "统计这一列的最大值". The actual arithmetic is done deterministically by code;
  you only specify WHAT to compute and WHERE it goes — never do the math
  yourself.
  params:
    * "table_id": OPTIONAL id of the target table (omit if the page has one table).
    * "op": one of "sum" | "mean" | "min" | "max" | "count".
    * "axis": "column" (default) aggregates DOWN each listed column and writes one
      result per column into a target ROW; "row" aggregates ACROSS each listed
      row and writes one result per row into a target COLUMN.
    * "indices": list of 0-based column indices (axis="column") or row indices
      (axis="row") to aggregate. Omit to aggregate ALL columns/rows (except a
      header when include_header=false).
    * "include_header": bool (default false) — whether the first row/col is part
      of the numbers being aggregated. Usually false (headers are labels).
    * "target": where the results go: "append" (default; add a NEW row/col at the
      end) OR an integer index of an EXISTING row (axis="column") / column
      (axis="row") to fill.
    * "label": OPTIONAL text for the target's leading label cell (e.g. "Total",
      "合计"), placed at column 0 (axis="column") or row 0 (axis="row") of the
      target line.
  Runs in the TABLE phase. Requests a visual re-layout."""


_COMPUTE_ORDERING_NOTE = (
    "TABLE phase, after all text edits and after the table.build that created "
    "the table. Use `after` on that table.build intent when both target the "
    "same new table."
)

_OPS = set(tc.operator_names())


def _repair_compute_params(params: dict[str, Any]) -> list[str]:
    repairs: list[str] = []
    op = params.get("op")
    if not isinstance(op, str) or op not in _OPS:
        params["op"] = "sum"
        repairs.append("repaired_compute_op_to_sum")
    axis = params.get("axis")
    if axis not in ("column", "row"):
        params["axis"] = "column"
        repairs.append("repaired_compute_axis_to_column")
    if params.get("indices") is not None and not isinstance(params.get("indices"), list):
        params.pop("indices", None)
        repairs.append("repaired_compute_dropped_bad_indices")
    if not isinstance(params.get("include_header"), bool):
        params["include_header"] = False
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
    real = [t for t in tables if isinstance(t, dict)]
    return real[0] if len(real) == 1 else None


def _set_cell_ref(table: dict[str, Any], row: int, col: int, ref: str) -> None:
    """Point cell (row,col) at `ref`, replacing any existing cell there."""
    cells = [c for c in tc.cells_of(table)
             if not (tc._as_int(c.get("row")) == row and tc._as_int(c.get("col")) == col)]
    cells.append({"row": row, "col": col, "ref": ref})
    table["cells"] = cells


def _run_compute(
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
    op = str(params.get("op") or "sum")
    if op not in _OPS:
        warnings.append(f"compute_bad_op[{iid}]: {op!r}")
        return SkillResult(warnings=warnings, triggered_visual=False)
    axis = str(params.get("axis") or "column")

    table = _select_table(state, str(params.get("table_id") or ""))
    if table is None:
        warnings.append(f"compute_no_target_table[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    rows = tc._as_int(table.get("rows"))
    cols = tc._as_int(table.get("cols"))
    if rows <= 0 or cols <= 0:
        warnings.append(f"compute_bad_table_dims[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)

    texts = state.get("texts") if isinstance(state.get("texts"), list) else []
    by_id = {str(t.get("id") or ""): t for t in texts if isinstance(t, dict)}
    include_header = bool(params.get("include_header", False))
    label = params.get("label")
    label = str(label) if isinstance(label, str) and label.strip() else ""

    if axis == "column":
        applied = _compute_columns(
            table, params, op, rows, cols, by_id, state, include_header, label,
            warnings, iid,
        )
    else:
        applied = _compute_rows(
            table, params, op, rows, cols, by_id, state, include_header, label,
            warnings, iid,
        )

    if applied == 0:
        warnings.append(f"compute_no_values[{iid}]")
        return SkillResult(warnings=warnings, triggered_visual=False)
    warnings.append(f"compute_applied[{iid}]: {op} x{applied}")
    return SkillResult(warnings=warnings, triggered_visual=True)


def _target_index(params: dict[str, Any], current: int) -> tuple[int, bool]:
    """Return (index, is_append). `target` = "append" grows the table by one."""
    tgt = params.get("target")
    if isinstance(tgt, int):
        return tgt, False
    if isinstance(tgt, str) and tgt.strip().isdigit():
        return int(tgt.strip()), False
    return current, True  # append at the end


def _source_line(indices: list[int], header_skip: bool, size: int) -> list[int]:
    if indices:
        return [i for i in indices if 0 <= i < size]
    start = 1 if header_skip and size > 1 else 0
    return list(range(start, size))


def _compute_columns(
    table, params, op, rows, cols, by_id, state, include_header, label,
    warnings, iid,
) -> int:
    idxs_raw = params.get("indices") if isinstance(params.get("indices"), list) else []
    columns = [tc._as_int(x) for x in idxs_raw]
    columns = [c for c in columns if 0 <= c < cols]
    if not columns:
        # default: every column except a label column 0 when there is a header
        columns = list(range(1, cols)) if cols > 1 else list(range(cols))

    target_row, is_append = _target_index(params, rows)
    if is_append:
        table["rows"] = rows + 1
        target_row = rows
    elif not (0 <= target_row < rows):
        warnings.append(f"compute_target_row_out_of_range[{iid}]: {target_row}")
        return 0

    # Source rows are every row above the target that isn't the header (unless
    # include_header) — never the target row itself.
    src_rows = [
        r for r in _source_line([], not include_header, table["rows"])
        if r != target_row
    ]

    applied = 0
    for c in columns:
        vals: list[float] = []
        prefix = suffix = ""
        decimals = 0
        contributing: list[str] = []
        for r in src_rows:
            raw = tc.cell_display_text(table, r, c, by_id)
            if raw is None:
                continue
            parsed = tc.parse_number(raw)
            if parsed is None:
                continue
            v, pfx, sfx, dec = parsed
            vals.append(v)
            decimals = max(decimals, dec)
            if pfx:
                prefix = pfx
            if sfx:
                suffix = sfx
            cell_ref = _ref_at(table, r, c)
            if cell_ref:
                contributing.append(cell_ref)
        if not vals:
            continue
        result = tc.reduce_values(op, vals)
        if result is None:
            continue
        text = tc.format_number(
            result, currency_prefix=prefix, percent_suffix=suffix, decimals=decimals
        )
        new_id = _append_derived_text(
            state=state, text=text, derived_from=contributing, derived_op=op
        )
        if new_id:
            _set_cell_ref(table, target_row, c, new_id)
            applied += 1

    if applied and label:
        label_id = _append_derived_text(state=state, text=label, derived_op="label")
        if label_id:
            _set_cell_ref(table, target_row, 0, label_id)
    return applied


def _compute_rows(
    table, params, op, rows, cols, by_id, state, include_header, label,
    warnings, iid,
) -> int:
    idxs_raw = params.get("indices") if isinstance(params.get("indices"), list) else []
    row_idxs = [tc._as_int(x) for x in idxs_raw]
    row_idxs = [r for r in row_idxs if 0 <= r < rows]
    if not row_idxs:
        row_idxs = list(range(1, rows)) if rows > 1 else list(range(rows))

    target_col, is_append = _target_index(params, cols)
    if is_append:
        table["cols"] = cols + 1
        target_col = cols
    elif not (0 <= target_col < cols):
        warnings.append(f"compute_target_col_out_of_range[{iid}]: {target_col}")
        return 0

    src_cols = [
        c for c in _source_line([], not include_header, table["cols"])
        if c != target_col
    ]

    applied = 0
    for r in row_idxs:
        vals: list[float] = []
        prefix = suffix = ""
        decimals = 0
        contributing: list[str] = []
        for c in src_cols:
            raw = tc.cell_display_text(table, r, c, by_id)
            if raw is None:
                continue
            parsed = tc.parse_number(raw)
            if parsed is None:
                continue
            v, pfx, sfx, dec = parsed
            vals.append(v)
            decimals = max(decimals, dec)
            if pfx:
                prefix = pfx
            if sfx:
                suffix = sfx
            cell_ref = _ref_at(table, r, c)
            if cell_ref:
                contributing.append(cell_ref)
        if not vals:
            continue
        result = tc.reduce_values(op, vals)
        if result is None:
            continue
        text = tc.format_number(
            result, currency_prefix=prefix, percent_suffix=suffix, decimals=decimals
        )
        new_id = _append_derived_text(
            state=state, text=text, derived_from=contributing, derived_op=op
        )
        if new_id:
            _set_cell_ref(table, r, target_col, new_id)
            applied += 1

    if applied and label:
        label_id = _append_derived_text(state=state, text=label, derived_op="label")
        if label_id:
            _set_cell_ref(table, 0, target_col, label_id)
    return applied


def _ref_at(table: dict[str, Any], row: int, col: int) -> str:
    for c in tc.cells_of(table):
        if tc._as_int(c.get("row")) == row and tc._as_int(c.get("col")) == col:
            return str(c.get("ref") or "")
    return ""


SKILL = Skill(
    id="table.compute",
    canonical_rank=5,
    summary="derive numeric values (sum/mean/min/max/count) into a table row/col",
    plan_doc=_COMPUTE_PLAN_DOC,
    repair=_repair_compute_params,
    execute=_run_compute,
    ordering_note=_COMPUTE_ORDERING_NOTE,
    phase="table",
)
