"""Deterministic calculation helpers: a CLOSED set of enumerated numeric
operators plus numeric parsing utilities used by `data.calculate` and Patch.

Design guardrails (see changelog 2026-07-21_03):
- **No formula-string evaluation.** The model only ever picks an enumerated
  operator name + which columns/rows to feed it; ALL arithmetic happens here in
  plain Python. There is no `eval`, no expression parser, no open-ended math.
- **Closed operator set.** `OPERATORS` below is the whole vocabulary. Adding a
  capability = adding one named function here, never accepting arbitrary code.

All callers now read native TableSpec cells directly. This module owns only
numeric parsing, formatting, and the closed arithmetic operator set.
"""

from __future__ import annotations

import re
from typing import Callable, Optional


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
