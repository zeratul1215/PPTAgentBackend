"""Light-weight, read-only inspection tools.

These tools never mutate the deck. They let the agent understand the document
and resolve a user's page references before deciding what to edit.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from agent_backend.agent.tools.context import (
    outline,
    require_project_id,
    workspace_for,
)
from agent_backend.workspace import pageorder


@tool
def get_deck_outline(runtime: ToolRuntime) -> dict[str, Any]:
    """List every page of the current deck with its title and a short text preview.

    Use this first to understand the whole document before deciding which pages
    a request refers to. Page numbers are 1-based.
    """
    pid = require_project_id(runtime)
    paths = workspace_for(pid)
    pages = [_agent_page_view(e) for e in outline(paths)]
    return {
        "project_id": pid,
        "revision": pageorder.revision(paths),
        "page_count": len(pages),
        "pages": pages,
    }


def _agent_page_view(entry: dict[str, Any]) -> dict[str, Any]:
    """Project an outline entry onto what the agent should see: 1-based display
    ``page`` only. ``slot`` / ``source_pdf_index`` are internal plumbing the LLM
    must never reason about (page identity is positional from its point of view).
    """
    return {
        "page": entry.get("page"),
        "page_ref": entry.get("page_ref"),
        "title": entry.get("title", ""),
        "preview": entry.get("preview", ""),
        "num_texts": entry.get("num_texts"),
        "num_images": entry.get("num_images"),
        "loaded": entry.get("loaded", False),
        "processed": entry.get("processed", False),
    }


@tool
def locate_pages(query: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Resolve a page reference to concrete 1-based page numbers.

    `query` may be a page number ("page 3"), an ordinal ("the second slide"),
    a scope ("the whole document", "all pages"), or a topic description
    ("the revenue page"). Returns ranked candidate pages with their titles and
    previews; when confident there is a single best match it is ranked first.
    Prefer confirming ambiguous matches with the user only as a last resort.
    """
    pid = require_project_id(runtime)
    paths = workspace_for(pid)
    pages = [_agent_page_view(e) for e in outline(paths)]
    rev = pageorder.revision(paths)

    deterministic = _resolve_page_positions(query or "", len(pages))
    if deterministic is not None:
        scope = str(deterministic.get("scope") or "positions")
        if scope == "invalid":
            return {
                "project_id": pid,
                "revision": rev,
                "query": query,
                "scope": "invalid",
                "resolution": "deterministic",
                "matches": [],
                "positions": [],
                "invalid_positions": deterministic.get("invalid_positions", []),
                "error": deterministic.get("error")
                or "The requested page reference is outside the current deck.",
            }
        positions = [int(p) for p in deterministic.get("positions") or []]
        by_position = {int(e.get("page") or 0): e for e in pages}
        matches = [by_position[p] for p in positions if p in by_position]
        return {
            "project_id": pid,
            "revision": rev,
            "query": query,
            "scope": scope,
            "resolution": "deterministic",
            "positions": positions,
            "matches": matches,
        }

    q = _normalize_query(query or "")

    tokens = [tok for tok in _tokenize(q) if tok]
    scored: list[tuple[int, dict[str, Any]]] = []
    for entry in pages:
        hay = f"{entry.get('title', '')} {entry.get('preview', '')}".lower()
        score = sum(hay.count(tok) for tok in tokens) if tokens else 0
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [e for s, e in scored if s > 0] or pages
    return {
        "project_id": pid,
        "revision": rev,
        "query": query,
        "scope": "match",
        "resolution": "semantic",
        "matches": ranked,
    }


_ALL_PAGE_RE = re.compile(
    r"(?:\b(?:all|every|everything|entire|whole)\b.*\b(?:deck|document|presentation|page|pages|slide|slides)\b)|"
    r"(?:全(?:部|文|篇|文件|部文档|部页面|部幻灯片)|整(?:个|份)(?:ppt|PPT|文档|文件|演示文稿)|所有(?:页面|页|幻灯片))",
    re.IGNORECASE,
)
_ZH_NUM = r"[零〇一二三四五六七八九十百两\d]+"
_ZH_PAGE_SEG_RE = re.compile(
    rf"(第?\s*{_ZH_NUM}(?:\s*(?:[,，、和及]|到|至|-|~|—|–|－)\s*第?\s*{_ZH_NUM})*)\s*页"
)
_ZH_PAGE_RANGE_WITH_PAGE_RE = re.compile(
    rf"第?\s*({_ZH_NUM})\s*页\s*(?:到|至|-|~|—|–|－)\s*第?\s*({_ZH_NUM})\s*页"
)
_ZH_RANGE_RE = re.compile(rf"第?\s*({_ZH_NUM})\s*(?:到|至|-|~|—|–|－)\s*第?\s*({_ZH_NUM})")
_EN_NUM_TOKEN = (
    r"(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth|twentieth|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty)"
)
_EN_RANGE_RE = re.compile(
    rf"\b(?:pages?|slides?)\s+({_EN_NUM_TOKEN})\s*(?:-|~|to|through)\s*({_EN_NUM_TOKEN})\b",
    re.IGNORECASE,
)
_EN_LIST_RE = re.compile(
    rf"\b(?:pages?|slides?)\s+({_EN_NUM_TOKEN}(?:\s*(?:,|，|and)\s*{_EN_NUM_TOKEN})*)\b",
    re.IGNORECASE,
)
_EN_ORDINAL_RE = re.compile(
    rf"\b(?:the\s+)?({_EN_NUM_TOKEN})\s+(?:page|slide)\b",
    re.IGNORECASE,
)

_EN_NUM_WORDS = {
    "one": 1,
    "first": 1,
    "two": 2,
    "second": 2,
    "three": 3,
    "third": 3,
    "four": 4,
    "fourth": 4,
    "five": 5,
    "fifth": 5,
    "six": 6,
    "sixth": 6,
    "seven": 7,
    "seventh": 7,
    "eight": 8,
    "eighth": 8,
    "nine": 9,
    "ninth": 9,
    "ten": 10,
    "tenth": 10,
    "eleven": 11,
    "eleventh": 11,
    "twelve": 12,
    "twelfth": 12,
    "thirteen": 13,
    "thirteenth": 13,
    "fourteen": 14,
    "fourteenth": 14,
    "fifteen": 15,
    "fifteenth": 15,
    "sixteen": 16,
    "sixteenth": 16,
    "seventeen": 17,
    "seventeenth": 17,
    "eighteen": 18,
    "eighteenth": 18,
    "nineteen": 19,
    "nineteenth": 19,
    "twenty": 20,
    "twentieth": 20,
}


def _normalize_query(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip().lower()


def _resolve_page_positions(query: str, page_count: int) -> dict[str, Any] | None:
    text = _normalize_query(query)
    if not text:
        return None
    if _ALL_PAGE_RE.search(text) or text in {
        "all",
        "all pages",
        "every page",
        "everything",
        "whole document",
        "the whole document",
        "entire deck",
        "entire presentation",
    }:
        return {"scope": "all", "positions": list(range(1, int(page_count) + 1))}

    positions: list[int] = []
    invalid: list[int | str] = []
    page_syntax = False

    for vals in _special_position_refs(text, page_count):
        page_syntax = True
        positions.extend(vals)

    for match in _ZH_PAGE_RANGE_WITH_PAGE_RE.finditer(text):
        page_syntax = True
        start = _parse_page_number(match.group(1), chinese=True)
        end = _parse_page_number(match.group(2), chinese=True)
        _append_range(start, end, page_count, positions, invalid)

    for match in _ZH_PAGE_SEG_RE.finditer(text):
        prefix = text[max(0, match.start() - 3) : match.start()]
        if prefix.endswith(("前", "后", "倒数")):
            continue
        page_syntax = True
        segment = match.group(1)
        _append_page_segment(segment, page_count, positions, invalid, chinese=True)

    for match in _EN_RANGE_RE.finditer(text):
        page_syntax = True
        start = _parse_page_number(match.group(1), chinese=False)
        end = _parse_page_number(match.group(2), chinese=False)
        _append_range(start, end, page_count, positions, invalid)

    for match in _EN_LIST_RE.finditer(text):
        page_syntax = True
        segment = match.group(1)
        _append_page_segment(segment, page_count, positions, invalid, chinese=False)

    for match in _EN_ORDINAL_RE.finditer(text):
        page_syntax = True
        val = _parse_page_number(match.group(1), chinese=False)
        _append_single(val, page_count, positions, invalid, raw=match.group(1))

    if not page_syntax:
        return None
    if invalid:
        return {
            "scope": "invalid",
            "invalid_positions": _dedupe(invalid),
            "error": f"Page reference is outside the current deck of {page_count} pages.",
        }
    resolved = [int(p) for p in _dedupe(positions) if 1 <= int(p) <= int(page_count)]
    if not resolved:
        return {
            "scope": "invalid",
            "invalid_positions": [],
            "error": "No valid page positions were resolved from the page reference.",
        }
    return {"scope": "positions", "positions": resolved}


def _special_position_refs(text: str, page_count: int) -> list[list[int]]:
    refs: list[list[int]] = []
    if re.search(r"(?:首页|第一页|第1页|\bfirst\s+(?:page|slide)\b)", text):
        refs.append([1])
    if re.search(r"(?:最后一页|末页|尾页|\blast\s+(?:page|slide)\b)", text):
        refs.append([page_count])
    m = re.search(r"倒数\s*第?\s*(" + _ZH_NUM + r")\s*页", text)
    if m:
        n = _parse_page_number(m.group(1), chinese=True)
        if n is not None:
            refs.append([page_count - n + 1])
    m = re.search(r"\b(?:second|2nd)\s+to\s+last\s+(?:page|slide)\b", text)
    if m:
        refs.append([page_count - 1])
    for m in re.finditer(r"前\s*(" + _ZH_NUM + r")\s*页", text):
        n = _parse_page_number(m.group(1), chinese=True)
        if n is not None:
            refs.append(list(range(1, min(n, page_count) + 1)))
    for m in re.finditer(r"后\s*(" + _ZH_NUM + r")\s*页", text):
        n = _parse_page_number(m.group(1), chinese=True)
        if n is not None:
            refs.append(list(range(max(1, page_count - n + 1), page_count + 1)))
    for m in re.finditer(r"\bfirst\s+(\d+)\s+(?:pages|slides)\b", text):
        n = int(m.group(1))
        refs.append(list(range(1, min(n, page_count) + 1)))
    for m in re.finditer(r"\blast\s+(\d+)\s+(?:pages|slides)\b", text):
        n = int(m.group(1))
        refs.append(list(range(max(1, page_count - n + 1), page_count + 1)))
    return refs


def _append_page_segment(
    segment: str,
    page_count: int,
    positions: list[int],
    invalid: list[int | str],
    *,
    chinese: bool,
) -> None:
    if chinese:
        range_match = _ZH_RANGE_RE.search(segment)
        nums = re.findall(_ZH_NUM, segment)
    else:
        range_match = re.search(
            r"([a-z0-9]+(?:st|nd|rd|th)?)\s*(?:-|~|to|through)\s*([a-z0-9]+(?:st|nd|rd|th)?)",
            segment,
        )
        nums = re.findall(r"\b(?:\d+(?:st|nd|rd|th)?|[a-z]+)\b", segment)
        nums = [
            n
            for n in nums
            if n not in {"and", "to", "through", "page", "pages", "slide", "slides"}
        ]
    if range_match:
        start = _parse_page_number(range_match.group(1), chinese=chinese)
        end = _parse_page_number(range_match.group(2), chinese=chinese)
        _append_range(start, end, page_count, positions, invalid)
        return
    for raw in nums:
        val = _parse_page_number(raw, chinese=chinese)
        if (
            chinese
            and val is not None
            and isinstance(raw, str)
            and raw.isdigit()
            and len(raw) > 2
            and val > page_count
        ):
            digits = [int(ch) for ch in raw if ch.isdigit()]
            if (
                digits
                and len(digits) == len(raw)
                and all(1 <= d <= page_count for d in digits)
                and len(set(digits)) == len(digits)
            ):
                positions.extend(digits)
                continue
        _append_single(val, page_count, positions, invalid, raw=raw)


def _append_range(
    start: int | None,
    end: int | None,
    page_count: int,
    positions: list[int],
    invalid: list[int | str],
) -> None:
    if start is None or end is None:
        invalid.append("unparsed-range")
        return
    if not (1 <= start <= page_count and 1 <= end <= page_count):
        invalid.extend([p for p in (start, end) if not (1 <= p <= page_count)])
        return
    step = 1 if start <= end else -1
    positions.extend(list(range(start, end + step, step)))


def _append_single(
    val: int | None,
    page_count: int,
    positions: list[int],
    invalid: list[int | str],
    *,
    raw: Any,
) -> None:
    if val is None:
        invalid.append(str(raw))
    elif 1 <= val <= page_count:
        positions.append(val)
    else:
        invalid.append(val)


def _parse_page_number(raw: str, *, chinese: bool) -> int | None:
    token = _normalize_query(raw)
    token = re.sub(r"^(?:第)\s*", "", token)
    if re.match(r"^\d+(?:st|nd|rd|th)$", token):
        token = re.sub(r"(?:st|nd|rd|th)$", "", token)
    if token.isdigit():
        return int(token)
    if not chinese and token in _EN_NUM_WORDS:
        return _EN_NUM_WORDS[token]
    if chinese:
        return _parse_chinese_number(token)
    return None


def _parse_chinese_number(token: str) -> int | None:
    if not token:
        return None
    if token.isdigit():
        return int(token)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    total = 0
    current = 0
    has_num = False
    for ch in token:
        if ch.isdigit():
            current = current * 10 + int(ch)
            has_num = True
        elif ch in digits:
            current = digits[ch]
            has_num = True
        elif ch == "百":
            total += (current or 1) * 100
            current = 0
            has_num = True
        elif ch == "十":
            total += (current or 1) * 10
            current = 0
            has_num = True
        else:
            return None
    if not has_num:
        return None
    return total + current


def _dedupe(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _tokenize(text: str) -> list[str]:
    tok: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok.append("".join(cur))
                cur = []
            # keep CJK chars as single-character tokens for coarse matching
            if ch.strip() and ord(ch) > 0x2E00:
                tok.append(ch)
    if cur:
        tok.append("".join(cur))
    return [t for t in tok if len(t) >= 1]


__all__ = ["get_deck_outline", "locate_pages"]
