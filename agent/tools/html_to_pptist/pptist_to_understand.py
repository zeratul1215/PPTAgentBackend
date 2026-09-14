"""Adapter: a PPTist slide (JSON) -> a step1 ``understand_input`` ``plan_page``.

This is the reverse-direction bridge used by the JSON-driven reread. Tables
remain native structures here; they are never duplicated into top-level text.

Deliberately narrow contract, mirroring what ``understand_step`` actually reads:

* ``blocks[]`` — one entry per on-slide text run (from ``text`` elements and
  ``shape.text`` cards). We emit only
  ``{text}``: step1 RE-TYPES ``kind`` purely from the rendered page image, and
  the JSON path carries no kind by design (parity with the pptx path). Reading
  order is top-to-bottom then left-to-right, so fragments arrive in natural
  reading order. No positional label is emitted (placement was retired).
* ``tables[]`` — complete PPTist-native table specs. The model receives a
  separate slim projection containing only cell text and spans.
* ``images[]`` — one entry per ``image`` element, ``{id, src}`` plus the
  element's on-canvas size as ``display_w_pt``/``display_h_pt`` (the PPTist px
  value, kept as-is in the shared 1000-wide space). This size is a SOFT
  reference only — downstream re-layout may freely rescale it. The page image
  for reread is still produced by the HTML->PDF->PNG render, so these are only a
  baseline list for step1's per-image description task.
* ``palette`` — ``{primary, fills[]}`` aggregated from element fills, most
  frequent first. Purely advisory context for step1.

The slide's text ``content`` is PPTist rich HTML (``<p><span>...</span></p>``);
we strip tags to plain text and never alter characters otherwise, so step1's
character-integrity anchor still holds.
"""

from __future__ import annotations

import html as _html
import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any

from agent_backend.agent.tools.table_spec import slim_table_for_model, table_from_pptist

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class _RichParagraphParser(HTMLParser):
    """Extract real ``p`` boundaries while concatenating inline runs.

    PPTist stores formatting as nested spans. Those spans are presentation
    details, not separate semantic paragraphs; a paragraph is always bounded
    by ``p`` (with ``br`` retained as an internal line break)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "p":
            if self._depth:
                self._finish()
            self._depth += 1
            self._parts = []
        elif tag.lower() == "br" and self._depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p" and self._depth:
            self._finish()
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._parts.append(data)

    def _finish(self) -> None:
        text = _html.unescape("".join(self._parts)).strip()
        if text:
            self.paragraphs.append(text)
        self._parts = []


def _rich_paragraphs(rich: str) -> list[str]:
    if not rich:
        return []
    parser = _RichParagraphParser()
    try:
        parser.feed(str(rich))
        parser.close()
    except Exception:
        parser.paragraphs = []
    if parser.paragraphs:
        return parser.paragraphs
    # Defensive fallback for malformed legacy HTML without p tags.
    t = re.sub(r"(?i)<br\s*/?>", "\n", str(rich))
    t = _TAG_RE.sub("", t)
    t = _html.unescape(t)
    return [x.strip() for x in re.split(r"\n+", t) if x.strip()]


def _plain_text(rich: str) -> str:
    """Flatten rich HTML without joining distinct paragraphs."""
    return " ".join(_rich_paragraphs(rich))


def _safe_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _norm_color(c: Any) -> str | None:
    if not isinstance(c, str):
        return None
    s = c.strip()
    return s.lower() if _HEX_RE.match(s) else None


def _center(el: dict[str, Any]) -> tuple[float, float]:
    """(top, left) of an element's top-left, used to order by reading flow.
    Rounded so runs on the same visual line sort left-to-right, not by sub-pixel
    top jitter."""
    try:
        top = float(el.get("top") or 0)
    except (TypeError, ValueError):
        top = 0.0
    try:
        left = float(el.get("left") or 0)
    except (TypeError, ValueError):
        left = 0.0
    return (round(top / 10.0), left)


def _iter_text_fragments(elements: list[dict[str, Any]]) -> list[tuple[tuple[float, float], dict[str, Any]]]:
    """Collect (sortkey, text) for every text-bearing element.

    Fragments are ordered by reading flow (top-to-bottom, then left-to-right);
    PPTist elements carry absolute geometry, not the semantic band labels step1
    used to consume, and step1 recomputes layout from the rendered image anyway,
    so no positional label is emitted.
    """
    out: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for el_index, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        etype = el.get("type")
        element_id = str(el.get("id") or el.get("elementId") or f"element_{el_index}")
        if etype == "text":
            rich = str(el.get("content") or "")
            for paragraph_index, txt in enumerate(_rich_paragraphs(rich)):
                out.append((_center(el), {"text": txt, "source_element_id": element_id,
                                          "paragraph_index": paragraph_index}))
        elif etype == "shape":
            shape_text = el.get("text")
            if isinstance(shape_text, dict):
                rich = str(shape_text.get("content") or "")
                for paragraph_index, txt in enumerate(_rich_paragraphs(rich)):
                    out.append((_center(el), {"text": txt, "source_element_id": element_id,
                                              "paragraph_index": paragraph_index}))
    return out


def _extract_tables(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract complete native tables; no vision/model inference is involved."""
    out: list[dict[str, Any]] = []
    for idx, el in enumerate(elements):
        if not isinstance(el, dict) or el.get("type") != "table":
            continue
        try:
            out.append(table_from_pptist(el, fallback_id=f"table_{idx}"))
        except ValueError:
            # A malformed table must not be converted into misleading text.
            # Keep a compact warning for the understanding model instead.
            continue
    return out


def _collect_palette(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate element fill colors into {primary, fills[]}, most frequent
    first. Text/inline font colors are ignored: step1 uses palette only as a
    coarse color-context hint, and fills dominate a slide's visual identity."""
    counter: Counter[str] = Counter()
    for el in elements:
        if not isinstance(el, dict):
            continue
        col = _norm_color(el.get("fill"))
        if col and col not in ("#ffffff", "#fff"):
            counter[col] += 1
        grad = el.get("gradient")
        if isinstance(grad, dict):
            for stop in grad.get("colors") or []:
                if isinstance(stop, dict):
                    gc = _norm_color(stop.get("color"))
                    if gc:
                        counter[gc] += 1
    if not counter:
        return {}
    ranked = [c for c, _ in counter.most_common()]
    return {"primary": ranked[0], "fills": ranked[:6]}


def slide_to_plan_page(slide: dict[str, Any], *, page_id: str = "") -> dict[str, Any]:
    """Build a step1 ``plan_page`` from one PPTist slide."""
    elements = slide.get("elements")
    if not isinstance(elements, list):
        elements = []

    fragments = _iter_text_fragments(elements)
    fragments.sort(key=lambda f: f[0])
    blocks: list[dict[str, Any]] = []
    block_by_key: dict[tuple[str, int], str] = {}
    for idx, (_, fragment) in enumerate(fragments):
        block_id = f"b{idx}"
        block = {"id": block_id, **fragment}
        blocks.append(block)
        block_by_key[(f"{fragment['source_element_id']}:{fragment['paragraph_index']}", 0)] = block_id

    tables = _extract_tables(elements)

    images: list[dict[str, Any]] = []
    for i, el in enumerate(elements):
        if isinstance(el, dict) and el.get("type") == "image":
            src = str(el.get("src") or "").strip()
            if src:
                img: dict[str, Any] = {"id": f"{page_id or 'p'}_i{i}", "src": src}
                # Carry the element's on-canvas size (PPTist is a shared 1000-wide
                # px space; we keep the px value under display_w_pt/h_pt without
                # unit conversion, matching the rest of this path). This is a
                # SOFT reference size only — downstream re-layout may freely
                # rescale it; it exists so step2/step3 have a real aspect ratio
                # and a sane starting size instead of guessing.
                dw = _safe_float(el.get("width"))
                dh = _safe_float(el.get("height"))
                if dw is not None and dw > 0:
                    img["display_w_pt"] = round(dw, 2)
                if dh is not None and dh > 0:
                    img["display_h_pt"] = round(dh, 2)
                images.append(img)

    plan_page: dict[str, Any] = {
        "page_id": page_id,
        "layout_notes": "",
        "blocks": blocks,
    }
    if tables:
        plan_page["tables"] = tables
    palette = _collect_palette(elements)
    if palette:
        plan_page["palette"] = palette
    if images:
        plan_page["images"] = images
    return plan_page


def slide_to_understand_input(
    slide: dict[str, Any],
    *,
    page_num: int,
    page_png_path: str,
    page_size_pt: dict[str, Any] | None = None,
    bundle_dir: str | None = None,
) -> dict[str, Any]:
    """Assemble a full ``understand_input_v1`` object from one PPTist slide.

    ``page_png_path`` (the HTML->PDF->PNG render of the same page) and the
    optional ``bundle_dir`` (where image srcs resolve) are supplied by the
    caller; this adapter only owns the structured text/image/palette extraction.
    """
    page_id = f"page{int(page_num) - 1}"
    plan_page = slide_to_plan_page(slide, page_id=page_id)
    return {
        "schema_version": "understand_input_v1",
        "page_num": int(page_num),
        "page_size_pt": page_size_pt or {"w": None, "h": None},
        "bundle_dir": bundle_dir or "",
        "plan_page": plan_page,
        "page_png_path": page_png_path,
        "options": {
            "need_image_descriptions": True,
            "need_original_layout_description": True,
        },
    }


def _main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="PPTist slide JSON -> step1 understand_input")
    ap.add_argument("--slide", required=True, help="Path to a single PPTist slide JSON (one {id, elements, ...}).")
    ap.add_argument("--page-num", type=int, default=1)
    ap.add_argument("--page-png", default="", help="Path to the page render PNG (HTML->PDF->PNG).")
    ap.add_argument("--out", default="", help="Write understand_input JSON here (default: stdout).")
    args = ap.parse_args()

    slide = json.loads(Path(args.slide).read_text(encoding="utf-8"))
    # Accept either a bare slide or a { slide: {...} } / deck wrapper for convenience.
    if isinstance(slide, dict) and "elements" not in slide:
        if isinstance(slide.get("slide"), dict):
            slide = slide["slide"]
        elif isinstance(slide.get("slides"), list) and slide["slides"]:
            slide = slide["slides"][0]

    obj = slide_to_understand_input(
        slide,
        page_num=int(args.page_num),
        page_png_path=str(args.page_png),
    )
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    _main()
