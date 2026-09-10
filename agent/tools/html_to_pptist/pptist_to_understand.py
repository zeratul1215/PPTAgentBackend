"""Adapter: a PPTist slide (JSON) -> a step1 ``understand_input`` ``plan_page``.

This is the reverse-direction bridge used by the JSON-driven reread. The
forward path bakes flow HTML into a PPTist slide; here we read a PPTist slide
back into the flat, deterministic text/image/palette baseline that step1's
UNDERSTAND stage consumes.

Deliberately narrow contract, mirroring what ``understand_step`` actually reads:

* ``blocks[]`` — one entry per on-slide text run (from ``text`` elements, from
  ``shape.text`` cards, and from every non-empty ``table`` cell). We emit only
  ``{text}``: step1 RE-TYPES ``kind`` purely from the rendered page image, and
  the JSON path carries no kind by design (parity with the pptx path). Reading
  order is top-to-bottom then left-to-right, so fragments arrive in natural
  reading order. No positional label is emitted (placement was retired).
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
from typing import Any

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _plain_text(rich: str) -> str:
    """PPTist rich HTML -> plain text. Paragraph/break tags become spaces so
    adjacent runs don't get glued into one word; other tags are dropped."""
    if not rich:
        return ""
    # Turn block/newline boundaries into spaces before stripping tags.
    t = re.sub(r"(?i)</p>|<br\s*/?>", " ", rich)
    t = _TAG_RE.sub("", t)
    t = _html.unescape(t)
    return _WS_RE.sub(" ", t).strip()


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


def _iter_text_fragments(elements: list[dict[str, Any]]) -> list[tuple[tuple[float, float], str]]:
    """Collect (sortkey, text) for every text-bearing element.

    Fragments are ordered by reading flow (top-to-bottom, then left-to-right);
    PPTist elements carry absolute geometry, not the semantic band labels step1
    used to consume, and step1 recomputes layout from the rendered image anyway,
    so no positional label is emitted.
    """
    out: list[tuple[tuple[float, float], str]] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        etype = el.get("type")
        if etype == "text":
            txt = _plain_text(str(el.get("content") or ""))
            if txt:
                out.append((_center(el), txt))
        elif etype == "shape":
            shape_text = el.get("text")
            if isinstance(shape_text, dict):
                txt = _plain_text(str(shape_text.get("content") or ""))
                if txt:
                    out.append((_center(el), txt))
        elif etype == "table":
            # Each non-empty cell is its own fragment, ordered by (row, col) but
            # anchored to the table's position so it sits correctly in the flow.
            base = _center(el)
            data = el.get("data")
            if isinstance(data, list):
                for r, row in enumerate(data):
                    if not isinstance(row, list):
                        continue
                    for c, cell in enumerate(row):
                        if not isinstance(cell, dict):
                            continue
                        txt = _plain_text(str(cell.get("text") or ""))
                        if txt:
                            out.append(((base[0] + r * 0.001, base[1] + c * 0.001), txt))
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
    blocks = [{"text": text} for _, text in fragments]

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
