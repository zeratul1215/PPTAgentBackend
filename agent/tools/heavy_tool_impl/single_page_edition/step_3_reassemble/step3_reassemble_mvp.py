"""
Step 3 (MVP): Reassemble a single page into HTML from Step 2 output.

Vendored into `agent_backend` so the LangGraph pipeline is self-contained.
"""

# (Vendored implementation.)

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx


DEFAULT_MODEL = os.getenv("PPT_LLM_MODEL", "claude-opus-4-8")


_SYSTEM_PROMPT_BASE = """You are a “single-page HTML code generator”.

You will receive:
- `page_state` (JSON): page meta + content. Includes `page_size_pt`, `palette` (primary + fills; only a hint),
  `texts[]` (id/kind/text/segments), `images[]` (id/src/display sizes/description_en),
  `tables[]` (OPTIONAL, authoritative table structure: each `{id, rows, cols, cells:[{row,col,ref}]}` — see the table
  rules), and `required_refs` (the ONLY allowed reference ids).
- `layout_notes_brief` (plain text): a short skeleton-level layout description for THIS page (secondary hint).
- `css_library_doc` (plain text): the base CSS that is ALWAYS loaded. It only fixes the slide size (`.page` =
  959.76pt x 540pt) and the Office-friendly font stack. It provides NO visual components; you build everything else.
- `reference_image` (image): the VISUAL TARGET for this page. Reproduce its layout AND its visual detail as closely
  and as beautifully as you can. See "How to use reference_image" for the exact rules.

Your task:
- Generate the HTML for THIS page only, reproducing `reference_image` as faithfully and beautifully as possible.

Output requirements (VERY IMPORTANT):
- Output HTML only (no explanation, no markdown, no ``` fences).
- Your output MUST start with `<!-- PAGE_START -->` and MUST end with `<!-- PAGE_END -->`.
  Do NOT output any characters outside these two markers.
- Required structure:
  <!-- PAGE_START -->
  <div class="page" id="..."> ... </div>
  <!-- PAGE_END -->

Styling freedom (you have FULL creative control):
- You MAY write your own CSS in a single `<style>` block placed as the FIRST child inside `<div class="page">`.
- You MAY use flex, grid, transforms (scale/rotate/skew), and any colors you judge best to match the reference image.
- You MAY use `px`, `pt`, `%`, `rem`, `em` freely.
- Goal: match the reference image's look as closely as possible while producing clean HTML/CSS that a browser renders
  exactly as written AND that converts cleanly into an editable PowerPoint later (see the SVG rule below).

Shape technique — draw colored/non-text graphics as INLINE SVG (IMPORTANT):
- Every COLORED / NON-TEXT graphic MUST be drawn as an inline `<svg>` using vector primitives: colored blocks/bands,
  cards/bubbles, circles, dots, triangles/pointers, connector lines, timeline axes, dividers, arrows, decorative
  geometry. Use `<rect>` (with `rx` for rounded corners), `<circle>`, `<ellipse>`, `<polygon>`, `<path>`, `<line>`.
  Put the shape's color in the SVG `fill`/`stroke`. You MAY use SVG gradients (`<linearGradient>`) and SVG filters
  (`<feDropShadow>`) INSIDE the svg when the reference clearly shows a gradient/shadow.
- Do NOT draw shapes with CSS instead: NO CSS `background-color` color blocks/cards, NO `border`-triangle hacks,
  NO CSS `box-shadow`, NO CSS gradients for shapes, NO `border-radius` used to fake a pill/rounded card. Those CSS
  effects get rasterized/locked when converted to PowerPoint; SVG shapes convert to native editable shapes.
  (Exception: a real `<table>`'s cell/row/header background fill and cell borders MAY use plain CSS — see the
  tabular-data rule below — because they map to native editable PowerPoint table cells.)
- Each `<svg>` MUST declare a `viewBox` and use coordinates RELATIVE to that viewBox (a small local coordinate
  system). Never use page-level coordinates inside the SVG.
- Prefer ONE primitive per `<svg>` (IMPORTANT): whenever a graphic can be drawn as a SINGLE primitive, an `<svg>`
  MUST contain exactly one drawing element (one `<rect>` / `<circle>` / `<ellipse>` / `<polygon>` / `<path>` /
  `<line>`) and nothing else. A single-primitive svg converts to a fully EDITABLE native shape; a multi-primitive svg
  must be flattened to a locked picture, so only use two-or-more primitives when the shape genuinely cannot be
  expressed as one (and never to fake an icon — icons are still forbidden).
  * Merge a body + its pointer/tail into ONE `<path>` instead of a rect plus a separate triangle/polygon. A speech
    bubble, callout, tab, or arrow-box is a single `<path>` that traces the whole outline (rounded corners via `Q`,
    the tail as a couple of extra `L` segments). Example bubble in a `0 0 600 340` viewBox:
    `<path d="M40,4 L590,4 Q600,4 600,14 L600,326 Q600,336 590,336 L40,336 Q30,336 30,326 L30,196 L4,170 L30,144 L30,14 Q30,4 40,4 Z" fill="#1f4064"/>`
  * Use `<path>` with `M`/`L`/`Q`/`A`/`Z` (straight sides `L`, rounded corners `Q`, arcs `A`, close with `Z`) to
    express rounded cards, pills, chevrons, banners-with-notch, and arrows as one outline.
  * Do NOT stack a background `<rect>` plus decorative `<rect>`/`<circle>` accents in the same `<svg>` just because
    they touch — give each its OWN single-primitive `<svg>` (each becomes its own editable shape), unless they must
    read as one indivisible graphic.
  * Only fall back to multiple primitives in one `<svg>` when the shape truly needs it (e.g. a ring/donut, a shape
    with a real hole, or overlapping cut-outs that one `<path>` cannot capture).
- Text is ALWAYS real HTML text rendered ON TOP of the SVG — never `<text>` inside the SVG, never text as paths,
  never text baked into an image. Text must stay selectable/editable.
- SVG is for SIMPLE NATIVE SHAPES ONLY — NEVER for detailed icons/pictograms. Use SVG only for the basic PPT-style
  geometry listed above (rectangles/rounded cards, circles/dots, ellipses, triangles/pointers, straight or elbow
  connector lines, timeline axes, dividers, plain arrows, color blocks/bands). You MUST NOT try to reconstruct a
  detailed ICON, GLYPH, LOGO, or PICTOGRAM as SVG — e.g. a chart-line icon, a yen/currency glyph, a network/hub
  icon, a gear, a shield, a robot, a warning-triangle-with-mark, a person/handshake, or any multi-`<path>`
  illustrative symbol. Reconstructed icons look bad and are forbidden. An icon may ONLY appear on the page when it
  is provided as a real image file in `page_state.images[]` (rendered via `<img src=...>`). If a wanted icon has NO
  such image file, OMIT it entirely — draw nothing, leave the space empty, do not fake it with SVG, an emoji, a
  text character, or a CSS shape. A single plain color dot/circle/bar used purely as a bullet or decorative accent
  is still fine (that is a native shape, not an icon).

How to overlay HTML text on an SVG shape WITHOUT computing page coordinates:
- Give the shape's container `position: relative` and let normal flow / flex / grid place that container.
- Put the SVG as a FILL-PARENT background layer inside it: `position:absolute; inset:0; width:100%; height:100%;`
  with `preserveAspectRatio="none"` so it stretches to the container. This LOCAL fill-parent absolute is the ONLY
  allowed use of absolute positioning.
- Put the text as a normal in-flow child (with padding) so it sits above the SVG (use `z-index` if needed).
- CRITICAL — the `position:absolute; inset:0` SVG MUST be a DIRECT child of that `position:relative` container, and
  the CSS selector that applies the absolute rule MUST actually match that SVG in the DOM. A fill-parent / divider /
  background SVG that FAILS to become absolute (wrong selector, or not a direct child of a relative parent) drops
  into normal flow; with `preserveAspectRatio="none"` + a tall `viewBox` + `height:100%` it then balloons to
  thousands of px tall and shoves every following element off the page. Double-check: for every absolute SVG, the
  selector (e.g. `#page-x .colleft > svg.divider`) resolves to a real element, and its parent is `position:relative`.
  Prefer targeting the SVG directly (`#page-x .divider`) over a descendant path (`#page-x .cn .divider`) that only
  works if the SVG sits inside another element.
- When a shape WRAPS its own label/text (a bubble, pill, card, callout — the text belongs INSIDE that one shape),
  mark the shape container with `data-group="1"`. This tells the editor to treat the shape + its text as ONE atomic
  object (they move together and the text can't be dragged out of the shape; the text stays editable). On that same
  container use flex centering so the text sits centered on the shape: `display:flex; align-items:center;
  justify-content:center; text-align:center;` (drop `justify-content:center` and keep `align-items:center` if the
  text should be left-aligned inside the shape). Do NOT put `data-group` on large layout regions (bands, columns,
  grids) — only on a single shape that owns its text.

Text sizing and editable alignment:
- Use compact, legible text sizes as a starting point: main title usually 26-36pt; section/card title 20-28pt;
  subtitle 14-20pt; body 12-18pt; caption/footnote 10-12pt. Title line-height is usually 1.05-1.2; body
  line-height is usually 1.2-1.5. These are soft guides: follow the reference image and content needs when they differ.
- Every text-bearing container should use a single-value padding declaration such as `padding: 12pt;`. Do NOT use
  asymmetric padding, `margin-left`, repeated spaces, or a narrowed text box to fake left/center/right alignment.
- For right-aligned text, give the text container the full intended width and set `text-align:right`; do NOT use
  `align-items:flex-end` or a shrink-wrapped text box to simulate right alignment.
- Mark logical text regions for Step4 autoshrink with `data-autofit-group="header"` / `"body"` / `"card-1"` etc.
  Use `data-autofit-sync="body-columns"` on peer regions that must keep the same final text scale. Do not nest
  `data-autofit-group`; if a page has no obvious grouping, put all ordinary text in one fallback group.

Hard constraints (MUST follow — these are the ONLY styling limits):
- Absolute positioning is allowed ONLY for a fill-parent SVG background layer, and ONLY in this exact form:
  `position:absolute; inset:0;` (with `width:100%; height:100%`). You MUST NOT use `position:absolute` for anything
  else, and you MUST NOT use `top:` / `left:` / `right:` / `bottom:` offsets to position elements. Build the entire
  layout with normal flow / flex / grid. (Use `position:relative` only on a shape container that holds a fill-parent
  SVG; avoid `sticky`/`fixed`.)
- Content images MUST NOT use `border-radius` (no rounded corners on `<img>`): a rounded image becomes a clipped,
  locked object in PowerPoint. Keep real photos as plain square-cornered `<img>`.
- `<style>` MUST be scoped: EVERY selector inside your `<style>` block MUST be prefixed with the page's own id
  (e.g. `#page-xyz .card { ... }`, `#page-xyz .title { ... }`). Never write a bare/global selector (like `.card {}`
  or `div {}`) because multiple pages share one document and would collide. `@keyframes` / `@font-face` at-rules are
  allowed as-is. Do NOT reference or generate any external CSS file.
- NO scrolling/clipping to hide content: do NOT use `overflow:auto/scroll` (including overflow-x/y). All content MUST
  be visible within the one fixed-size page.
- Images MUST use real `<img src="...">` with the exact `page_state.images[].src`. Do NOT invent image files. Do NOT
  use `background-image`/`url()` (draw decorative shapes/gradients as inline SVG instead). Real photos MUST NOT have
  `border-radius` (rounded images get clipped and locked in PowerPoint) — keep them square-cornered.
  - Image sizing (IMPORTANT): match each image's size and the space it fills to `reference_image`. If the image looks
    large / full-width / edge-to-edge in the reference, render it that big here too — do NOT default to a small, timid
    image that leaves empty space around it.
  - Give every content `<img>` an explicit, positive rendered width AND height (directly or through a parent with a
    definite height plus `height:100%`). Do NOT rely on `height:auto`, intrinsic sizing, or an indefinite flex/grid
    height: HTML is converted to editable PPTist geometry and zero/auto dimensions cannot be preserved.
  - Fill vs. distortion trade-off: when the image's real aspect ratio does not match its slot, you MAY crop it moderately
    (`object-fit: cover`) and/or stretch it slightly (keep width/height scale within ~15% of each other) to fill the slot
    and avoid empty bars. Never crop so hard that the main subject or meaningful content (charts, faces, labels) is lost.
    When in doubt between a big empty bar and a small crop, choose the small crop.
  - Frame sizing rule: if you wrap an image in a visible frame/background, size the frame to the image's aspect ratio
    (use `page_state.images[].display_w_pt` / `display_h_pt`) so there are no large empty bars.
- Text MUST be used verbatim: every `page_state.texts[].text` MUST be copied character-for-character into the page.
  Do NOT rewrite, translate, summarize, or add any extra text.
  IMPORTANT: each text item may also include `page_state.texts[].segments` (an ordered array of strings). When present,
  `segments` is the source of truth for the text's internal structure (lines/items). The displayed content for a text id
  MUST equal the EXACT concatenation of its `segments` (no separators inserted). If `segments` has multiple elements,
  you MUST render each segment as its own child element (e.g. multiple `<div>` rows) INSIDE the single container element
  that carries `data-ref="tN"`, in order. Do NOT add bullets/markers that are not already in the segment text.
  (Exception: in a DECLARED `page_state.tables[]` column — see the table rules — a text's `data-ref="tN"` is instead
    placed on that text's `<td>` cell in every row, one segment per cell; the per-row cells collectively cover all its
    segments.)
- No extra text:
  - The page may contain ONLY the text provided by `page_state.texts[].text`.
  - Especially forbidden: do NOT read/transcribe/translate any text from `reference_image` or from images
    (including `images[].description_en`). No OCR of image text into the page.
  - If some information exists only in the image (e.g. names, table/chart labels), keep it as image content. Do NOT retype it.
- Bilingual rule (applies when a single `texts[].text` already contains both Chinese and an English line/paragraph):
  Render Chinese first, then English immediately below. Differentiate by font size, but do NOT use a lighter text color
  for the English part. Body English >= 80% of Chinese body size; Title English >= 50% of Chinese title size.
- Naming / references:
  - In HTML, you may ONLY refer to items using `data-ref="<id>"` where `<id>` is from `page_state.required_refs.all`
  (text ids look like `tN`, image ids look like `pX_iY`).
  - Do NOT output any internal layout IDs (e.g. `floating:top_left`, `p3_c1`).
- Coverage requirement:
  - Every id in `page_state.required_refs.all` MUST appear as a `data-ref` value in the page.
  - Each text corresponds to one container element carrying `data-ref="tN"`; each image to one container or `<img>`
    carrying `data-ref="pX_iY"`. Use each id exactly ONCE.
  - Put a text's `data-ref` on the BLOCK/FLEX/GRID container whose full rendered border box should become the editable
    PPTist text box. Never put it only on an inner inline `<span>` or another shrink-wrapped styling child. Inner spans
    may carry font/color emphasis, but the outer `data-ref` container owns width, padding, wrapping, and alignment.
  - THE ONLY EXCEPTION is a DECLARED `page_state.tables[]` column (see the table rules below): when a text node is a
    column of a declared `<table>`, its `data-ref="tN"` is placed on that text's cell in EVERY row, so it repeats once
    per row. This is allowed ONLY when every occurrence of that id is a `<td>`/`<th>` inside the same `<table>`. Outside
    that case, never repeat a `data-ref`.
  - Do NOT use any `data-ref` values outside `page_state.required_refs.all`. Do NOT put the same `data-ref` twice on the
    same element.
- Space-first sizing (IMPORTANT): shrinking the font is the LAST resort, NOT the first move. Text should fill the space it
  is given and match the visual density of `reference_image`; do NOT pre-emptively pick a small/timid font "to be safe".
  When a text block is long, fit it WITHOUT shrinking first by: spreading it vertically to use the block's full height,
  splitting it into 2+ columns (via grid/flex containers), widening its region, or tightening margins/padding and
  line-height. Only after those are exhausted should you reduce the font, and then only just enough to avoid overflow —
  never smaller than needed. Never hide content.
- Long-list rule: when a single `texts[].text` is long and clearly enumerates multiple items
  (e.g. it starts with repeated date/year markers like "2012.06 ... 2013.10 ... 2014.07 ..."), do NOT dump it into one
  paragraph. Split it visually across multiple containers / lines so the structure is readable. The text content itself
  must still be used verbatim (no rewriting); the split is purely visual.
  Note: if `texts[].segments` is present and has multiple elements, that split has ALREADY been provided — use it directly
  and do not guess.
- If you implement "two columns" for text, implement it via grid/flex (two containers), NOT via CSS multi-column properties
  (`columns`, `column-count`, `column-width`, etc.).
- Real tables — use a real `<table>` ONLY for an explicit `page_state.tables[]` entry. `tables[]` is authoritative:
  when it is empty, you MUST NOT emit `<table>`, `<tr>`, `<td>`, or `<th>` anywhere on the page. A timeline, dated
  achievement list, label-and-description list, text over an SVG shape/card, or any other ordinary text structure is
  NOT a table merely because it has visually aligned columns. Render those with normal HTML text containers
  (`div`/`p`/`span`) using flex or grid; their text must remain ordinary PPTist text or shape-contained text.
  * DECLARED table (AUTHORITATIVE): when `page_state.tables[]` is present, each entry `{id, rows, cols, cells:[{row,col,ref}]}`
    is an EXPLICIT instruction: build exactly ONE `<table>` with that many `rows`x`cols`. Emit one `<tr>` per row and, in
    each row, one `<td>`/`<th>` per column in `col` order; put a cell's `ref` as the `data-ref` on that `<td>`/`<th>`.
    When several cells in the same column share one `ref` (a text node with one segment per row), that column's `ref`
    repeats once per row — place that node's row-th `segment` in the `<td>` for that row, so the segments cover the
    column top-to-bottom in order and concatenate back to the verbatim text. The `<tr>` element is what keeps columns
    aligned even when a cell wraps to multiple lines. NEVER split a declared table into multiple `<table>` elements or
    parallel `<div>` columns, and never align separate tables by matching row heights. This overrides your visual guess.
  * SINGLE-node table (one text): put `data-ref="tN"` on the `<table>` element. If `segments` is present, render EACH
    segment as its own `<tr>` (in order), splitting the segment's parts across `<td>` cells; the concatenation of the
    row texts MUST still equal the verbatim text (no added/removed characters, no invented column labels beyond what the
    text already contains).
  * Table cell / row / header backgrounds (header shading, zebra striping, a solid cell fill) are the ONE case where a
    CSS `background`/`background-color` IS allowed — cell fills map to native editable PowerPoint table cell fills, so
    they do NOT need to be inline SVG. All OTHER colored graphics on the page still follow the inline-SVG rule.
  * Cell borders / gridlines use normal CSS `border` on `<td>`/`<th>` (this is a table, not a faked shape).
  * One style per cell: all text inside a single `<td>`/`<th>` MUST share the same text style (same font-size, color,
    weight, italic, alignment). A native PowerPoint table cell stores one text style, so do NOT mix sizes/colors or wrap
    only part of a cell's text in a differently-styled `<span>`/`<strong>`. If two runs need different styles, they
    belong in different cells, not the same one.
  * Cells must WRAP, never overflow: a `table-layout:fixed` column has a fixed width, so a long unbreakable token
    (e.g. a slash-joined run like `SAP/ServiceNow/OutSystems`) will spill past the cell border. Do NOT use
    `white-space:nowrap` on cells; let cell text wrap freely (the base CSS already forces `overflow-wrap:break-word`
    on `<td>`/`<th>`). If a cell's content is long, the row simply grows taller — that is fine and expected.
"""


_SYSTEM_PROMPT_TAIL = """
How to use reference_image (the VISUAL TARGET):
- `reference_image` is the layout you must reproduce. It is either a redesigned mockup (when the user asked to change
  the layout) or a render of the original page (otherwise). In BOTH cases your job is the same: rebuild it as HTML.
- Reproduce its layout and visual detail as closely as the hard constraints allow. Match, as faithfully as you can:
  * spatial structure: region division, number of rows/columns, left/right or top/bottom splits, and the relative
    position of every element;
  * relative sizes and proportions: how big each block/column/image is compared to the others, and their alignment;
  * image handling: each image's relative size, aspect ratio, and whether it sits in a frame/card or bleeds full-width;
  * color blocks & hierarchy: background bands, cards, accent panels, gradients, and the visual weight/emphasis of each element;
  * rhythm: whitespace, alignment, and the tightness/looseness of spacing.
- You do NOT need pixel-perfect equality, but DO aim for the same overall composition and the same level of detail.
- The following CONTENT rules always OVERRIDE the reference image (the image is only a visual guide):
  * Text is ALWAYS taken verbatim from `page_state.texts[].text` / `segments`. The reference image's text may be
    rewritten, shortened, or slightly wrong — never OCR or copy text from the image. Use the image only to decide
    WHERE a text block goes and how big it is.
  * Images are ALWAYS the real `page_state.images[].src`. The picture drawn in the reference image is only a guide for
    placement and size; never treat it as the actual image content.

Color usage:
- Match the colors shown in `reference_image` as closely as you can — use its exact accent colors, background bands,
  gradients, and card colors. `page_state.palette` is only a fallback hint; when it conflicts with the reference image,
  FOLLOW THE REFERENCE IMAGE.
- Remember colored blocks/bands/cards/gradients are drawn as inline SVG (fill/stroke/`<linearGradient>`), NOT as CSS
  backgrounds. Only text color, the page's base background, and `<table>` cell/row/header fills use plain CSS.

Layout plan hints (secondary; the reference image is the primary target):
- `layout_notes_brief` is a one-line summary. If it explicitly names a structure ("left/right", "two-column", "grid",
  "band", "timeline", ...) honor it; if it conflicts with `reference_image`, follow the image.

Readability / contrast:
- Ensure strong contrast between text and its background container. Dark background -> light text; light background ->
  dark text. Avoid low-contrast combos.

Long-list / repeated-item rule (in addition to the base long-list rule):
- When a single text fragment is a chronological / numbered list, render the items as multiple sibling containers
  (repeated cards / list rows), NOT as one big paragraph.
"""


_SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE + _SYSTEM_PROMPT_TAIL


_FENCE_RE = re.compile(r"```(?:html)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)

_WHITE_BG_RE = re.compile(
    r"(?i)\bbackground(?:-color)?\s*:\s*(?:#fff(?:fff)?|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)|rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*1(?:\.0+)?\s*\))\s*;?"
)


def _normalize_model_output(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return ""
    m = _FENCE_RE.search(t)
    if m:
        return (m.group(1) or "").strip()
    return t


def _normalize_page_background_vars(page_block: str) -> str:
    t = page_block or ""
    m = re.search(r'(?is)<div\b[^>]*\bclass\s*=\s*"[^"]*\bpage\b[^"]*"[^>]*>', t)
    if not m:
        return t
    tag = m.group(0)
    sm = re.search(r'(?is)\bstyle\s*=\s*"([^"]*)"', tag)
    if not sm:
        return t
    style0 = sm.group(1) or ""
    style1 = _WHITE_BG_RE.sub("background: var(--paper);", style0)
    if style1 == style0:
        return t
    new_tag = tag[: sm.start(1)] + style1 + tag[sm.end(1) :]
    return t.replace(tag, new_tag, 1)


def _resolve_backend(*, api_key_arg: str) -> tuple[str, str]:
    base_url = (os.getenv("PPT_LLM_BASE_URL") or "").strip()
    if not base_url:
        raise SystemExit(
            "Missing PPT_LLM_BASE_URL. Set PPT_LLM_BASE_URL to an OpenAI-compatible endpoint (e.g. https://.../v1)."
        )
    key = (api_key_arg or os.getenv("PPT_LLM_API_KEY") or "").strip()
    if not key:
        raise SystemExit("Missing API key. Provide --api-key or set PPT_LLM_API_KEY.")
    return base_url, key


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3 (MVP): reassemble per-page HTML from Step 2 output.")
    p.add_argument("--step2-outputs-dir", type=str, required=True, help="Directory containing *.step2_output.json files.")
    p.add_argument("--out-bundle-dir", type=str, required=True, help="Output bundle directory.")

    p.add_argument("--css-dir", type=str, default="", help="Directory containing css_library.css + css_library.md.")
    p.add_argument("--images-dir", type=str, default="", help="Directory containing extracted images/ (symlinked into output).")
    p.add_argument("--pages-png-dir", type=str, default="", help="Directory containing page_XXX.png renders (optional).")
    p.add_argument(
        "--beautify-refs-dir",
        type=str,
        default="",
        help="Directory containing beautify reference images (page_XXX.png / page_XXX.gen.png). "
        "When a page has one, it is used as the layout target instead of the original render.",
    )

    p.add_argument("--pages", type=str, default="", help="Comma-separated 1-based page numbers to run (e.g. 2,4,10).")
    p.add_argument("--dry-run", action="store_true", help="Do not call the model; only write request bundles.")
    p.add_argument("--print-prompt", action="store_true", help="Print system+user prompt for the first selected page then exit.")

    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens (default: 8192)")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0)")
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=-1,
        help="Thinking budget (ignored for OpenAI-compatible backends). Use 0 to disable thinking. Negative means not set (default: -1).",
    )
    p.add_argument("--retries", type=int, default=6, help="Retry count on transient 5xx/429 (default: 6)")
    p.add_argument("--retry-base-seconds", type=float, default=2.0, help="Base sleep seconds (default: 2.0)")
    p.add_argument("--retry-max-seconds", type=float, default=60.0, help="Max sleep seconds (default: 60.0)")
    p.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier (default: 2.0)")
    p.add_argument("--max-concurrency", type=int, default=3, help="Max concurrent in-flight model calls (default: 3)")
    p.add_argument("--save-raw", action="store_true", help="Save raw model output per page.")

    p.add_argument("--api-key", type=str, default="", help="API key (avoid; prefer env PPT_LLM_API_KEY).")
    p.add_argument("--title", type=str, default="PPTAgent", help="HTML <title> text (default: PPTAgent)")
    return p.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _ensure_css_library(*, out_bundle_dir: Path, css_dir: Path) -> str:
    css_dir = css_dir.expanduser().resolve()
    out_bundle_dir = out_bundle_dir.expanduser().resolve()

    css_src = css_dir / "css_library.css"
    css_doc_src = css_dir / "css_library.md"
    if not css_src.exists():
        raise SystemExit(f"css_library.css not found: {css_src}")
    if not css_doc_src.exists():
        raise SystemExit(f"css_library.md not found: {css_doc_src}")

    css_dst = out_bundle_dir / "css_library.css"
    out_bundle_dir.mkdir(parents=True, exist_ok=True)

    css_src_txt = css_src.read_text(encoding="utf-8", errors="replace")
    if (not css_dst.exists()) or (css_dst.read_text(encoding="utf-8", errors="replace") != css_src_txt):
        css_dst.write_text(css_src_txt, encoding="utf-8")

    return css_doc_src.read_text(encoding="utf-8", errors="replace").strip()


def _image_file_exists(*, images_dir: Path, src: str) -> bool:
    """Whether the image referenced by `src` actually exists under images_dir.

    `src` is a bundle-relative path like `images/00.png`; the renderer symlinks
    the whole images dir, so an image resolves by its basename under
    `images_dir`. We match on basename (robust to `images/` prefixes) and also
    try the raw relative path against the images dir's parent as a fallback."""
    name = os.path.basename(src.strip())
    if not name:
        return False
    try:
        if (images_dir / name).exists():
            return True
        # Fallback: honor an explicit relative path (e.g. subdir/foo.png)
        rel = src.strip().lstrip("/")
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        return (images_dir / rel).exists()
    except OSError:
        return False


def _try_symlink_images(*, out_bundle_dir: Path, images_dir: Path) -> str | None:
    out_bundle_dir = out_bundle_dir.expanduser().resolve()
    images_dir = images_dir.expanduser().resolve()
    if not images_dir.exists():
        return f"images_dir_not_found: {images_dir}"

    dst = out_bundle_dir / "images"
    if dst.exists():
        return None
    try:
        dst.symlink_to(images_dir, target_is_directory=True)
        return None
    except Exception as e:
        return f"images_symlink_failed: {type(e).__name__}: {e}"

def _write_page_outputs(
    *,
    out_bundle_dir: Path,
    chunks_dir: Path,
    page_num: int,
    page_block: str,
    base_href: str,
    title: str,
    system_prompt: str,
    user_prompt: str,
    include_page_png: bool,
    step2_path: Path,
    debug: dict[str, Any],
    save_raw: bool,
) -> None:
    out_bundle_dir = out_bundle_dir.expanduser().resolve()
    chunks_dir = chunks_dir.expanduser().resolve()
    chunks_dir.mkdir(parents=True, exist_ok=True)

    page_dir = out_bundle_dir / f"page_{page_num:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        page_dir / "request.json",
        {
            "page_num": int(page_num),
            "step2_output_path": str(step2_path),
            "include_page_png": bool(include_page_png),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )

    _write_text(page_dir / "page_block.html", page_block)

    soft = debug.get("soft_warnings") if isinstance(debug.get("soft_warnings"), list) else []
    if soft:
        _write_json(page_dir / "warnings.json", {"warnings": soft})

    if save_raw:
        raw = debug.get("raw")
        if isinstance(raw, str) and raw:
            _write_text(page_dir / "raw.txt", raw)
        raw2 = debug.get("raw_repair_format")
        if isinstance(raw2, str) and raw2:
            _write_text(page_dir / "raw_repair_format.txt", raw2)
        raw3 = debug.get("raw_repair_hard")
        if isinstance(raw3, str) and raw3:
            _write_text(page_dir / "raw_repair_hard.txt", raw3)

    chunk_html = _wrap_chunk_html(page_block=page_block, base_href=base_href, title=title)
    chunk_path = chunks_dir / f"chunk_{page_num:03d}_{page_num:03d}.html"
    _write_text(chunk_path, chunk_html)


def _get_step2_page_num(step2: dict[str, Any]) -> int:
    um = (
        (((step2.get("compile") or {}).get("understand_modified")) if isinstance(step2.get("compile"), dict) else None)
        or {}
    )
    pno = um.get("page_num")
    try:
        return int(pno)
    except Exception:
        return 0


def _extract_page_state(step2: dict[str, Any], *, images_dir: Path | None = None) -> dict[str, Any]:
    compile_obj = step2.get("compile") if isinstance(step2.get("compile"), dict) else {}
    um = (compile_obj.get("understand_modified") if isinstance(compile_obj, dict) else None) or {}
    if not isinstance(um, dict):
        um = {}

    page_size_pt = um.get("page_size_pt") if isinstance(um.get("page_size_pt"), dict) else {}
    palette = um.get("palette") if isinstance(um.get("palette"), dict) else {}

    texts_in: list[Any] = um.get("texts") if isinstance(um.get("texts"), list) else []
    texts_out: list[dict[str, Any]] = []
    for t in texts_in:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        kind = t.get("kind")
        txt = t.get("text")
        segs_any = t.get("segments")
        if not isinstance(tid, str) or not tid:
            continue
        if not isinstance(txt, str):
            txt = ""
        segments: list[str]
        if isinstance(segs_any, list) and segs_any and all(isinstance(x, str) for x in segs_any):
            segments = [str(x) for x in segs_any]
        else:
            segments = [txt] if txt else []
        if not isinstance(kind, str):
            kind = ""
        texts_out.append(
            {
                "id": tid,
                "kind": kind,
                "text": txt,
                "segments": segments,
            }
        )

    images_in: list[Any] = um.get("images") if isinstance(um.get("images"), list) else []
    images_out: list[dict[str, Any]] = []
    for im in images_in:
        if not isinstance(im, dict):
            continue
        iid = im.get("id")
        src = im.get("src")
        if not isinstance(iid, str) or not iid:
            continue
        if not isinstance(src, str) or not src:
            continue
        # Drop images whose backing file is missing on disk: they can't be
        # rendered, and keeping them would force the model (via required_refs)
        # to emit an <img> pointing at a non-existent file. This is how an icon
        # with no original image gets IGNORED rather than faked. Only filters
        # when we actually know the images dir; otherwise keep prior behavior.
        if images_dir is not None and not _image_file_exists(images_dir=images_dir, src=src):
            continue
        out_im: dict[str, Any] = {"id": iid, "src": src}
        desc = im.get("description_en")
        if isinstance(desc, str) and desc:
            out_im["description_en"] = desc
        dw = im.get("display_w_pt")
        dh = im.get("display_h_pt")
        if isinstance(dw, (int, float)) and isinstance(dh, (int, float)):
            out_im["display_w_pt"] = float(dw)
            out_im["display_h_pt"] = float(dh)
        images_out.append(out_im)

    original_layout_desc_en = um.get("original_layout_description_en")
    if not isinstance(original_layout_desc_en, str):
        original_layout_desc_en = ""

    valid_text_ids = {t["id"] for t in texts_out}
    tables_in: list[Any] = um.get("tables") if isinstance(um.get("tables"), list) else []
    tables_out: list[dict[str, Any]] = []
    for tbl in tables_in:
        if not isinstance(tbl, dict):
            continue
        try:
            rows = int(tbl.get("rows"))
            cols = int(tbl.get("cols"))
        except (TypeError, ValueError):
            continue
        if rows <= 0 or cols <= 0:
            continue
        cells_in = tbl.get("cells")
        if not isinstance(cells_in, list):
            continue
        cells_out: list[dict[str, Any]] = []
        for cell in cells_in:
            if not isinstance(cell, dict):
                continue
            try:
                r = int(cell.get("row"))
                c = int(cell.get("col"))
            except (TypeError, ValueError):
                continue
            ref = str(cell.get("ref") or "")
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if ref not in valid_text_ids:
                continue
            cells_out.append({"row": r, "col": c, "ref": ref})
        if cells_out:
            tables_out.append(
                {
                    "id": str(tbl.get("id") or f"tbl{len(tables_out)}"),
                    "rows": rows,
                    "cols": cols,
                    "cells": cells_out,
                }
            )

    required_text_ids: list[str] = []
    required_image_ids: list[str] = []
    required_set: set[str] = set()
    for t in texts_out:
        tid = t["id"]
        if tid not in required_set:
            required_set.add(tid)
            required_text_ids.append(tid)
    for im in images_out:
        iid = im["id"]
        if iid not in required_set:
            required_set.add(iid)
            required_image_ids.append(iid)
    required_all = list(required_image_ids) + list(required_text_ids)

    primary = palette.get("primary") if isinstance(palette, dict) else None
    if not isinstance(primary, str):
        primary = ""
    fills_in = palette.get("fills") if isinstance(palette, dict) else None
    fills_out: list[str] = []
    if isinstance(fills_in, list):
        for c in fills_in:
            if isinstance(c, str) and c:
                fills_out.append(c)

    user_request = step2.get("user_request")
    if not isinstance(user_request, str):
        user_request = ""

    # Built-in default layout details a skill injected because the user gave no
    # visual requirement for that intent (e.g. default bilingual layout). We take
    # the step3 phrasing (`step3_text`), which is DOM/render discipline.
    default_layout_details: list[str] = []
    vi = step2.get("visual_intent")
    if isinstance(vi, dict):
        for d in vi.get("default_details") or []:
            if isinstance(d, dict):
                txt = d.get("step3_text")
                if isinstance(txt, str) and txt.strip():
                    default_layout_details.append(txt.strip())

    out: dict[str, Any] = {
        "page_size_pt": page_size_pt if isinstance(page_size_pt, dict) else {},
        "palette": {"primary": primary, "fills": fills_out},
        "texts": texts_out,
        "tables": tables_out,
        "images": images_out,
        "required_refs": {
            "images": required_image_ids,
            "texts": required_text_ids,
            "all": required_all,
        },
        "original_layout_description_en": original_layout_desc_en,
        "user_request": user_request,
        "default_layout_details": default_layout_details,
    }
    return out


def _has_layout_intent(step2: dict[str, Any]) -> bool:
    """Read the resolved visual signal produced by step2_run (top-level
    `visual_intent`), folded from the planner's request + any skill that
    requested a re-flow at runtime."""
    vi = step2.get("visual_intent")
    return bool(isinstance(vi, dict) and vi.get("enabled", False))


def _read_page_png_bytes(*, pages_png_dir: Path, page_num: int) -> bytes | None:
    if page_num <= 0:
        return None
    p = pages_png_dir / f"page_{page_num:03d}.png"
    if not p.exists():
        return None
    return p.read_bytes()


def _read_reference_image_bytes(*, beautify_refs_dir: Path | None, page_num: int) -> bytes | None:
    """Read a per-page beautify reference image if one exists.

    Looks for `page_XXX.png` (or `page_XXX.gen.png`) inside `beautify_refs_dir`.
    Returns None when no directory is configured or no matching file exists.
    """
    if beautify_refs_dir is None or page_num <= 0:
        return None
    for name in (f"page_{page_num:03d}.png", f"page_{page_num:03d}.gen.png"):
        p = beautify_refs_dir / name
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                return None
    return None


def _build_typography_guidance(deck_style: dict[str, Any] | None) -> str:
    if not isinstance(deck_style, dict):
        return ""
    typography = deck_style.get("typography") if isinstance(deck_style.get("typography"), dict) else {}
    title = typography.get("title") if isinstance(typography.get("title"), dict) else {}
    body = typography.get("body") if isinstance(typography.get("body"), dict) else {}
    title_font = str(title.get("fontFamily") or "").strip()
    body_font = str(body.get("fontFamily") or "").strip()
    if not title_font and not body_font:
        return ""
    return (
        "\n\ntypography_guidance_from_selected_deck_style:\n"
        f"- Recommended title font: {title_font or 'unspecified'}\n"
        f"- Recommended body font: {body_font or 'unspecified'}\n"
        "- These fonts are style recommendations, not hard constraints. Use them when they suit the page and help it remain consistent with the deck. You may retain or adapt typography when required by the user's explicit request, the language of the content, readability, or this page's visual hierarchy. Keep font usage restrained and internally consistent.\n"
    )


def _build_user_prompt(
    *,
    page_state: dict[str, Any],
    css_library_doc: str,
    has_layout_intent: bool,
    deck_style: dict[str, Any] | None = None,
) -> str:
    """Build the single (unified) user prompt.

    `layout_notes_brief` is a secondary hint (the reference image is the primary
    layout target): when a layout change was requested it is the user's ORIGINAL
    request (the same intent that drove the reference image); otherwise it is the
    original page description (keep the existing layout, only the text changed).
    """
    state_view: dict[str, Any] = dict(page_state)
    original_desc = str(state_view.pop("original_layout_description_en", "") or "")
    user_request = str(state_view.pop("user_request", "") or "").strip()
    default_layout_details = state_view.pop("default_layout_details", None)
    default_layout_details = [
        d.strip()
        for d in (default_layout_details or [])
        if isinstance(d, str) and d.strip()
    ]

    if has_layout_intent:
        layout_notes_brief = user_request or original_desc.strip()
        # When a beautify reference image drives the layout, each text's visual
        # role (title/body/...) is already carried by the image; `kind` becomes a
        # redundant hint that could bias the model away from the reference. Drop
        # it. (Text-only edits keep `kind` — they reproduce the original layout,
        # where the role label helps preserve the existing hierarchy.)
        texts_view = state_view.get("texts")
        if isinstance(texts_view, list):
            state_view["texts"] = [
                {k: v for k, v in t.items() if k != "kind"} if isinstance(t, dict) else t
                for t in texts_view
            ]
    else:
        layout_notes_brief = original_desc.strip()

    required_refs = (
        state_view.get("required_refs")
        if isinstance(state_view.get("required_refs"), dict)
        else {"all": []}
    )

    # Text-only turns (no layout intent) target a render of the CURRENT page,
    # but the new text can be larger than before (e.g. bilingual adds a whole
    # second language). Tell the model the reference image reflects the layout
    # BEFORE the text grew, so container sizes are a positional guide, not a
    # capacity limit — grow the container (within reason) rather than cram or
    # prematurely shrink the font.
    text_growth_note = ""
    if not has_layout_intent:
        text_growth_note = (
            "\n\ntext_volume_note:\n"
            "This is a text-only edit: reproduce the reference image's overall composition and element "
            "positions, BUT the text in `page_state.texts[]` may be LONGER than what the reference image "
            "shows (e.g. an added second language). Treat each block's size in the reference as a POSITION "
            "guide, NOT a capacity limit. When a block's text no longer fits its reference-sized container, "
            "first ENLARGE that container within reason (extend its height/width, use the surrounding "
            "whitespace, split into more rows) while keeping the overall layout recognizable; only shrink the "
            "font as a last resort. This applies equally to text sitting inside an SVG shape/bubble: enlarge "
            "the shape to fit its text rather than clipping or overflowing it. Never hide or truncate content.\n"
        )

    # Built-in default layout details (e.g. the default bilingual layout) injected
    # only when the user gave no visual requirement for that intent. They refine
    # HOW the changed content is placed; the user's own requirement (already in
    # layout_notes_brief) always takes precedence.
    default_layout_note = ""
    if default_layout_details:
        default_layout_note = (
            "\n\ndefault_layout_details (apply UNLESS layout_notes_brief above "
            "already specifies otherwise):\n"
            + "\n".join(f"- {d}" for d in default_layout_details)
            + "\n"
        )

    return (
        "page_state:\n"
        + json.dumps(state_view, ensure_ascii=False, indent=2)
        + "\n\nrequired_refs:\n"
        + json.dumps(required_refs, ensure_ascii=False, indent=2)
        + "\n\ncss_library_doc:\n"
        + (css_library_doc or "")
        + "\n\nlayout_notes_brief:\n"
        + (layout_notes_brief or "")
        + _build_typography_guidance(deck_style)
        + default_layout_note
        + text_growth_note
        + "\n"
    )


def _parse_pages_arg(pages: str) -> set[int]:
    s = (pages or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return {p for p in out if p > 0}


def _find_prep_root(start: Path) -> Path | None:
    p = start.expanduser().resolve()
    for _ in range(12):
        if (p / "pages_png").exists() and (p / "ref_html" / "images").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def _page_root_id(page_block: str) -> str | None:
    """Return the id of the root `<div class="page" id="...">`, if any."""
    m = re.search(
        r'(?is)<div\b[^>]*\bclass\s*=\s*"[^"]*\bpage\b[^"]*"[^>]*>', page_block or ""
    )
    if not m:
        return None
    idm = re.search(r'(?is)\bid\s*=\s*"([^"]+)"', m.group(0))
    return idm.group(1).strip() if idm and idm.group(1).strip() else None


def _check_style_scope(*, page_block: str) -> list[str]:
    """Every selector inside an inline `<style>` block MUST be scoped under the
    page's own id (e.g. `#page-x .card`), so multiple pages in one document do
    not collide. `@`-rules (media/keyframes/font-face) and keyframe steps
    (`0%`/`from`/`to`) are exempt. Returns hard-error strings (empty = ok)."""
    t = page_block or ""
    styles = re.findall(r"(?is)<style\b[^>]*>(.*?)</style>", t)
    if not styles:
        return []

    root_id = _page_root_id(t)
    if not root_id:
        return ["style_scope: page root has no id (cannot scope <style>)"]

    id_token = f"#{root_id}"
    bad: list[str] = []
    for css in styles:
        # Drop comments so `{`/`}` inside them do not confuse the split.
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        depth = 0
        # Walk block-by-block; only inspect selector prefaces at brace depth 0.
        for chunk in re.split(r"([{}])", css):
            if chunk == "{":
                depth += 1
                continue
            if chunk == "}":
                depth = max(0, depth - 1)
                continue
            if depth != 0:
                continue  # inside a declaration/at-rule body
            selector_group = chunk.strip()
            if not selector_group:
                continue
            for sel in selector_group.split(","):
                sel = sel.strip()
                if not sel:
                    continue
                low = sel.lower()
                if low.startswith("@"):
                    continue  # @media / @keyframes / @font-face ...
                if re.fullmatch(r"(?:from|to|\d+%)", low):
                    continue  # keyframe step selectors
                if id_token.lower() not in low:
                    bad.append(sel[:60])

    if bad:
        uniq = sorted(set(bad))
        return [
            f"style_scope: selectors not scoped under {id_token}: "
            f"{uniq[:20]}{' ...' if len(uniq) > 20 else ''}"
        ]
    return []


def _has_rounded_image(*, page_block: str) -> bool:
    """Detect `border-radius` applied to a content `<img>` (rounded images become
    clipped + locked objects when converted to PowerPoint). Two cases:
    1) inline `<img ... style="...border-radius...">`;
    2) a `<style>` rule whose selector targets `img` and sets a non-zero radius."""
    t = page_block or ""

    for tag in re.findall(r"(?is)<img\b[^>]*>", t):
        sm = re.search(r'(?is)\bstyle\s*=\s*"([^"]*)"', tag)
        if sm and _nonzero_border_radius(sm.group(1)):
            return True

    for css in re.findall(r"(?is)<style\b[^>]*>(.*?)</style>", t):
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        depth = 0
        selector = ""
        for chunk in re.split(r"([{}])", css):
            if chunk == "{":
                depth += 1
                continue
            if chunk == "}":
                depth = max(0, depth - 1)
                selector = ""
                continue
            if depth == 0:
                selector = chunk.strip()
            elif depth == 1:
                # `chunk` is the rule body for the last-seen selector group.
                if re.search(r"(?i)(?:^|[\s,>+~*.#\[])img\b", selector) and _nonzero_border_radius(chunk):
                    return True
    return False


def _nonzero_border_radius(decls: str) -> bool:
    """True if a CSS declaration text sets a non-zero border-radius."""
    for m in re.finditer(r"(?i)border(?:-[a-z]+)?-radius\s*:\s*([^;}\"]+)", decls or ""):
        val = m.group(1).strip()
        # Non-zero if any number in the value is not zero.
        nums = re.findall(r"-?\d*\.?\d+", val)
        if any(abs(float(n)) > 0 for n in nums):
            return True
    return False


def _iter_style_decl_segments(page_block: str) -> list[str]:
    """Yield every CSS declaration segment we should inspect for positioning:
    each rule body inside `<style>` blocks (brace depth 1) plus every inline
    `style="..."` attribute value. Comments in `<style>` are stripped first."""
    segments: list[str] = []
    t = page_block or ""

    for css in re.findall(r"(?is)<style\b[^>]*>(.*?)</style>", t):
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        depth = 0
        buf: list[str] = []
        for chunk in re.split(r"([{}])", css):
            if chunk == "{":
                depth += 1
                buf = []
                continue
            if chunk == "}":
                if depth == 1 and buf:
                    segments.append("".join(buf))
                depth = max(0, depth - 1)
                buf = []
                continue
            if depth >= 1:
                buf.append(chunk)

    for sm in re.findall(r'(?is)\bstyle\s*=\s*"([^"]*)"', t):
        segments.append(sm)

    return segments


def _check_positioning(*, page_block: str) -> list[str]:
    """Positioning policy:
    - `position:absolute` is allowed ONLY as a fill-parent layer, i.e. the same
      declaration block must also set `inset` to 0 (the SVG background technique).
      Any other absolute is a hard error.
    - `top/left/right/bottom` positional offsets are never allowed (the `inset`
      shorthand is used for fill-parent instead, so it is exempt).
    Returns hard-error strings (empty = ok)."""
    hard: list[str] = []

    _INSET_ZERO_RE = re.compile(r"(?i)\binset\s*:\s*0(?:px|pt|%)?\s*(?:!important)?\s*;?")
    _ABS_RE = re.compile(r"(?i)position\s*:\s*absolute")

    bad_abs = 0
    for seg in _iter_style_decl_segments(page_block):
        if _ABS_RE.search(seg) and not _INSET_ZERO_RE.search(seg):
            bad_abs += 1
    if bad_abs:
        hard.append(
            "forbidden_css: position:absolute is only allowed as a fill-parent SVG layer "
            "(must also set `inset:0`)"
        )

    # Positional offsets are banned everywhere. The negative lookbehind for "-"
    # keeps `border-top` / `padding-left` / `border-bottom` allowed.
    if re.search(r"(?<!-)\btop\s*:", page_block, flags=re.I) or re.search(r"(?<!-)\bleft\s*:", page_block, flags=re.I):
        hard.append("forbidden_css: top/left (positional offsets are not allowed; use inset:0 for SVG fill)")
    if re.search(r"(?<!-)\bright\s*:", page_block, flags=re.I) or re.search(r"(?<!-)\bbottom\s*:", page_block, flags=re.I):
        hard.append("forbidden_css: right/bottom (positional offsets are not allowed; use inset:0 for SVG fill)")

    return hard


_TABLE_RE = re.compile(r"(?is)<table\b[^>]*>.*?</table>")
_CELL_RE = re.compile(r"(?is)<(?:td|th)\b[^>]*\bdata-ref\s*=\s*\"([^\"]+)\"[^>]*>")


def _is_table_column_ref(page_block: str, ref: str) -> bool:
    """A repeated `data-ref` is legal ONLY when it marks one column of a single
    <table>: every one of its occurrences sits on a <td>/<th> cell, and all of
    those cells belong to the SAME <table>. This is how two text nodes are
    aligned row-by-row inside one table (each column = one data-ref, repeated
    once per row). Returns True when `ref` satisfies that; False otherwise
    (cross-table repetition, occurrences on non-cell elements, or a mix)."""
    total = len(re.findall(r'data-ref\s*=\s*"' + re.escape(ref) + r'"', page_block))
    if total <= 1:
        return True

    tables = list(_TABLE_RE.finditer(page_block))
    for tbl in tables:
        cell_refs = _CELL_RE.findall(tbl.group(0))
        in_this_table = sum(1 for r in cell_refs if r == ref)
        # All occurrences of `ref` must be cells of THIS one table.
        if in_this_table == total:
            return True
    return False


def _validate_and_normalize_page_block(
    *, page_state: dict[str, Any], page_block: str
) -> tuple[str, list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []

    t0 = page_block or ""
    t = t0

    if 'data-ref=""' in t:
        soft.append("soft_fix: stripped empty data-ref attributes")
        t = re.sub(r'\s+data-ref\s*=\s*""', "", t)

    t1 = _normalize_page_background_vars(t)
    if t1 != t:
        soft.append("soft_fix: normalized page root background to var(--paper)")
        t = t1

    required_all = (
        ((page_state.get("required_refs") or {}).get("all") or [])
        if isinstance(page_state.get("required_refs"), dict)
        else []
    )
    if not isinstance(required_all, list):
        required_all = []
    required_all = [r for r in required_all if isinstance(r, str) and r]
    required_set = set(required_all)

    # Two independently switchable QA groups (default ON). Soft auto-fixes above
    # are never gated. See `_qa_flag` for the env var contract.
    qa_contract = _qa_flag("PPT_STEP3_QA_CONTRACT", default=True)
    qa_layout = _qa_flag("PPT_STEP3_QA_LAYOUT", default=True)

    if qa_contract:
        declared_tables = page_state.get("tables") if isinstance(page_state.get("tables"), list) else []
        if not declared_tables and re.search(r"(?is)</?(?:table|tr|td|th)\b", t):
            hard.append("undeclared_table: page_state.tables is empty; use normal text containers, not table markup")

        data_refs = re.findall(r'data-ref\s*=\s*"([^"]+)"', t)
        counts = Counter([r for r in data_refs if isinstance(r, str)])

        missing = [r for r in required_all if counts.get(r, 0) == 0]
        if missing:
            hard.append(f"missing_data_ref: {missing[:50]}{' ...' if len(missing) > 50 else ''}")

        unknown = sorted([r for r in counts.keys() if r and (r not in required_set)])
        if unknown:
            hard.append(f"unknown_data_ref: {unknown[:50]}{' ...' if len(unknown) > 50 else ''}")

        dup_candidates = [r for r, c in counts.items() if r and c > 1]
        # A data-ref may legitimately repeat ONLY when it labels one COLUMN of a
        # single <table>: every occurrence sits on a <td>/<th> cell inside the same
        # <table>. That is how we align two text nodes row-by-row (Chinese column
        # data-ref="t2", English column data-ref="t3") via native <tr> rows. Any
        # other repetition (across two tables, or on non-cell elements, or a mix) is
        # still a hard error.
        dup = sorted([r for r in dup_candidates if not _is_table_column_ref(t, r)])
        if dup:
            hard.append(f"duplicate_data_ref: {dup[:50]}{' ...' if len(dup) > 50 else ''}")

        forbidden = sorted(set(re.findall(r"\bfloating:[A-Za-z_]+\b", t)))
        forbidden += sorted(set(re.findall(r"\bp\d+_c\d+\b", t)))
        forbidden += sorted(set(re.findall(r"\bp\d+_i\d+_[A-Za-z_]+\w*\b", t)))
        forbidden = sorted(set(forbidden))
        if forbidden:
            hard.append(f"forbidden_internal_ids: {forbidden[:50]}{' ...' if len(forbidden) > 50 else ''}")

        # <style> selectors MUST be scoped under the page id (multiple pages share
        # one document) — a scoping leak collides across pages, so it is a contract
        # check rather than a layout preference.
        hard.extend(_check_style_scope(page_block=t))

    if qa_layout:
        # Layout-preference limits (CSS/colors/units are otherwise fully unlocked):
        # 1) positioning policy (absolute only as fill-parent SVG layer; no offsets);
        # 2) no external stylesheet inside the page block; 3) no overflow scroll/clip;
        # 4) content images MUST NOT be rounded.
        hard.extend(_check_positioning(page_block=t))

        if re.search(r"(?is)<link\b[^>]*rel=[\"']stylesheet[\"']", t):
            hard.append("forbidden_tag: link[rel=stylesheet] inside page block")

        if re.search(r"(?i)overflow(?:-x|-y)?\s*:\s*(auto|scroll)\b", t):
            hard.append("forbidden_css: overflow auto/scroll")

        if _has_rounded_image(page_block=t):
            hard.append("forbidden_css: border-radius on a content <img> (rounded images get locked in PowerPoint)")

    return t, hard, soft


def _generate_page_block_with_repair(
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retry_backoff: float,
    system_prompt: str,
    user_prompt: str,
    page_png_bytes: bytes | None,
    page_state: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    debug: dict[str, Any] = {
        "raw": "",
        "raw_repair_format": "",
        "raw_repair_hard": "",
        "hard_errors": [],
        "soft_warnings": [],
    }

    raw1 = _openai_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_text=user_prompt,
        image_bytes=page_png_bytes,
        max_tokens=int(max_output_tokens),
        temperature=float(temperature),
        retries=int(retries),
        retry_base_seconds=float(retry_base_seconds),
        retry_max_seconds=float(retry_max_seconds),
        retry_backoff=float(retry_backoff),
        tag="step3.render",
    )
    debug["raw"] = raw1 or ""
    page_block = _extract_page_markup(raw1)

    if not page_block:
        repair_prompt = _build_format_repair_prompt(user_prompt=user_prompt, last_raw=raw1)
        raw2 = _openai_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_text=repair_prompt,
            image_bytes=page_png_bytes,
            max_tokens=int(max_output_tokens),
            temperature=float(temperature),
            retries=int(retries),
            retry_base_seconds=float(retry_base_seconds),
            retry_max_seconds=float(retry_max_seconds),
            retry_backoff=float(retry_backoff),
            tag="step3.repair_format",
        )
        debug["raw_repair_format"] = raw2 or ""
        page_block = _extract_page_markup(raw2)

    if not page_block:
        debug["hard_errors"] = ["missing_PAGE_START_or_PAGE_END"]
        return None, debug

    normalized, hard, soft = _validate_and_normalize_page_block(page_state=page_state, page_block=page_block)
    debug["hard_errors"] = hard
    debug["soft_warnings"] = soft

    if hard:
        repair_prompt = _build_hard_repair_prompt(
            user_prompt=user_prompt, violations=hard, last_page_block=normalized
        )
        raw3 = _openai_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_text=repair_prompt,
            image_bytes=page_png_bytes,
            max_tokens=int(max_output_tokens),
            temperature=float(temperature),
            retries=int(retries),
            retry_base_seconds=float(retry_base_seconds),
            retry_max_seconds=float(retry_max_seconds),
            retry_backoff=float(retry_backoff),
            tag="step3.repair_hard",
        )
        debug["raw_repair_hard"] = raw3 or ""
        page_block2 = _extract_page_markup(raw3)
        if not page_block2:
            debug["hard_errors"] = ["missing_PAGE_START_or_PAGE_END_after_hard_repair"]
            return None, debug
        normalized2, hard2, soft2 = _validate_and_normalize_page_block(
            page_state=page_state, page_block=page_block2
        )
        debug["hard_errors"] = hard2
        debug["soft_warnings"] = soft2
        if hard2:
            return None, debug
        return normalized2, debug

    return normalized, debug


def _extract_page_markup(raw_html: str) -> str | None:
    t = _normalize_model_output(raw_html)

    starts = list(re.finditer(r"<!--\s*PAGE_START\s*-->", t, flags=re.I))
    ends = list(re.finditer(r"<!--\s*PAGE_END\s*-->", t, flags=re.I))
    if not starts or not ends:
        return None

    best_inner: str | None = None
    best_score = -10_000.0
    for sm in starts:
        for em in ends:
            if em.start() <= sm.end():
                continue
            inner = (t[sm.end() : em.start()] or "").strip()
            if not inner:
                continue

            score = 0.0
            if re.search(r"(?is)<div\b", inner):
                score += 1.0
            if re.search(r'(?is)\bclass\s*=\s*["\']page\b', inner):
                score += 5.0
            if re.search(r'(?is)\bid\s*=\s*["\']page', inner):
                score += 1.0
            score += min(4.0, float(len(inner)) / 800.0)

            if score > best_score:
                best_score = score
                best_inner = inner

    if not best_inner:
        return None
    if not re.search(r'(?is)\bclass\s*=\s*["\']page\b', best_inner):
        return None
    page_block = "<!-- PAGE_START -->\n" + best_inner + "\n<!-- PAGE_END -->\n"
    return _normalize_page_background_vars(page_block)


def _wrap_chunk_html(*, page_block: str, base_href: str, title: str) -> str:
    bh = (base_href or "").strip()
    if not bh:
        bh = "./"
    if not bh.endswith("/"):
        bh = bh + "/"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh">',
            "<head>",
            f'<base href="{bh}">',
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"<title>{title}</title>",
            '<link rel="stylesheet" href="css_library.css" />',
            "</head>",
            "<body>",
            page_block.strip(),
            "</body>",
            "</html>",
            "",
        ]
    )


def _extract_page_div_from_chunk_html(chunk_html: str) -> str | None:
    t = chunk_html or ""
    m = re.search(r"(?is)<!--\s*PAGE_START\s*-->\s*(.*?)\s*<!--\s*PAGE_END\s*-->", t)
    if not m:
        return None
    inner = (m.group(1) or "").strip()
    if not inner:
        return None
    if not re.search(r'(?is)\bclass\s*=\s*["\']page\b', inner):
        return None
    return inner


def _assemble_preview_html(*, out_bundle_dir: Path, title: str) -> Path:
    out_bundle_dir = out_bundle_dir.expanduser().resolve()
    chunks_dir = out_bundle_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.html"))
    pages: list[str] = []
    for cf in chunk_files:
        txt = cf.read_text(encoding="utf-8", errors="replace")
        page_div = _extract_page_div_from_chunk_html(txt)
        if page_div:
            pages.append(page_div)

    html_txt = "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh">',
            "<head>",
            '<base href="./">',
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"<title>{title}</title>",
            '<link rel="stylesheet" href="css_library.css" />',
            "</head>",
            "<body>",
            "\n\n".join(pages),
            "</body>",
            "</html>",
            "",
        ]
    )
    out_path = out_bundle_dir / "index.html"
    out_path.write_text(html_txt, encoding="utf-8")
    return out_path


def _rel_base_href(*, from_dir: Path, to_dir: Path) -> str:
    rel = os.path.relpath(str(to_dir), str(from_dir))
    rel = rel.replace("\\", "/")
    if rel == ".":
        return "./"
    return rel


def _downscale_jpeg(raw: bytes, *, max_edge: int = 1568, quality: int = 85) -> bytes:
    try:
        from io import BytesIO

        from PIL import Image  # type: ignore

        img = Image.open(BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_edge) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        return buf.getvalue()
    except Exception:
        return raw


def _llm_timeout_s(default: float = 300.0) -> float:
    """Per-request HTTP timeout for text-LLM calls, in seconds.

    Configurable at launch via the `PPT_LLM_TIMEOUT_S` env var so ops can retune
    it without a code change. Falls back to `default` on missing/invalid values,
    and floors at 1s so a typo can't disable the timeout entirely."""
    raw = (os.getenv("PPT_LLM_TIMEOUT_S") or "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val >= 1.0 else default


def _reasoning_effort(default: str = "minimal") -> str | None:
    """OpenAI-style `reasoning_effort` sent with step3 render calls.

    Benchmarks (see changelog 2026-07-21_00) showed `minimal` matches baseline
    quality on this task while roughly halving latency and being far more stable
    under concurrency, so step3 defaults to `minimal`. Override at launch via
    `PPT_LLM_REASONING_EFFORT`; set it to an empty string / `none` / `off` to
    omit the field entirely (fall back to the proxy's default reasoning)."""
    raw = os.getenv("PPT_LLM_REASONING_EFFORT")
    if raw is None:
        return default
    val = raw.strip()
    if not val or val.lower() in {"none", "off", "default"}:
        return None
    return val


def _qa_flag(env_name: str, *, default: bool = True) -> bool:
    """Read a boolean QA toggle from the environment.

    step3's post-generation checks split into two groups, each gated by one env
    var (default ON so behaviour is unchanged unless explicitly disabled):
    - `PPT_STEP3_QA_CONTRACT`  -> ref-mapping / structural contract checks that
      keep the HTML convertible to editable PPT (missing/unknown/duplicate
      data-ref, forbidden internal ids, <style> scoping). Turning these OFF can
      break the html->pptist bridge.
    - `PPT_STEP3_QA_LAYOUT`    -> layout-preference checks (positioning offsets,
      external stylesheet, overflow scroll, rounded <img>). Turning these OFF
      only risks a page that reflows/looks worse once converted to PPT.
    Accepts 1/true/yes/on to enable, 0/false/no/off to disable (case-insensitive).
    Soft auto-fixes (empty data-ref, background normalization) are never gated."""
    raw = os.getenv(env_name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if not val:
        return default
    if val in {"0", "false", "no", "off"}:
        return False
    if val in {"1", "true", "yes", "on"}:
        return True
    return default


def _log_llm_timing(*, tag: str, model: str, attempt: int, elapsed: float, status: Any) -> None:
    """Temporary instrumentation: print how long each text-LLM request took.

    Emitted to stderr with a stable `[llm_timing]` prefix so it can be grepped
    out of the uvicorn log. Meant to validate whether the 180s per-request
    timeout is oversized for Opus 4.8. Remove once the timeout has been retuned."""
    import sys

    print(
        f"[llm_timing] tag={tag} model={model} attempt={attempt} "
        f"elapsed={elapsed:.2f}s status={status}",
        file=sys.stderr,
        flush=True,
    )


def _openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_bytes: bytes | None,
    max_tokens: int,
    temperature: float,
    retries: int = 2,
    retry_base_seconds: float = 2.0,
    retry_max_seconds: float = 60.0,
    retry_backoff: float = 2.0,
    tag: str = "step3",
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_bytes is not None:
        jb = _downscale_jpeg(image_bytes)
        b64 = base64.b64encode(jb).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }
    effort = _reasoning_effort()
    if effort:
        body["reasoning_effort"] = effort
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.4.0",
    }
    last_err: str | None = None
    for attempt in range(max(0, int(retries)) + 1):
        t0 = time.monotonic()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=_llm_timeout_s())
            _log_llm_timing(tag=tag, model=model, attempt=attempt, elapsed=time.monotonic() - t0, status=resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                return str(data["choices"][0]["message"]["content"] or "")
            last_err = f"http_{resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as e:
            _log_llm_timing(tag=tag, model=model, attempt=attempt, elapsed=time.monotonic() - t0, status=type(e).__name__)
            last_err = f"{type(e).__name__}: {e}"
        if attempt < int(retries):
            sleep_s = min(
                float(retry_max_seconds),
                float(retry_base_seconds) * (float(retry_backoff) ** float(attempt)),
            )
            time.sleep(max(0.0, float(sleep_s)))
    raise RuntimeError(f"openai_chat_failed: {last_err}")


def _build_format_repair_prompt(*, user_prompt: str, last_raw: str) -> str:
    last_raw = (last_raw or "").strip()
    if len(last_raw) > 1500:
        last_raw = last_raw[:1500] + "\n... (truncated)\n"
    return (
        "Your previous output format is invalid (missing PAGE_START/PAGE_END, or not HTML).\n"
        "Please output again, strictly HTML only, and satisfy:\n"
        "1) First line MUST be `<!-- PAGE_START -->`\n"
        "2) Last line MUST be `<!-- PAGE_END -->`\n"
        "3) Do not output any text outside the markers. Do not output ``` fences.\n"
        "\n"
        "Here is the original input:\n\n"
        + user_prompt
        + "\n\n"
        + ("Previous invalid output (for reference only):\n" + last_raw + "\n" if last_raw else "")
    )


def _build_hard_repair_prompt(*, user_prompt: str, violations: list[str], last_page_block: str) -> str:
    vtxt = "\n".join([f"- {v}" for v in (violations or [])][:80]).strip()
    last_page_block = (last_page_block or "").strip()
    if len(last_page_block) > 1500:
        last_page_block = last_page_block[:1500] + "\n... (truncated)\n"
    return (
        "Your previous HTML violates one or more hard constraints.\n"
        "Please output again, strictly HTML only, and satisfy all constraints.\n"
        "You MUST still start with `<!-- PAGE_START -->` and end with `<!-- PAGE_END -->`.\n"
        "\n"
        "Violations detected:\n"
        + (vtxt + "\n" if vtxt else "")
        + "\n"
        "Here is the original input (do NOT rewrite any text content):\n\n"
        + user_prompt
        + "\n\n"
        + ("Previous invalid page block (for reference only):\n" + last_page_block + "\n" if last_page_block else "")
    )


def main() -> int:
    args = _parse_args()
    step2_dir = Path(args.step2_outputs_dir).expanduser().resolve()
    if not step2_dir.exists():
        raise SystemExit(f"--step2-outputs-dir not found: {step2_dir}")

    out_bundle_dir = Path(args.out_bundle_dir).expanduser().resolve()
    chunks_dir = out_bundle_dir / "chunks"

    from agent_backend.agent.tools.heavy_tool_impl._shared import api_script_dir

    default_css_dir = api_script_dir()
    css_dir = Path(args.css_dir).expanduser().resolve() if str(args.css_dir or "").strip() else default_css_dir

    prep_root = _find_prep_root(step2_dir)
    default_pages_png_dir = (prep_root / "pages_png") if prep_root is not None else (step2_dir.parent / "pages_png")
    default_images_dir = (prep_root / "ref_html" / "images") if prep_root is not None else (step2_dir.parent / "ref_html" / "images")

    pages_png_dir = (
        Path(args.pages_png_dir).expanduser().resolve()
        if str(args.pages_png_dir or "").strip()
        else default_pages_png_dir
    )
    images_dir = (
        Path(args.images_dir).expanduser().resolve()
        if str(args.images_dir or "").strip()
        else default_images_dir
    )

    beautify_refs_dir = (
        Path(args.beautify_refs_dir).expanduser().resolve()
        if str(args.beautify_refs_dir or "").strip()
        else None
    )

    css_doc = _ensure_css_library(out_bundle_dir=out_bundle_dir, css_dir=css_dir)
    img_warn = _try_symlink_images(out_bundle_dir=out_bundle_dir, images_dir=images_dir)
    if img_warn:
        (out_bundle_dir / "warnings.images.txt").write_text(img_warn + "\n", encoding="utf-8")

    selected_pages = _parse_pages_arg(str(args.pages))

    step2_paths = sorted(step2_dir.glob("*.step2_output.json"))
    if not step2_paths:
        raise SystemExit(f"No *.step2_output.json files found in: {step2_dir}")

    pages_to_run: list[tuple[int, Path, dict[str, Any]]] = []
    for p in step2_paths:
        try:
            obj = _read_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        page_num = _get_step2_page_num(obj)
        if selected_pages and (page_num not in selected_pages):
            continue
        pages_to_run.append((page_num, p, obj))

    pages_to_run = [x for x in pages_to_run if x[0] > 0]
    pages_to_run.sort(key=lambda x: (x[0], x[1].name))
    if not pages_to_run:
        raise SystemExit("No pages selected to run (check --pages and step2 outputs).")

    base_href = _rel_base_href(from_dir=chunks_dir, to_dir=out_bundle_dir)

    def _prepare_prompts(*, step2_obj: dict[str, Any]) -> tuple[int, dict[str, Any], str, str, bool, bytes | None, list[str]]:
        page_num = _get_step2_page_num(step2_obj)
        page_state = _extract_page_state(step2_obj, images_dir=images_dir)
        prep_warnings: list[str] = []

        has_layout = _has_layout_intent(step2_obj)

        # Reference image: prefer a beautify reference (when present in
        # --beautify-refs-dir), else fall back to the original page render.
        ref_bytes = _read_reference_image_bytes(
            beautify_refs_dir=beautify_refs_dir, page_num=page_num
        )
        if ref_bytes is None and pages_png_dir.exists():
            ref_bytes = _read_page_png_bytes(pages_png_dir=pages_png_dir, page_num=page_num)
        include_png = ref_bytes is not None

        system_prompt = _SYSTEM_PROMPT
        user_prompt = _build_user_prompt(
            page_state=page_state,
            css_library_doc=css_doc,
            has_layout_intent=has_layout,
        )
        return page_num, page_state, system_prompt, user_prompt, include_png, ref_bytes, prep_warnings

    if args.print_prompt:
        page_num, page_state, system_prompt, user_prompt, include_png, _png_bytes, prep_warnings = _prepare_prompts(
            step2_obj=pages_to_run[0][2]
        )
        print("===== SYSTEM PROMPT =====")
        print(system_prompt)
        print("\n===== USER PROMPT =====")
        print(user_prompt)
        if prep_warnings:
            print("\n===== PREP WARNINGS =====")
            print("\n".join(prep_warnings))
        return 0

    for page_num, step2_path, step2_obj in pages_to_run:
        page_num2, _page_state, system_prompt, user_prompt, include_png, _png_bytes, prep_warnings = _prepare_prompts(
            step2_obj=step2_obj
        )
        if page_num2 != page_num:
            page_num = page_num2
        page_dir = out_bundle_dir / f"page_{page_num:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            page_dir / "request.json",
            {
                "page_num": int(page_num),
                "step2_output_path": str(step2_path),
                "include_page_png": bool(include_png),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )
        if prep_warnings:
            _write_json(page_dir / "warnings_prep.json", {"warnings": prep_warnings})

    if args.dry_run:
        print(str(out_bundle_dir))
        return 0

    base_url, api_key = _resolve_backend(api_key_arg=str(args.api_key or ""))

    max_conc = max(1, int(args.max_concurrency))
    successes: list[int] = []
    failures: list[int] = []

    def _process_one_page(page_num: int, step2_path: Path, step2_obj: dict[str, Any]) -> bool:
        page_dir = out_bundle_dir / f"page_{page_num:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        try:
            page_num2, page_state, system_prompt, user_prompt, include_png, page_png_bytes, prep_warnings = _prepare_prompts(
                step2_obj=step2_obj
            )
            if page_num2 != page_num:
                page_num = page_num2
                page_dir = out_bundle_dir / f"page_{page_num:03d}"
                page_dir.mkdir(parents=True, exist_ok=True)

            page_block, debug = _generate_page_block_with_repair(
                base_url=base_url,
                api_key=api_key,
                model=str(args.model),
                max_output_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                retries=int(args.retries),
                retry_base_seconds=float(args.retry_base_seconds),
                retry_max_seconds=float(args.retry_max_seconds),
                retry_backoff=float(args.retry_backoff),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                page_png_bytes=page_png_bytes,
                page_state=page_state,
            )

            if not page_block:
                hard = debug.get("hard_errors") if isinstance(debug.get("hard_errors"), list) else []
                _write_text(
                    page_dir / "error.txt",
                    "Page generation failed.\n\nhard_errors:\n" + "\n".join([str(x) for x in hard]) + "\n",
                )
                if args.save_raw:
                    raw = debug.get("raw")
                    if isinstance(raw, str) and raw:
                        _write_text(page_dir / "raw.txt", raw)
                    raw2 = debug.get("raw_repair_format")
                    if isinstance(raw2, str) and raw2:
                        _write_text(page_dir / "raw_repair_format.txt", raw2)
                    raw3 = debug.get("raw_repair_hard")
                    if isinstance(raw3, str) and raw3:
                        _write_text(page_dir / "raw_repair_hard.txt", raw3)
                return False

            _write_page_outputs(
                out_bundle_dir=out_bundle_dir,
                chunks_dir=chunks_dir,
                page_num=page_num,
                page_block=page_block,
                base_href=base_href,
                title=str(args.title),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                include_page_png=include_png,
                step2_path=step2_path,
                debug=debug,
                save_raw=bool(args.save_raw),
            )
            return True
        except Exception as e:
            _write_text(page_dir / "error.txt", f"Unhandled error: {type(e).__name__}: {e}\n")
            return False

    with ThreadPoolExecutor(max_workers=max_conc) as ex:
        futures = {
            ex.submit(_process_one_page, page_num, step2_path, step2_obj): page_num
            for page_num, step2_path, step2_obj in pages_to_run
        }
        for fut in as_completed(futures):
            pno = futures[fut]
            ok = False
            try:
                ok = bool(fut.result())
            except Exception:
                ok = False
            if ok:
                successes.append(int(pno))
            else:
                failures.append(int(pno))

    _assemble_preview_html(out_bundle_dir=out_bundle_dir, title=str(args.title))
    print(str(out_bundle_dir))

    if successes:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
