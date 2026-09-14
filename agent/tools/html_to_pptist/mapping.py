"""Mapping layer: turn baked visual primitives (from measure.py) into PPTist
elements. Pure Python, no browser. Reverse of PPTist's useImport.ts.

Every primitive already carries an absolute px box in the same coordinate space
PPTist uses (page width == viewport width), so there is no coordinate math here
beyond emitting the schema. Classification was done in the measurement layer;
this layer only formats.
"""

from __future__ import annotations

import base64
import mimetypes
import random
import re
import string
from pathlib import Path

_ID_ALPHABET = string.ascii_letters + string.digits + "-_"


def _gen_id() -> str:
    return "".join(random.choices(_ID_ALPHABET, k=10))


def _rect_path(w: float, h: float, r: float = 0.0) -> tuple[str, list[float]]:
    """Rectangle (optionally rounded) as an SVG path within its own viewBox."""
    vb = [round(w, 3), round(h, 3)]
    if r <= 0:
        d = f"M 0 0 L {vb[0]} 0 L {vb[0]} {vb[1]} L 0 {vb[1]} Z"
        return d, vb
    r = min(r, w / 2, h / 2)
    x, y = vb
    d = (
        f"M {r} 0 L {x - r} 0 Q {x} 0 {x} {r} "
        f"L {x} {y - r} Q {x} {y} {x - r} {y} "
        f"L {r} {y} Q 0 {y} 0 {y - r} "
        f"L 0 {r} Q 0 0 {r} 0 Z"
    )
    return d, vb


def _ellipse_path(w: float, h: float) -> tuple[str, list[float]]:
    vb = [round(w, 3), round(h, 3)]
    rx, ry = vb[0] / 2, vb[1] / 2
    d = f"M {rx} 0 A {rx} {ry} 0 1 1 {rx} {vb[1]} A {rx} {ry} 0 1 1 {rx} 0 Z"
    return d, vb


def _polygon_path(points: str) -> tuple[str, list[float], float, float, float, float]:
    pts = [tuple(map(float, p.split(","))) for p in points.replace("\n", " ").split() if "," in p]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, miny = min(xs), min(ys)
    w = max(xs) - minx
    h = max(ys) - miny
    vb = [round(w, 3), round(h, 3)]
    d = "M " + " L ".join(f"{px - minx} {py - miny}" for px, py in pts) + " Z"
    return d, vb, minx, miny, w, h


def _shape_base(
    left,
    top,
    w,
    h,
    path,
    viewbox,
    fill,
    opacity=1.0,
    text=None,
    gradient=None,
    outline=None,
):
    el = {
        "type": "shape",
        "id": _gen_id(),
        "left": round(left, 3),
        "top": round(top, 3),
        "width": round(w, 3),
        "height": round(h, 3),
        "viewBox": [round(viewbox[0], 3), round(viewbox[1], 3)],
        "path": path,
        "fill": fill,
        "fixedRatio": False,
        "rotate": 0,
        "outline": outline or {"color": "#000000", "width": 0, "style": "solid"},
        "flipH": False,
        "flipV": False,
    }
    # PPTist uses `gradient` in preference to `fill` when present.
    if gradient and gradient.get("colors"):
        el["gradient"] = {
            "type": gradient.get("type", "linear"),
            "colors": gradient["colors"],
            "rotate": gradient.get("rotate", 0),
        }
    if text is not None:
        el["text"] = text
    if opacity is not None and opacity < 1:
        el["opacity"] = round(opacity, 3)
    return el


def _outline_from_primitive(prim: dict, sx: float = 1.0, sy: float = 1.0) -> dict:
    color = str(prim.get("strokeColor") or "")
    source_width = float(prim.get("strokeWidth") or 0)
    width = source_width * ((abs(sx) + abs(sy)) / 2)
    if not color or width <= 0:
        return {"color": "#000000", "width": 0, "style": "solid"}
    return {
        "color": color,
        "width": round(width, 3),
        "style": prim.get("strokeStyle") or "solid",
    }


_DEFAULT_TEXT_INSET = 0
_BUILTIN_FONT_VALUES = {
    "SourceHanSans",
    "SourceHanSerif",
    "WenDingPLKaiTi",
    "WenDingPLSongTi",
    "ZhuQueFangSong",
    "LXGWWenKai",
    "LXGWNeoZhiSong",
    "LXGWNeoXiHei",
    "AlibabaPuHuiTi",
    "DeYiHei",
    "MiSans",
    "SourceSerif4",
    "JetBrainsMono",
    "Literata",
    "Inter",
    "Roboto",
    "OpenSans",
    "Montserrat",
    "SourceSansPro",
    "Merriweather",
    "Lato",
}
_SYSTEM_FONT_VALUES = {
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "NSimSun",
    "KaiTi",
    "FangSong",
    "DengXian",
    "PingFang SC",
    "Hiragino Sans GB",
    "STHeiti",
    "STSong",
    "STKaiti",
    "STFangsong",
    "Aptos",
    "Aptos Display",
    "Calibri",
    "Cambria",
    "Arial",
    "Helvetica",
    "Times New Roman",
    "Georgia",
    "Verdana",
    "Tahoma",
    "Trebuchet MS",
}
_PPTIST_FONT_VALUES = _BUILTIN_FONT_VALUES | _SYSTEM_FONT_VALUES
_FONT_ALIASES = {
    "source han sans": "SourceHanSans",
    "sourcehansans": "SourceHanSans",
    "source han serif": "SourceHanSerif",
    "sourcehanserif": "SourceHanSerif",
    "source serif 4": "SourceSerif4",
    "sourceserif4": "SourceSerif4",
    "source sans pro": "SourceSansPro",
    "sourcesanspro": "SourceSansPro",
    "open sans": "OpenSans",
    "opensans": "OpenSans",
    "jetbrains mono": "JetBrainsMono",
    "jetbrainsmono": "JetBrainsMono",
    "mi sans": "MiSans",
    "misans": "MiSans",
    "微软雅黑": "Microsoft YaHei",
    "microsoft yahei": "Microsoft YaHei",
    "microsoft yahei ui": "Microsoft YaHei",
    "microsoft yahei light": "Microsoft YaHei",
    "microsoft yahei bold": "Microsoft YaHei",
    "microsoft yahei regular": "Microsoft YaHei",
    "黑体": "SimHei",
    "simhei": "SimHei",
    "宋体": "SimSun",
    "simsun": "SimSun",
    "新宋体": "NSimSun",
    "nsimsun": "NSimSun",
    "楷体": "KaiTi",
    "kaiti": "KaiTi",
    "仿宋": "FangSong",
    "fangsong": "FangSong",
    "等线": "DengXian",
    "dengxian": "DengXian",
    "苹方": "PingFang SC",
    "pingfang sc": "PingFang SC",
    "hiragino sans gb": "Hiragino Sans GB",
    "冬青黑体": "Hiragino Sans GB",
    "华文黑体": "STHeiti",
    "stheiti": "STHeiti",
    "华文宋体": "STSong",
    "stsong": "STSong",
    "华文楷体": "STKaiti",
    "stkaiti": "STKaiti",
    "华文仿宋": "STFangsong",
    "stfangsong": "STFangsong",
    "aptos": "Aptos",
    "aptos display": "Aptos Display",
    "calibri": "Calibri",
    "cambria": "Cambria",
    "arial": "Arial",
    "helvetica": "Helvetica",
    "times new roman": "Times New Roman",
    "georgia": "Georgia",
    "verdana": "Verdana",
    "tahoma": "Tahoma",
    "trebuchet ms": "Trebuchet MS",
}


def _font_name(value: object) -> str:
    text = str(value or "").strip().strip("'\"")
    if not text:
        return ""
    first = text.split(",")[0].strip().strip("'\"")
    if first in _PPTIST_FONT_VALUES:
        return first
    alias = _FONT_ALIASES.get(re.sub(r"\s+", " ", first).lower())
    if alias:
        return alias
    if re.fullmatch(r"[\w\s\-\u4e00-\u9fff.()]+", first, re.U) and len(first) <= 80:
        return first
    return ""


def _uniform_inset(value: object, default: float = _DEFAULT_TEXT_INSET) -> list[float]:
    vals: list[float] = []
    if isinstance(value, (list, tuple)) and len(value) == 4:
        for v in value:
            if isinstance(v, (int, float)):
                vals.append(max(0.0, float(v)))
    if len(vals) == 4:
        x = round(min(vals), 3)
    else:
        x = round(max(0.0, float(default)), 3)
    return [x, x, x, x]


def _shape_text(prim: dict) -> dict:
    """Build the PPTist ShapeText payload from a text-in-shape svg primitive.

    paragraphSpace is forced to 0: PPTist defaults it to 5px per paragraph, but
    the source HTML stacks its lines with no inter-paragraph margin, so keeping
    the default would inflate the text block by 5px * (#paragraphs) and push it
    past the shape box."""
    inset = _uniform_inset(prim.get("textInset"))
    text = {
        "content": prim.get("textContent", ""),
        "defaultFontName": _font_name(prim.get("textDefaultFontName")),
        "defaultColor": prim.get("textDefaultColor", "#333a4d"),
        "align": prim.get("textAlign", "middle"),
        "inset": inset,
        "lineHeight": prim.get("textLineHeight", 1.2),
        "paragraphSpace": 0,
    }
    if prim.get("textWordSpace"):
        text["wordSpace"] = round(float(prim["textWordSpace"]), 3)
    return text


def _text_element(prim: dict) -> dict:
    b = prim["box"]
    # richContent is a ready-made <p>-stack (used by grouped text-in-svgimage);
    # otherwise wrap the single run in one aligned paragraph.
    content = prim.get("richContent") or f'<p style="text-align: {prim["align"]};">{prim["content"]}</p>'
    el = {
        "type": "text",
        "id": _gen_id(),
        "left": round(b["x"], 3),
        "top": round(b["y"], 3),
        "width": round(b["w"], 3),
        "height": round(b["h"], 3),
        "rotate": 0,
        "defaultFontName": _font_name(prim.get("defaultFontName")),
        "defaultColor": prim.get("defaultColor", "#333a4d"),
        "content": content,
        "lineHeight": prim.get("lineHeight", 1.2),
        "outline": {"color": "#000000", "width": 0, "style": "solid"},
        "fill": "",
        "vertical": False,
        "inset": _uniform_inset(prim.get("inset")),
        # The baked box is authoritative: it already fits the content in the step3
        # flow layout. Without fixedHeight, PPTist's ResizeObserver re-measures the
        # text with its OWN font stack and writes the (different) height back into
        # the store, growing boxes until they overflow the page and overlap their
        # neighbours. Pinning the height (as PPTist's own PPTX importer does)
        # keeps our absolute-baked geometry intact. Vertical alignment is copied
        # from the source container when Flex/Grid expresses it explicitly.
        "fixedHeight": True,
        "vAlign": prim.get("vAlign", "top"),
    }
    if prim.get("wordSpace"):
        el["wordSpace"] = round(float(prim["wordSpace"]), 3)
    if prim.get("opacity") is not None and float(prim["opacity"]) < 1:
        el["opacity"] = round(float(prim["opacity"]), 3)
    return el


def _image_element(prim: dict, base_dir: Path) -> dict:
    b = dict(prim["box"])
    src = _resolve_src(prim["src"], base_dir)
    natural_w = float(prim.get("naturalWidth") or 0)
    natural_h = float(prim.get("naturalHeight") or 0)
    position = prim.get("objectPosition") or [0.5, 0.5]
    pos_x = min(1.0, max(0.0, float(position[0])))
    pos_y = min(1.0, max(0.0, float(position[1])))
    object_fit = str(prim.get("objectFit") or "fill").lower()
    clip = None
    if natural_w > 0 and natural_h > 0 and b["w"] > 0 and b["h"] > 0:
        if object_fit == "cover":
            scale = max(b["w"] / natural_w, b["h"] / natural_h)
            rendered_w = natural_w * scale
            rendered_h = natural_h * scale
            visible_w = min(1.0, b["w"] / rendered_w)
            visible_h = min(1.0, b["h"] / rendered_h)
            start_x = (1.0 - visible_w) * pos_x * 100
            start_y = (1.0 - visible_h) * pos_y * 100
            clip = {
                "shape": "rect",
                "range": [
                    [round(start_x, 3), round(start_y, 3)],
                    [round(start_x + visible_w * 100, 3), round(start_y + visible_h * 100, 3)],
                ],
            }
        elif object_fit == "contain":
            scale = min(b["w"] / natural_w, b["h"] / natural_h)
            rendered_w = natural_w * scale
            rendered_h = natural_h * scale
            b["x"] += (b["w"] - rendered_w) * pos_x
            b["y"] += (b["h"] - rendered_h) * pos_y
            b["w"] = rendered_w
            b["h"] = rendered_h
    el = {
        "type": "image",
        "id": _gen_id(),
        "left": round(b["x"], 3),
        "top": round(b["y"], 3),
        "width": round(b["w"], 3),
        "height": round(b["h"], 3),
        "rotate": 0,
        "fixedRatio": False,
        "src": src,
    }
    if prim.get("radius"):
        el["radius"] = round(prim["radius"], 3)
    if clip:
        el["clip"] = clip
    if prim.get("opacity") is not None and float(prim["opacity"]) < 1:
        el["opacity"] = round(float(prim["opacity"]), 3)
    return el


def _line_element(prim: dict) -> dict:
    return {
        "type": "line",
        "id": _gen_id(),
        "left": round(min(prim["x1"], prim["x2"]), 3),
        "top": round(min(prim["y1"], prim["y2"]), 3),
        "width": round(prim.get("width", 1) or 1, 2),
        "start": [round(prim["x1"] - min(prim["x1"], prim["x2"]), 3),
                  round(prim["y1"] - min(prim["y1"], prim["y2"]), 3)],
        "end": [round(prim["x2"] - min(prim["x1"], prim["x2"]), 3),
                round(prim["y2"] - min(prim["y1"], prim["y2"]), 3)],
        "style": prim.get("style", "solid"),
        "color": prim.get("color", "#000000"),
        "points": prim.get("points") or ["", ""],
    }


def _clamp_lines_to_page(elements: list[dict], page_h: float) -> None:
    """Cap every ``line`` element's vertical span to the page height, preserving
    its top and its start/end direction. A decorative divider is never meant to
    exceed the page; a runaway span (from a source-HTML layout glitch) would
    otherwise both draw off-canvas and — because it inflates the element's
    bounding box — push nothing else, but leave a stray line trailing far below.
    Horizontal lines (dy == 0) are untouched."""
    if page_h <= 0:
        return
    for el in elements:
        if el.get("type") != "line":
            continue
        start = el.get("start") or [0, 0]
        end = el.get("end") or [0, 0]
        top = float(el.get("top") or 0)
        y0, y1 = float(start[1]), float(end[1])
        span = abs(y1 - y0)
        # Max span so the line stays within the page from its current top.
        max_span = max(0.0, page_h - top)
        if span <= max_span + _LINE_CLAMP_TOLERANCE_PX or span == 0:
            continue
        scale = max_span / span if span else 1.0
        el["start"] = [round(start[0], 3), round(y0 * scale, 3)]
        el["end"] = [round(end[0], 3), round(y1 * scale, 3)]


def _svgimage_element(prim: dict) -> dict:
    """A multi-primitive / mixed svg kept verbatim as an image (data URI)."""
    b = prim["box"]
    return {
        "type": "image",
        "id": _gen_id(),
        "left": round(b["x"], 3),
        "top": round(b["y"], 3),
        "width": round(b["w"], 3),
        "height": round(b["h"], 3),
        "rotate": 0,
        "fixedRatio": False,
        "src": prim["src"],
    }


# PPTist table defaults (see useImport.ts / TableElement).
_TABLE_MIN_ROW_H = 24


def _table_cell(cell: dict, font_scale: float = 1.0) -> dict:
    style: dict = {}
    if cell.get("bold"):
        style["bold"] = True
    if cell.get("italic"):
        style["em"] = True
    if cell.get("underline"):
        style["underline"] = True
    if cell.get("strikethrough"):
        style["strikethrough"] = True
    if cell.get("color"):
        style["color"] = cell["color"]
    if cell.get("backcolor"):
        style["backcolor"] = cell["backcolor"]
    if cell.get("fontsize"):
        fs = cell["fontsize"] * font_scale
        style["fontsize"] = f"{round(fs, 2)}px"
    fontname = _font_name(cell.get("fontname"))
    if fontname:
        style["fontname"] = fontname
    if cell.get("align") and cell["align"] != "left":
        style["align"] = cell["align"]
    if cell.get("vAlign") and cell["vAlign"] != "top":
        style["vAlign"] = cell["vAlign"]
    out = {
        "id": cell.get("cellId") or _gen_id(),
        "colspan": cell.get("colspan", 1),
        "rowspan": cell.get("rowspan", 1),
        "text": cell.get("text", ""),
    }
    if style:
        out["style"] = style
    return out


def _table_element(prim: dict) -> dict:
    b = prim["box"]
    font_scale = prim.get("fontScale", 1.0) or 1.0
    data = [[_table_cell(c, font_scale) for c in row] for row in prim["rows"]]
    col_widths = prim.get("colWidths") or []
    total = sum(col_widths) or 1
    norm_widths = [round(w / total, 4) for w in col_widths] if col_widths else []
    outline = prim.get("outline", {})
    ow = outline.get("width", 0) or 0
    el = {
        "type": "table",
        "id": prim.get("tableRef") or _gen_id(),
        "left": round(b["x"], 3),
        "top": round(b["y"], 3),
        "width": round(b["w"], 3),
        "height": round(prim.get("renderHeight", b["h"]), 3),
        "colWidths": norm_widths,
        "cellMinHeight": round(prim.get("cellMinHeight") or _TABLE_MIN_ROW_H, 3),
        "rotate": 0,
        "data": data,
        "outline": {
            "width": round(ow, 2) if ow else 1,
            "style": outline.get("style", "solid"),
            "color": outline.get("color", "#eeece1"),
        },
    }
    if prim.get("theme"):
        el["theme"] = prim["theme"]
    return el


def _resolve_src(src: str, base_dir: Path) -> str:
    """Inline a local image as a data URI so the JSON is self-contained and the
    PPTist frontend can render it without file access. Passes through data:/http."""
    if src.startswith(("data:", "http://", "https://")):
        return src
    candidate = (base_dir / src).resolve()
    if not candidate.exists():
        return src
    mime = mimetypes.guess_type(str(candidate))[0] or "image/png"
    data = base64.b64encode(candidate.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _prim_path_in_viewbox(p: dict, vw: float, vh: float) -> tuple[str, list[float]]:
    """Express a single svg primitive as a path in the svg's OWN viewBox coords,
    so PPTist can scale it to the element box like the browser did (the svg uses
    preserveAspectRatio="none"). Keeps arbitrary <path> verbatim."""
    tag = p["tag"]
    vb = [round(vw, 3), round(vh, 3)]
    if tag == "path" and p.get("d"):
        return p["d"], vb
    if tag == "rect":
        w = p["width"] or vw
        h = p["height"] or vh
        d, _ = _rect_path(w, h, p.get("rx", 0))
        # rect may be offset within the viewBox; translate by prepending nothing
        # since step3 rects are full-bleed (x=y=0). Fall back to full viewBox.
        return d, [round(w, 3), round(h, 3)]
    if tag == "circle":
        d, _ = _ellipse_path(2 * p["r"], 2 * p["r"])
        return d, [round(2 * p["r"], 3), round(2 * p["r"], 3)]
    if tag == "ellipse":
        d, _ = _ellipse_path(2 * p["rxE"], 2 * p["ryE"])
        return d, [round(2 * p["rxE"], 3), round(2 * p["ryE"], 3)]
    if tag == "polygon" and p.get("points"):
        d, pvb, *_ = _polygon_path(p["points"])
        return d, pvb
    d, _ = _rect_path(vw, vh)
    return d, vb


def _svg_shape_elements(prim: dict) -> list[dict]:
    """A single-primitive <svg> becomes one editable PPTist shape.

    When the primitive carries embedded text (a "text-in-shape" card), the shape
    spans the whole card box and the serialized paragraphs go into shape.text so
    it stays one draggable unit in PPTist."""
    box = prim["box"]
    vw, vh = prim["viewBox"]

    if "textContent" in prim:
        text = _shape_text(prim)
        p0 = prim["prims"][0] if prim.get("prims") else None
        fill = p0["fill"] if p0 else "#ffffff"
        grad = p0.get("gradient") if p0 else None
        opacity = p0.get("opacity", 1.0) if p0 else 1.0
        outline = _outline_from_primitive(
            p0,
            box["w"] / vw if p0 and vw else 1.0,
            box["h"] / vh if p0 and vh else 1.0,
        ) if p0 else None
        if p0:
            d, vb = _prim_path_in_viewbox(p0, vw, vh)
        else:
            d, vb = _rect_path(box["w"], box["h"])
        return [_shape_base(
            box["x"], box["y"], box["w"], box["h"], d, vb, fill,
            opacity=opacity, text=text, gradient=grad, outline=outline,
        )]

    sx = box["w"] / vw if vw else 1
    sy = box["h"] / vh if vh else 1
    out: list[dict] = []
    for p in prim["prims"]:
        tag = p["tag"]
        grad = p.get("gradient")
        if tag == "rect":
            lw = (p["width"] or vw) * sx
            lh = (p["height"] or vh) * sy
            left = box["x"] + p["x"] * sx
            top = box["y"] + p["y"] * sy
            d, vb = _rect_path(lw, lh, p.get("rx", 0) * sx)
            out.append(_shape_base(
                left, top, lw, lh, d, vb, p["fill"],
                opacity=p.get("opacity", 1.0), gradient=grad,
                outline=_outline_from_primitive(p, sx, sy),
            ))
        elif tag == "circle":
            r = p["r"]
            lw, lh = 2 * r * sx, 2 * r * sy
            left = box["x"] + (p["cx"] - r) * sx
            top = box["y"] + (p["cy"] - r) * sy
            d, vb = _ellipse_path(lw, lh)
            out.append(_shape_base(
                left, top, lw, lh, d, vb, p["fill"],
                opacity=p.get("opacity", 1.0), gradient=grad,
                outline=_outline_from_primitive(p, sx, sy),
            ))
        elif tag == "ellipse":
            rx, ry = p["rxE"], p["ryE"]
            lw, lh = 2 * rx * sx, 2 * ry * sy
            left = box["x"] + (p["cx"] - rx) * sx
            top = box["y"] + (p["cy"] - ry) * sy
            d, vb = _ellipse_path(lw, lh)
            out.append(_shape_base(
                left, top, lw, lh, d, vb, p["fill"],
                opacity=p.get("opacity", 1.0), gradient=grad,
                outline=_outline_from_primitive(p, sx, sy),
            ))
        elif tag == "polygon" and p["points"]:
            d, vb, minx, miny, w, h = _polygon_path(p["points"])
            left = box["x"] + minx * sx
            top = box["y"] + miny * sy
            out.append(_shape_base(
                left, top, w * sx, h * sy, d, vb, p["fill"],
                opacity=p.get("opacity", 1.0), gradient=grad,
                outline=_outline_from_primitive(p, sx, sy),
            ))
        elif tag == "path" and p["d"]:
            # Single <path> in the svg's own viewBox. PPTist parses M/L/Q/A/Z into
            # editable geometry, so keep it as a normal (non-special) shape.
            out.append(_shape_base(
                box["x"], box["y"], box["w"], box["h"], p["d"], [vw, vh], p["fill"],
                opacity=p.get("opacity", 1.0), gradient=grad,
                outline=_outline_from_primitive(p, sx, sy),
            ))
    return out


def _rect_element(prim: dict) -> dict:
    b = prim["box"]
    d, vb = _rect_path(b["w"], b["h"], prim.get("radius", 0))
    return _shape_base(b["x"], b["y"], b["w"], b["h"], d, vb, prim["fill"], prim.get("opacity", 1.0))


_LINE_CLAMP_TOLERANCE_PX = 5.0
_FONT_SIZE_RE = re.compile(r"font-size:\s*([\d.]+)px")

# PPTist renders text inside `.ProseMirror-static { font-size: 16px }`. The CSS
# line box of every line carries a "strut" sized by the BLOCK font (the <p>,
# which inherits this 16px) times the element's unitless line-height ratio. When
# our emitted run font is < 16px (the norm after scaling to the 1000px canvas),
# that 16px strut dominates and every line becomes `ratio * 16` tall instead of
# `ratio * font`, inflating the whole text block (measured up to +70% on real
# pages) so it overflows and overlaps its neighbours.
#
# We can't pin the block font (ProseMirror's schema strips a font-size off <p>,
# keeping only align/indent), so we compensate through the one knob that DOES
# reach `.element-content`: the element's lineHeight ratio. Emitting
#   lineHeight = flowRatio * min(font, 16) / 16
# makes `emittedRatio * 16 == flowRatio * font` for font < 16 (and leaves
# font >= 16 untouched, since min()==16 -> flowRatio), so the rendered line box
# equals the flow's intended line height for every font size.
_PPTIST_TEXT_STRUT_PX = 16.0


def _max_font_px(content: str, default: float = _PPTIST_TEXT_STRUT_PX) -> float:
    sizes = [float(m) for m in _FONT_SIZE_RE.findall(content or "")]
    return max(sizes) if sizes else default


def _strut_compensated_ratio(flow_ratio: float, font_px: float) -> float:
    strut = _PPTIST_TEXT_STRUT_PX
    return round(flow_ratio * min(font_px, strut) / strut, 4)


def _compensate_line_heights(slides: list[dict]) -> None:
    """After all geometry scaling, rewrite every text lineHeight so PPTist's fixed
    16px ProseMirror strut reproduces the flow's real line height. Runs on the
    final emitted font sizes (which live in each run's inline `font-size:px`)."""
    for slide in slides:
        for el in slide["elements"]:
            if el["type"] == "text" and el.get("content"):
                font = _max_font_px(el["content"])
                el["lineHeight"] = _strut_compensated_ratio(el.get("lineHeight", 1.2), font)
            elif el["type"] == "shape" and el.get("text", {}).get("content"):
                font = _max_font_px(el["text"]["content"])
                el["text"]["lineHeight"] = _strut_compensated_ratio(
                    el["text"].get("lineHeight", 1.2), font
                )


def _scale_font_sizes(html: str, k: float) -> str:
    def repl(m: re.Match) -> str:
        return f"font-size: {round(float(m.group(1)) * k, 2)}px"
    return _FONT_SIZE_RE.sub(repl, html)


def _scale_element(el: dict, k: float) -> None:
    """Shrink an element's geometry (and embedded font sizes) by factor k<1."""
    for key in ("left", "top", "width", "height"):
        if key in el:
            el[key] = round(el[key] * k, 3)
    if el["type"] == "line":
        el["start"] = [round(v * k, 3) for v in el["start"]]
        el["end"] = [round(v * k, 3) for v in el["end"]]
    if el["type"] == "shape":
        # Shape paths live in their own local viewBox. Scaling both the element
        # box and every path number is unnecessary, and corrupts SVG arc flags
        # (the required 0/1 flags became values such as 0.781). Keep local
        # geometry unchanged and let PPTist scale the viewBox into the resized
        # element box.
        if el.get("text", {}).get("content"):
            el["text"]["content"] = _scale_font_sizes(el["text"]["content"], k)
            if el["text"].get("inset"):
                el["text"]["inset"] = [round(v * k, 3) for v in el["text"]["inset"]]
            if el["text"].get("wordSpace"):
                el["text"]["wordSpace"] = round(el["text"]["wordSpace"] * k, 3)
        if el.get("outline", {}).get("width"):
            el["outline"]["width"] = round(el["outline"]["width"] * k, 3)
    if el["type"] == "text" and el.get("content"):
        el["content"] = _scale_font_sizes(el["content"], k)
        if el.get("inset"):
            el["inset"] = [round(v * k, 3) for v in el["inset"]]
        if el.get("wordSpace"):
            el["wordSpace"] = round(el["wordSpace"] * k, 3)
    if el["type"] == "image" and el.get("radius"):
        el["radius"] = round(el["radius"] * k, 3)
    if el["type"] == "table":
        if el.get("cellMinHeight"):
            el["cellMinHeight"] = round(el["cellMinHeight"] * k, 3)
        for row in el.get("data", []):
            for cell in row:
                fs = cell.get("style", {}).get("fontsize")
                if fs:
                    m = re.match(r"([\d.]+)px", fs)
                    if m:
                        cell["style"]["fontsize"] = f"{round(float(m.group(1)) * k, 2)}px"


def map_page(page: dict, base_dir: Path) -> dict:
    elements: list[dict] = []
    for prim in page["elements"]:
        kind = prim["kind"]
        made: list[dict] = []
        if kind == "text":
            made.append(_text_element(prim))
        elif kind == "image":
            made.append(_image_element(prim, base_dir))
        elif kind == "svgimage":
            made.append(_svgimage_element(prim))
        elif kind == "line":
            made.append(_line_element(prim))
        elif kind == "rect":
            made.append(_rect_element(prim))
        elif kind == "shape":
            made.extend(_svg_shape_elements(prim))
        elif kind == "table":
            made.append(_table_element(prim))
        # Carry a shared groupId (svg-image + its text box) so PPTist selects and
        # drags them together while the text stays editable.
        gid = prim.get("groupId")
        if gid:
            for el in made:
                el["groupId"] = gid
        elements.extend(made)
    return {
        "id": _gen_id(),
        "elements": elements,
        "background": {"type": "solid", "color": page.get("background", "#ffffff")},
    }


# PPTist's default canvas width (store.viewportSize). We normalise every output
# to this width so the JSON drops into any presentation regardless of the import
# path: importJSON only updates viewportSize on cover / empty-deck imports, and
# addSlidesFromData (appending to an existing deck) never touches it, so a
# 1280-wide doc would overflow the 1000-wide canvas by ~28%.
_TARGET_WIDTH = 1000.0


def build_pptist(pages: list[dict], base_dir: Path, title: str = "Converted") -> dict:
    slides = [map_page(p, base_dir) for p in pages]

    src_w = pages[0]["width"] if pages else 1280
    src_h = pages[0]["height"] if pages else 720
    k = _TARGET_WIDTH / src_w if src_w else 1.0
    if abs(k - 1.0) > 1e-6:
        for slide in slides:
            for el in slide["elements"]:
                _scale_element(el, k)
    width = _TARGET_WIDTH
    height = src_h * k

    # Defensive: a decorative line/divider is never intended to be taller than
    # the page. The measure-time `svg { max-height }` cap already prevents the
    # ~20000px runaway from an absolute-positioning-that-didn't-apply divider,
    # but clamp here too so a `line` element's vertical span can't exceed the
    # page height regardless of how the source HTML laid it out.
    for slide in slides:
        _clamp_lines_to_page(slide["elements"], height)

    # Now that font sizes are final (post-scale), rewrite text lineHeights to
    # cancel PPTist's fixed 16px ProseMirror strut so the editor reproduces the
    # flow's real line height instead of inflating short-font text.
    _compensate_line_heights(slides)

    # No `theme` key on purpose: every element carries its own baked-in styles,
    # and PPTist supplies sane theme defaults. Emitting a theme here would only
    # affect newly-created elements and risk fighting the host deck's theme.
    return {
        "title": title,
        "width": round(width, 3),
        "height": round(height, 3),
        "slides": slides,
    }


def finalize_text_heights(doc: dict, rendered: dict[str, dict]) -> None:
    """Overwrite each text element's fixedHeight box with the height PPTist
    actually renders (measured in the editor's own DOM model, keyed by element
    id) so the box matches the real render and never clips or gets re-measured.
    Shape boxes remain authoritative because their geometry belongs to the SVG;
    shape text is therefore not resized in this pass."""
    for slide in doc["slides"]:
        for el in slide["elements"]:
            info = rendered.get(el["id"])
            if not info:
                continue
            if el["type"] == "text":
                el["height"] = round(info["height"], 3)
