"""Measurement layer: render step3 flow HTML in a headless browser and bake every
visual element down to an absolute-positioned primitive (px, page-relative).

This layer is intentionally dumb about PPTist. It only knows about the browser:
it traverses the DOM of each top-level page and classifies each node into a
generic visual primitive (text / image / rect-shape / svg-shape / line) using
ONLY tag name, computed style and DOM structure. No per-page / per-example rules.

Output: a list of "page" dicts, each { width, height, background, elements: [...] }
where every element carries a page-relative px box plus the raw data the mapping
layer needs to emit a PPTist element.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

# The JS runs inside the rendered page. It returns one record per visual
# primitive. All geometry is page-relative CSS px. Kept free of any knowledge
# about specific class names so it generalises to every step3 page.
_EXTRACT_JS = r"""
() => {
  const INLINE_TAGS = new Set(['span','b','strong','i','em','u','a','sub','sup','small','mark','font','label','abbr','s','strike','br','code']);

  // PPTist's $textElementFont (src/assets/styles/variable.scss). Table cells fall
  // back to this stack, so offscreen table measurement must use it verbatim.
  const PPTIST_TABLE_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"';

  const pxNum = (v) => parseFloat(v) || 0;
  const finiteNum = (v, fallback = 0) => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : fallback;
  };

  const isTransparent = (c) => {
    if (!c) return true;
    if (c === 'transparent') return true;
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return false;
    const parts = m[1].split(',').map(s => s.trim());
    if (parts.length === 4 && parseFloat(parts[3]) === 0) return true;
    return false;
  };

  const rgbToHex = (c) => {
    if (!c) return '#000000';
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return c.startsWith('#') ? c : '#000000';
    const p = m[1].split(',').map(s => parseFloat(s.trim()));
    const h = (n) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
    return '#' + h(p[0]) + h(p[1]) + h(p[2]);
  };

  const colorAlpha = (c) => {
    const m = c && c.match(/rgba?\(([^)]+)\)/);
    if (!m) return 1;
    const parts = m[1].split(',').map(s => s.trim());
    return parts.length === 4 ? parseFloat(parts[3]) : 1;
  };

  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const isBold = (fw) => fw === 'bold' || parseInt(fw, 10) >= 600;

  // Serialise the inline runs of a text-leaf into PPTist paragraph HTML,
  // carrying each run's own computed color / size / weight / style.
  const serializeRuns = (el) => {
    let html = '';
    const emit = (text, cs) => {
      if (!text) return;
      const color = rgbToHex(cs.color);
      const fs = Math.round(pxNum(cs.fontSize) * 100) / 100;
      const fontFamily = (cs.fontFamily || '').split(',')[0].trim().replace(/^["']|["']$/g, '');
      let spanStyle = `color: ${color};font-size: ${fs}px;`;
      if (fontFamily) spanStyle += `font-family: ${fontFamily};`;
      // Color / size / family go on the span; ProseMirror keeps these via its
      // forecolor / fontsize / fontname marks. Bold / italic / underline MUST be emitted as
      // <strong>/<em>/<u> tags, because the editor's schema parses those marks
      // from tags, not from a font-weight/text-decoration style on a span.
      let inner = escapeHtml(text);
      if (cs.fontStyle === 'italic') inner = `<em>${inner}</em>`;
      if (cs.textDecorationLine && cs.textDecorationLine.includes('underline')) inner = `<u>${inner}</u>`;
      if (isBold(cs.fontWeight)) inner = `<strong>${inner}</strong>`;
      html += `<span style="${spanStyle}">${inner}</span>`;
    };
    const rec = (node, cs) => {
      for (const n of node.childNodes) {
        if (n.nodeType === 3) {
          emit(n.textContent, cs);
        } else if (n.nodeType === 1) {
          if (n.tagName.toLowerCase() === 'br') { html += '<br>'; continue; }
          rec(n, getComputedStyle(n));
        }
      }
    };
    rec(el, getComputedStyle(el));
    return html;
  };

  const mapAlign = (a) => {
    if (a === 'start') return 'left';
    if (a === 'end') return 'right';
    if (['left', 'right', 'center', 'justify'].includes(a)) return a;
    return 'left';
  };

  const horizontalAlignFromStyle = (cs) => {
    const explicit = mapAlign(String(cs.textAlign || '').toLowerCase());
    if (explicit !== 'left' || String(cs.textAlign || '').toLowerCase() === 'left') return explicit;
    const display = String(cs.display || '').toLowerCase();
    if (!display.includes('flex') && !display.includes('grid')) return explicit;
    const direction = String(cs.flexDirection || 'row').toLowerCase();
    const axisValue = direction.startsWith('column') ? cs.alignItems : cs.justifyContent;
    const value = String(axisValue || '').toLowerCase();
    if (value.includes('center')) return 'center';
    if (value.includes('end')) return 'right';
    return 'left';
  };

  const verticalAlignFromStyle = (cs) => {
    const display = String(cs.display || '').toLowerCase();
    if (!display.includes('flex') && !display.includes('grid')) return 'top';

    // Grid containers commonly center their rows with align-content. This is
    // distinct from align-items, which aligns content inside each grid area.
    // Reading only align-items loses the vertical position of a centered list.
    if (display.includes('grid')) {
      const contentValue = String(cs.alignContent || '').toLowerCase();
      if (contentValue.includes('center')) return 'middle';
      if (contentValue.includes('end')) return 'bottom';
    }

    const direction = String(cs.flexDirection || 'row').toLowerCase();
    const axisValue = direction.startsWith('column') ? cs.justifyContent : cs.alignItems;
    const value = String(axisValue || '').toLowerCase();
    if (value.includes('center')) return 'middle';
    if (value.includes('end')) return 'bottom';
    return 'top';
  };

  const lineStyleFromDash = (dashValue) => {
    const dash = String(dashValue || '').replace(/none/g, '').trim();
    if (!dash) return 'solid';
    const nums = (dash.match(/[\d.]+/g) || []).map(Number);
    if (nums.length >= 2 && nums[0] <= nums[1] * 0.75) return 'dotted';
    return 'dashed';
  };

  // A text-leaf: has visible text, contains no img/svg/table, and has no
  // block-level descendants (so it is a single visual text block).
  const isTextLeaf = (el) => {
    if (!el.textContent || !el.textContent.trim()) return false;
    if (el.querySelector('img, svg, table')) return false;
    for (const d of el.querySelectorAll('*')) {
      const tag = d.tagName.toLowerCase();
      if (!INLINE_TAGS.has(tag)) {
        const disp = getComputedStyle(d).display;
        if (disp !== 'inline' && disp !== 'inline-block') return false;
      }
    }
    return true;
  };

  // Some layouts put a decorative SVG next to a text node in the same
  // container (for example a timeline dot followed by an event description).
  // The container is not a text leaf because it contains an SVG, and walking
  // element children alone would silently drop that direct text node. Measure
  // each non-whitespace direct text node as its own editable text primitive.
  const emitDirectTextNodes = (el, origin, out) => {
    for (const node of el.childNodes) {
      if (node.nodeType !== Node.TEXT_NODE) continue;
      const raw = node.textContent || '';
      const start = raw.search(/\S/);
      if (start < 0) continue;
      const endMatch = raw.match(/\S\s*$/);
      const end = endMatch ? endMatch.index + endMatch[0].replace(/\s+$/, '').length : raw.length;
      const text = raw.slice(start, end);
      if (!text.trim()) continue;

      const range = document.createRange();
      range.setStart(node, start);
      range.setEnd(node, end);
      const r = range.getBoundingClientRect();
      if (!(r.width > 0 && r.height > 0)) continue;

      const cs = getComputedStyle(el);
      const color = rgbToHex(cs.color);
      const fs = Math.round(pxNum(cs.fontSize) * 100) / 100;
      const fontFamily = (cs.fontFamily || '').split(',')[0].trim().replace(/^['"]|['"]$/g, '');
      let style = `color: ${color};font-size: ${fs}px;`;
      if (fontFamily) style += `font-family: ${fontFamily};`;
      let inner = escapeHtml(text);
      if (cs.fontStyle === 'italic') inner = `<em>${inner}</em>`;
      if (cs.textDecorationLine && cs.textDecorationLine.includes('underline')) inner = `<u>${inner}</u>`;
      if (isBold(cs.fontWeight)) inner = `<strong>${inner}</strong>`;

      const lh = pxNum(cs.lineHeight);
      out.push({
        kind: 'text',
        box: { x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height },
        content: `<span style="${style}">${inner}</span>`,
        align: horizontalAlignFromStyle(cs),
        vAlign: verticalAlignFromStyle(cs),
        lineHeight: (lh && fs) ? +(lh / fs).toFixed(3) : 1.2,
        defaultColor: color,
        defaultFontName: fontFamily,
        inset: [0, 0, 0, 0],
        wordSpace: pxNum(cs.letterSpacing),
        opacity: finiteNum(cs.opacity, 1),
      });
    }
  };

  const relBox = (el, origin) => {
    const r = el.getBoundingClientRect();
    return { x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height };
  };

  // Serialise a whole <svg> into a self-contained data URI so it can be kept
  // verbatim as a PPTist image element (pixel-perfect, not editable). Used for
  // any svg we don't reduce to a single native line/shape.
  const svgToDataUri = (svg) => {
    let markup = svg.outerHTML;
    if (!/xmlns=/.test(markup)) {
      markup = markup.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    return 'data:image/svg+xml;utf8,' + encodeURIComponent(markup);
  };

  // Resolve a fill that references an SVG gradient (fill="url(#id)") into a
  // PPTist gradient {type, colors:[{pos,color}], rotate}. PPTist's linear
  // gradient runs left->right and is then rotated `rotate` degrees about the
  // centre, so we derive the angle from the gradient's (x1,y1)->(x2,y2) vector.
  const parseGradientFill = (fillAttr, svgRoot) => {
    if (!fillAttr) return null;
    const m = fillAttr.match(/url\(["']?#([^"')]+)["']?\)/);
    if (!m || !svgRoot) return null;
    let def = svgRoot.querySelector('#' + CSS.escape(m[1]));
    if (!def) return null;
    // A gradient may inherit stops via xlink:href from another gradient.
    let stopsHost = def;
    if (!def.querySelector('stop')) {
      const href = def.getAttribute('href') || def.getAttribute('xlink:href');
      if (href && href.startsWith('#')) {
        const ref = svgRoot.querySelector(CSS.escape(href.slice(1)) ? '#' + CSS.escape(href.slice(1)) : href);
        if (ref) stopsHost = ref;
      }
    }
    const stops = Array.from(stopsHost.querySelectorAll('stop'));
    if (!stops.length) return null;
    const colors = stops.map((s, i) => {
      const scs = getComputedStyle(s);
      const off = s.getAttribute('offset');
      let pos;
      if (off == null) pos = (i / Math.max(1, stops.length - 1)) * 100;
      else pos = off.includes('%') ? parseFloat(off) : parseFloat(off) * 100;
      const col = s.getAttribute('stop-color') || scs.stopColor || '#000000';
      return { pos: Math.round(pos), color: rgbToHex(col) };
    });
    const type = def.tagName.toLowerCase() === 'radialgradient' ? 'radial' : 'linear';
    let rotate = 0;
    if (type === 'linear') {
      const x1 = parseFloat(def.getAttribute('x1') ?? '0');
      const y1 = parseFloat(def.getAttribute('y1') ?? '0');
      const x2 = parseFloat(def.getAttribute('x2') ?? '1');
      const y2 = parseFloat(def.getAttribute('y2') ?? '0');
      rotate = Math.round(Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI);
      if (rotate < 0) rotate += 360;
    }
    return { type, colors, rotate };
  };

  const readPrimGeometry = (p, svgRoot) => {
    const cs = getComputedStyle(p);
    const fillAttr = p.getAttribute('fill');
    const gradient = parseGradientFill(fillAttr, svgRoot);
    // Use computed colors for named values such as `white`; keep url(#...) so
    // gradients can still be resolved by parseGradientFill above.
    const computedFill = fillAttr && /^url\(/i.test(fillAttr) ? fillAttr : (cs.fill || fillAttr);
    const fill = gradient
      ? '#ffffff'
      : (!computedFill || computedFill === 'none' || isTransparent(computedFill) ? '' : rgbToHex(computedFill));
    const strokeAttr = p.getAttribute('stroke');
    const stroke = strokeAttr && /^url\(/i.test(strokeAttr) ? strokeAttr : (cs.stroke || strokeAttr);
    const strokeWidth = pxNum(p.getAttribute('stroke-width') || cs.strokeWidth);
    const strokeDash = p.getAttribute('stroke-dasharray') || cs.strokeDasharray;
    return {
      tag: p.tagName.toLowerCase(),
      fill,
      gradient: gradient,
      strokeColor: (!stroke || stroke === 'none' || isTransparent(stroke)) ? '' : rgbToHex(stroke),
      strokeWidth,
      strokeStyle: lineStyleFromDash(strokeDash),
      opacity: finiteNum(cs.opacity, 1) * finiteNum(cs.fillOpacity, 1),
      x: pxNum(p.getAttribute('x')), y: pxNum(p.getAttribute('y')),
      width: pxNum(p.getAttribute('width')), height: pxNum(p.getAttribute('height')),
      rx: pxNum(p.getAttribute('rx') || p.getAttribute('ry')),
      cx: pxNum(p.getAttribute('cx')), cy: pxNum(p.getAttribute('cy')),
      r: pxNum(p.getAttribute('r')),
      rxE: pxNum(p.getAttribute('rx')), ryE: pxNum(p.getAttribute('ry')),
      points: p.getAttribute('points') || '',
      d: p.getAttribute('d') || '',
    };
  };

  const DRAWING_TAGS = new Set(['line', 'polyline', 'rect', 'circle', 'ellipse', 'polygon', 'path']);
  const DEFINITION_TAGS = new Set(['defs', 'clippath', 'mask', 'marker', 'pattern', 'symbol', 'filter']);

  // Only count visible drawing nodes. querySelectorAll() on the whole SVG also
  // sees gradient definitions, clip paths and masks, which are not painted
  // objects and can incorrectly force a simple shape down the SVG-image path.
  const visibleSvgPrimitives = (svg) => Array.from(svg.querySelectorAll('*')).filter((node) => {
    const tag = node.tagName.toLowerCase();
    if (!DRAWING_TAGS.has(tag)) return false;
    let parent = node.parentElement;
    while (parent && parent !== svg) {
      if (DEFINITION_TAGS.has(parent.tagName.toLowerCase())) return false;
      parent = parent.parentElement;
    }
    const cs = getComputedStyle(node);
    return cs.display !== 'none' && cs.visibility !== 'hidden'
      && finiteNum(cs.opacity, 1) > 0;
  });

  const svgHasUnsupportedEffects = (svg, primitives) => {
    if (svg.querySelector('filter, mask, clipPath, marker, pattern, symbol, foreignObject, use')) return true;
    return primitives.some((node) => {
      const cs = getComputedStyle(node);
      return cs.filter && cs.filter !== 'none';
    });
  };

  const supportedNativePath = (d) =>
    /^[\s,0-9+\-.eEMmLlQqAaZz]+$/.test(String(d || ''));

  // Classify an <svg> by the number/kind of drawing primitives it holds:
  //  - exactly one <line>/<polyline>, nothing else  -> native PPTist 'line'
  //  - exactly one filled primitive (rect/circle/ellipse/polygon/path)
  //                                                  -> native editable 'shape'
  //  - anything else (2+ primitives, mixed)          -> keep whole svg as image
  const readSvg = (svg, origin) => {
    const box = relBox(svg, origin);
    const vb = (svg.getAttribute('viewBox') || '').trim().split(/[\s,]+/).map(parseFloat);
    const viewBox = (vb.length === 4 && vb[2] > 0 && vb[3] > 0) ? [vb[2], vb[3]] : [box.w, box.h];

    const primitives = visibleSvgPrimitives(svg);
    const lines = primitives.filter((node) => ['line', 'polyline'].includes(node.tagName.toLowerCase()));
    const shapes = primitives.filter((node) => !['line', 'polyline'].includes(node.tagName.toLowerCase()));
    const total = lines.length + shapes.length;

    // Native PPTist geometry is used only where this converter can preserve
    // the rendered meaning. Complex effects remain a self-contained SVG image.
    if (svgHasUnsupportedEffects(svg, primitives)) {
      return { kind: 'svgimage', box, viewBox, src: svgToDataUri(svg) };
    }

    // A stroke-only M/L path is a connector, not a filled shape. Step3 may
    // express a plain arrow as one path containing the shaft plus a tiny
    // arrowhead subpath. Converting that path as a PPTist shape discards its
    // white stroke and closes/fills it as a black triangle. Recognise the
    // simple M/L form and emit an editable native line instead.
    if (total === 1 && shapes.length === 1 && shapes[0].tagName.toLowerCase() === 'path') {
      const path = shapes[0];
      const fillAttr = (path.getAttribute('fill') || '').trim().toLowerCase();
      const cs = getComputedStyle(path);
      const stroke = path.getAttribute('stroke') || cs.stroke;
      const d = path.getAttribute('d') || '';
      const onlyMoveLine = /^[\s,0-9+\-.eEMLl]+$/.test(d);
      const nums = (d.match(/[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g) || []).map(Number);
      if (fillAttr === 'none' && stroke && stroke !== 'none' && onlyMoveLine && nums.length === 4) {
        const sx = box.w / viewBox[0];
        const sy = box.h / viewBox[1];
        const dash = (path.getAttribute('stroke-dasharray') || cs.strokeDasharray || '').replace(/none/, '').trim();
        return {
          kind: 'line', box,
          x1: box.x + nums[0] * sx, y1: box.y + nums[1] * sy,
          x2: box.x + nums[2] * sx, y2: box.y + nums[3] * sy,
          color: rgbToHex(stroke),
          width: pxNum(path.getAttribute('stroke-width') || cs.strokeWidth) * ((sx + sy) / 2) || 1,
          style: lineStyleFromDash(dash),
          points: ['', ''],
        };
      }
    }

    if (total === 1 && lines.length === 1) {
      const ln = lines[0];
      const sx = box.w / viewBox[0];
      const sy = box.h / viewBox[1];
      let x1, y1, x2, y2;
      if (ln.tagName.toLowerCase() === 'line') {
        x1 = pxNum(ln.getAttribute('x1')); y1 = pxNum(ln.getAttribute('y1'));
        x2 = pxNum(ln.getAttribute('x2')); y2 = pxNum(ln.getAttribute('y2'));
      } else {
        const pts = (ln.getAttribute('points') || '').trim().split(/\s+/).map(p => p.split(',').map(parseFloat));
        if (pts.length !== 2 || pts.some(p => p.length < 2 || !p.every(Number.isFinite))) {
          return { kind: 'svgimage', box, viewBox, src: svgToDataUri(svg) };
        }
        x1 = pts[0][0]; y1 = pts[0][1];
        x2 = pts[pts.length - 1][0]; y2 = pts[pts.length - 1][1];
      }
      const cs = getComputedStyle(ln);
      const stroke = ln.getAttribute('stroke') || cs.stroke;
      const dash = (ln.getAttribute('stroke-dasharray') || cs.strokeDasharray || '').replace(/none/, '').trim();
      return {
        kind: 'line',
        box,
        x1: box.x + x1 * sx, y1: box.y + y1 * sy,
        x2: box.x + x2 * sx, y2: box.y + y2 * sy,
        color: rgbToHex(stroke),
        width: pxNum(ln.getAttribute('stroke-width') || cs.strokeWidth) * ((sx + sy) / 2) || 1,
        style: lineStyleFromDash(dash),
      };
    }

    if (total === 1 && shapes.length === 1) {
      if (shapes[0].tagName.toLowerCase() === 'path'
          && !supportedNativePath(shapes[0].getAttribute('d') || '')) {
        return { kind: 'svgimage', box, viewBox, src: svgToDataUri(svg) };
      }
      // Single filled primitive -> one editable PPTist shape.
      return { kind: 'shape', box, viewBox, prims: [readPrimGeometry(shapes[0], svg)] };
    }

    // Multiple / mixed primitives: keep the svg verbatim as an image.
    return { kind: 'svgimage', box, viewBox, src: svgToDataUri(svg) };
  };

  // Serialise every text-leaf under `root` (skipping `skipEl`) into a stack of
  // <p> paragraphs, one per visual line-box, preserving reading order and each
  // paragraph's own horizontal text-align. This is what a PPTist shape's
  // text.content expects (a ProseMirror paragraph flow).
  const serializeParagraphs = (root, skipEl, fallbackAlign = null) => {
    const rootTextAlign = String(getComputedStyle(root).textAlign || '').toLowerCase();
    const paragraphAlign = (cs) => {
      const raw = String(cs.textAlign || '').toLowerCase();
      if (fallbackAlign && (raw === rootTextAlign || raw === 'start' || raw === 'auto')) return fallbackAlign;
      return mapAlign(raw);
    };
    if (isTextLeaf(root)) {
      const cs = getComputedStyle(root);
      const align = fallbackAlign || paragraphAlign(cs);
      return `<p style="text-align: ${align};">${serializeRuns(root)}</p>`;
    }
    let html = '';
    const visit = (el) => {
      for (const child of el.children) {
        if (child === skipEl) continue;
        const t = child.tagName.toLowerCase();
        if (t === 'svg' || t === 'img') continue;
        const ccs = getComputedStyle(child);
        if (ccs.display === 'none' || ccs.visibility === 'hidden') continue;
        if (isTextLeaf(child)) {
          const align = paragraphAlign(ccs);
          html += `<p style="text-align: ${align};">${serializeRuns(child)}</p>`;
        } else {
          visit(child);
        }
      }
    };
    visit(root);
    return html;
  };

  const paddingScalar = (cs) => {
    const vals = [
      pxNum(cs.paddingTop),
      pxNum(cs.paddingRight),
      pxNum(cs.paddingBottom),
      pxNum(cs.paddingLeft),
    ].filter(v => Number.isFinite(v));
    if (vals.length !== 4) return 0;
    // PPTist's inset is a uniform inner padding value.  Do not discard zero
    // sides: `padding: 0 46px 0 0` is a right-aligned layout instruction, not
    // a 46px inset.  Taking the minimum preserves the usable text width and
    // leaves alignment to the paragraph's text-align value.
    return Math.min.apply(null, vals);
  };

  const textPaddingScalar = (root, skipEl) => {
    const own = paddingScalar(getComputedStyle(root));
    if (own > 0) return own;
    const stack = Array.from(root.children || []);
    while (stack.length) {
      const child = stack.shift();
      if (child === skipEl) continue;
      const tag = child.tagName.toLowerCase();
      if (tag === 'svg' || tag === 'img') continue;
      const cs = getComputedStyle(child);
      if (cs.display === 'none' || cs.visibility === 'hidden') continue;
      const p = paddingScalar(cs);
      if (p > 0 && (child.textContent || '').trim()) return p;
      stack.push(...Array.from(child.children || []));
    }
    return 0;
  };

  // The tight glyph bounding box (page-relative) of all text under `root`.
  // Kept for positioning standalone text over complex grouped SVG images. It is
  // never used to infer PPTist text insets.
  const textUnionBox = (root, origin, skipEl) => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const add = (r) => {
      if (!r || r.width <= 0 || r.height <= 0) return;
      const x = r.left - origin.left, y = r.top - origin.top;
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + r.width); maxY = Math.max(maxY, y + r.height);
    };
    const collect = (node) => {
      for (const child of node.childNodes) {
        if (child === skipEl) continue;
        if (child.nodeType === 3) {                 // text node
          if (!child.textContent.trim()) continue;
          const range = document.createRange();
          range.selectNodeContents(child);
          for (const r of range.getClientRects()) add(r);
        } else if (child.nodeType === 1) {          // element
          const t = child.tagName.toLowerCase();
          if (t === 'svg' || t === 'img') continue;
          const ccs = getComputedStyle(child);
          if (ccs.display === 'none' || ccs.visibility === 'hidden') continue;
          collect(child);
        }
      }
    };
    collect(root);
    if (minX === Infinity) return null;
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
  };

  // A direct-child <svg> that fills its parent's box and carries at least one
  // filled primitive is a background layer; its parent is a "text-in-shape"
  // card. (A line-only svg is not a background.)
  const findCoverBgSvg = (el, elBox) => {
    for (const child of el.children) {
      if (child.tagName.toLowerCase() !== 'svg') continue;
      const b = child.getBoundingClientRect();
      const covers = b.width >= elBox.w * 0.9 && b.height >= elBox.h * 0.9;
      if (!covers) continue;
      const shapes = visibleSvgPrimitives(child).filter((node) =>
        !['line', 'polyline'].includes(node.tagName.toLowerCase()));
      if (shapes.length > 0) return child;
    }
    return null;
  };

  // Return the sole content reference owned by this visual component. The
  // preferred contract puts data-ref on the geometry-bearing container, but
  // older/model-produced HTML may put it on one nested text element instead.
  const soleContentRef = (el) => {
    const refs = [];
    if (el.hasAttribute('data-ref')) refs.push(el);
    refs.push(...Array.from(el.querySelectorAll('[data-ref]')));
    return refs.length === 1 ? refs[0] : null;
  };

  const objectPositionRatios = (value) => {
    const keywords = { left: 0, top: 0, center: 0.5, right: 1, bottom: 1 };
    const parts = String(value || '50% 50%').trim().split(/\s+/);
    const parse = (part, fallback) => {
      const lower = String(part || '').toLowerCase();
      if (lower in keywords) return keywords[lower];
      if (lower.endsWith('%')) return Math.max(0, Math.min(1, finiteNum(lower, fallback * 100) / 100));
      return fallback;
    };
    return [parse(parts[0], 0.5), parse(parts[1] || parts[0], 0.5)];
  };

  let _groupSeq = 0;
  const nextGroupId = () => 'grp' + (++_groupSeq);

  // Read an HTML <table> into a PPTist-friendly grid. PPTist table cells hold
  // plain text plus one cell-level style, so we take innerText + the cell's
  // dominant computed style. Column widths come from the first row's measured
  // cell widths; row height from the first row.
  // Build a detached copy of the table that mirrors PPTist's StaticTable render
  // (table-layout:fixed, colgroup px widths, per-cell .cell-text padding:5px /
  // line-height:1.5 / minHeight, per-cell fontSize), measure its real pixel
  // height offscreen. fontScale multiplies every cell font size uniformly so we
  // can probe "how tall would PPTist draw this if we shrank all fonts by k".
  const measurePptistTable = (rows, normWidths, elementWidth, cellMinHeight, outline, fontScale) => {
    const host = document.createElement('div');
    host.style.cssText = 'position:absolute;left:-99999px;top:0;visibility:hidden;';
    host.style.width = elementWidth + 'px';
    // Match PPTist's $textElementFont EXACTLY: table cells inherit this stack,
    // and line wrapping (hence row height) is font-metric dependent. Measuring
    // in the source HTML's font would mis-count wrapped lines and under-estimate
    // the real rendered height, making autoshrink stop too early.
    host.style.fontFamily = PPTIST_TABLE_FONT;
    const tbl = document.createElement('table');
    tbl.style.cssText = 'width:100%;table-layout:fixed;border-collapse:collapse;border-spacing:0;border:0;word-wrap:break-word;';
    tbl.style.fontFamily = PPTIST_TABLE_FONT;
    const colgroup = document.createElement('colgroup');
    normWidths.forEach(w => {
      const col = document.createElement('col');
      col.setAttribute('span', '1');
      col.width = String(Math.round(w * elementWidth));
      colgroup.appendChild(col);
    });
    tbl.appendChild(colgroup);
    const tbody = document.createElement('tbody');
    const trEls = [];
    rows.forEach(rowCells => {
      const tr = document.createElement('tr');
      tr.style.height = cellMinHeight + 'px';
      rowCells.forEach(cell => {
        if (cell.covered) return;
        const td = document.createElement('td');
        td.rowSpan = cell.rowspan || 1;
        td.colSpan = cell.colspan || 1;
        td.style.cssText = 'position:relative;white-space:normal;word-wrap:break-word;vertical-align:middle;background-clip:padding-box;';
        td.style.borderStyle = 'solid';
        td.style.borderColor = outline.color || '#eeece1';
        td.style.borderWidth = (outline.width || 0) + 'px';
        const div = document.createElement('div');
        div.style.cssText = 'padding:5px;line-height:1.5;display:flex;flex-direction:column;align-items:stretch;';
        div.style.minHeight = (cellMinHeight - 4) + 'px';
        div.style.fontWeight = cell.bold ? 'bold' : 'normal';
        div.style.fontFamily = cell.fontname || PPTIST_TABLE_FONT;
        const baseFs = cell.fontsize || 14;
        div.style.fontSize = (baseFs * fontScale) + 'px';
        // PPTist formatText: \n -> <br>, spaces -> &nbsp;
        div.innerHTML = (cell.text || '')
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/\n/g, '<br>').replace(/ /g, '&nbsp;');
        td.appendChild(div);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
      trEls.push(tr);
    });
    tbl.appendChild(tbody);
    host.appendChild(tbl);
    document.body.appendChild(host);
    const total = tbl.getBoundingClientRect().height;
    const rowHeights = trEls.map(tr => tr.getBoundingClientRect().height);
    document.body.removeChild(host);
    return { total, rowHeights };
  };

  // Content-driven total height: measure with a near-zero minimum so each row is
  // as tall as its own text needs and nothing is forced taller. This is what
  // PPTist will actually draw once we emit a cellMinHeight <= the shortest row.
  const measurePptistTableHeight = (rows, normWidths, elementWidth, outline, fontScale) =>
    measurePptistTable(rows, normWidths, elementWidth, 1, outline, fontScale).total;

  const readTable = (table, origin) => {
    const box = relBox(table, origin);
    const rowsEl = Array.from(table.querySelectorAll('tr'));
    const physicalRows = [];
    const colWidths = [];
    rowsEl.forEach((tr, ri) => {
      const cells = [];
      const tds = Array.from(tr.children).filter(c => /^(td|th)$/i.test(c.tagName));
      tds.forEach((td, ci) => {
        const ccs = getComputedStyle(td);
        const cbox = td.getBoundingClientRect();
        if (ri === 0) colWidths.push(cbox.width || 1);
        cells.push({
          cellId: td.getAttribute('data-cell-id') || '',
          text: td.innerText || td.textContent || '',
          colspan: parseInt(td.getAttribute('colspan') || '1', 10),
          rowspan: parseInt(td.getAttribute('rowspan') || '1', 10),
          bold: isBold(ccs.fontWeight),
          italic: ccs.fontStyle === 'italic',
          underline: (ccs.textDecorationLine || '').includes('underline'),
          strikethrough: (ccs.textDecorationLine || '').includes('line-through'),
          color: rgbToHex(ccs.color),
          backcolor: isTransparent(ccs.backgroundColor) ? '' : rgbToHex(ccs.backgroundColor),
          fontsize: Math.round(pxNum(ccs.fontSize)),
          fontname: (ccs.fontFamily || '').split(',')[0].trim().replace(/^["']|["']$/g, ''),
          align: mapAlign(ccs.textAlign),
          vAlign: ccs.verticalAlign === 'middle'
            ? 'middle'
            : (ccs.verticalAlign === 'bottom' ? 'bottom' : 'top'),
        });
      });
      if (cells.length) physicalRows.push(cells);
    });
    // Convert physical HTML cells into PPTist's full rectangular matrix. A
    // rowspan/colspan cell occupies a rectangle; covered coordinates remain
    // empty placeholders because PPTist expects them in data[][].
    const rows = [], occupied = [], anchors = [];
    physicalRows.forEach((cells, r) => {
      if (!occupied[r]) occupied[r] = [];
      let ci = 0;
      cells.forEach(cell => {
        while (occupied[r][ci]) ci++;
        const rs = Math.max(1, cell.rowspan || 1), cs = Math.max(1, cell.colspan || 1);
        anchors.push({r, c: ci, cell});
        for (let rr = r; rr < r + rs; rr++) {
          if (!occupied[rr]) occupied[rr] = [];
          for (let cc = ci; cc < ci + cs; cc++) occupied[rr][cc] = true;
        }
        ci += cs;
      });
    });
    const logicalRows = Math.max(physicalRows.length, occupied.length);
    const logicalCols = Math.max(1, ...occupied.map(row => row ? row.length : 0));
    for (let r = 0; r < logicalRows; r++) {
      const out = Array.from({length: logicalCols}, (_, c) => ({text:'', colspan:1, rowspan:1, fontsize:14, covered:true}));
      anchors.filter(a => a.r === r).forEach(a => { out[a.c] = a.cell; });
      rows.push(out);
    }
    // Prefer the author-declared logical column widths. The first physical row
    // is not sufficient when it contains colspan cells.
    const declaredCols = Array.from(table.querySelectorAll(':scope > colgroup > col'));
    if (declaredCols.length === logicalCols) {
      const declared = declaredCols.map(col => {
        const width = col.getBoundingClientRect().width || parseFloat(col.getAttribute('width') || '0');
        return width > 0 ? width : 1;
      });
      colWidths.splice(0, colWidths.length, ...declared);
    } else if (colWidths.length !== logicalCols) {
      const equal = Math.max(1, box.w / logicalCols);
      colWidths.splice(0, colWidths.length, ...Array.from({length: logicalCols}, () => equal));
    }
    // Outline from a representative cell border.
    let borderColor = '#eeece1', borderWidth = 0;
    const firstCell = table.querySelector('td, th');
    if (firstCell) {
      const bcs = getComputedStyle(firstCell);
      borderWidth = pxNum(bcs.borderBottomWidth);
      if (borderWidth > 0) borderColor = rgbToHex(bcs.borderBottomColor);
    }
    const outline = { width: borderWidth, color: borderColor };

    // PPTist ignores a table element's stored height when rendering: its actual
    // height comes from one shared `cellMinHeight` plus wrapped cell content.
    // Preserve the existing overflow guard first: shrink fonts only when the
    // native table would run past the page bottom. Then, when the HTML
    // deliberately stretches a table (for example `height:100%` in a flex
    // region), solve for the shared minimum row height that best preserves the
    // occupied vertical area. This is the closest native PPTist can get because
    // its schema has no per-row height array.
    let fontScale = 1;
    const normTotal = colWidths.reduce((a, b) => a + b, 0) || 1;
    const normWidths = colWidths.map(w => w / normTotal);
    const elementWidth = box.w;
    const BOTTOM_MARGIN = 5;
    const avail = origin.height - box.y - BOTTOM_MARGIN;
    const targetHeight = Math.max(1, avail > 0 ? Math.min(box.h, avail) : box.h);
    let renderHeight = measurePptistTableHeight(rows, normWidths, elementWidth, outline, 1);
    if (avail > 0 && rows.length && renderHeight > avail) {
      const minFsRaw = Math.min.apply(null, rows.flatMap(r => r.map(c => c.fontsize || 14)));
      const minK = minFsRaw > 0 ? Math.min(1, 6 / minFsRaw) : 1;
      // Binary search the largest k in [minK, 1] whose rendered height fits the
      // available page space. Do not shrink merely to match a compact HTML box:
      // browser and PPTist font metrics differ, and that would reduce legibility.
      let lo = minK, hi = 1, best = minK;
      for (let i = 0; i < 14; i++) {
        const mid = (lo + hi) / 2;
        const h = measurePptistTableHeight(rows, normWidths, elementWidth, outline, mid);
        if (h <= avail) { best = mid; lo = mid; } else { hi = mid; }
      }
      fontScale = +best.toFixed(4);
    }

    const naturalMetrics = measurePptistTable(rows, normWidths, elementWidth, 1, outline, fontScale);
    renderHeight = naturalMetrics.total;
    let cellMinHeight = naturalMetrics.rowHeights.length
      ? Math.max(1, Math.floor(Math.min.apply(null, naturalMetrics.rowHeights)))
      : 1;

    // A larger shared minimum is only needed when the HTML table intentionally
    // occupies more height than its text naturally requires. The simulator is
    // monotonic in cellMinHeight, so a small binary search gives the closest
    // non-overflowing native-table height without page-specific heuristics.
    if (rows.length && renderHeight + 1 < targetHeight) {
      let lo = 1, hi = targetHeight, best = cellMinHeight;
      let bestMetrics = naturalMetrics;
      for (let i = 0; i < 14; i++) {
        const mid = (lo + hi) / 2;
        const metrics = measurePptistTable(
          rows, normWidths, elementWidth, mid, outline, fontScale
        );
        if (metrics.total <= targetHeight) {
          best = mid;
          bestMetrics = metrics;
          lo = mid;
        } else {
          hi = mid;
        }
      }
      cellMinHeight = +best.toFixed(3);
      renderHeight = bestMetrics.total;
    }

    return {
      kind: 'table', box, rows, colWidths,
      tableRef: table.getAttribute('data-ref') || '',
      theme: {
        color: table.getAttribute('data-theme-color') || '#67508F',
        rowHeader: table.getAttribute('data-theme-row-header') === '1',
        rowFooter: table.getAttribute('data-theme-row-footer') === '1',
        colHeader: table.getAttribute('data-theme-col-header') === '1',
        colFooter: table.getAttribute('data-theme-col-footer') === '1',
      },
      cellMinHeight,
      fontScale,
      renderHeight,
      outline,
    };
  };

  const walk = (el, origin, out) => {
    const tag = el.tagName.toLowerCase();
    if (tag === 'img') {
      const cs = getComputedStyle(el);
      const position = objectPositionRatios(cs.objectPosition);
      out.push({
        kind: 'image', box: relBox(el, origin),
        src: el.getAttribute('src') || '',
        objectFit: cs.objectFit,
        objectPosition: position,
        naturalWidth: el.naturalWidth || 0,
        naturalHeight: el.naturalHeight || 0,
        opacity: finiteNum(cs.opacity, 1),
        radius: pxNum(cs.borderTopLeftRadius),
      });
      return;
    }
    if (tag === 'svg') { out.push(readSvg(el, origin)); return; }
    if (tag === 'table') { out.push(readTable(el, origin)); return; }

    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;

    // Background rectangle (any non-page element with a visible fill).
    const bg = cs.backgroundColor;
    if (!isTransparent(bg)) {
      const box = relBox(el, origin);
      if (box.w > 0 && box.h > 0) {
        out.push({
          kind: 'rect', box,
          fill: rgbToHex(bg),
          opacity: colorAlpha(bg) * pxNum(cs.opacity || '1'),
          radius: pxNum(cs.borderTopLeftRadius),
        });
      }
    }

    // Explicit text-in-shape card: a container with data-group="1", a
    // fill-parent background <svg>, and text alongside it. Two sub-cases,
    // decided by the background svg:
    //  - single-primitive bg (rect/circle/path/...) -> ONE editable shape whose
    //    text.content carries all the text (moves as a single unit).
    //  - multi-primitive bg -> keep the svg verbatim as an image AND emit the
    //    text as its own box, tying both together with a shared groupId so they
    //    drag together while the text stays editable.
    const elBox = relBox(el, origin);
    if (elBox.w > 0 && elBox.h > 0 && el.getAttribute('data-group') === '1') {
      const bgSvg = findCoverBgSvg(el, elBox);
      if (bgSvg && el.querySelectorAll('img').length === 0
          && el.querySelectorAll('svg').length === 1
          && el.textContent && el.textContent.trim()) {
        const svgData = readSvg(bgSvg, origin);
        const horizontalAlign = horizontalAlignFromStyle(cs);
        const content = serializeParagraphs(el, bgSvg, horizontalAlign);
        const tbox = textUnionBox(el, origin, bgSvg);
        const valign = verticalAlignFromStyle(cs);
        const lh = pxNum(cs.lineHeight);
        const fs = pxNum(cs.fontSize);
        const lineHeight = (lh && fs) ? +(lh / fs).toFixed(3) : 1.2;
        const inset = textPaddingScalar(el, bgSvg);

        if (svgData.kind === 'shape') {
          svgData.textContent = content;
          svgData.textAlign = valign;
          svgData.textLineHeight = lineHeight;
          svgData.textDefaultColor = rgbToHex(cs.color);
          svgData.textDefaultFontName = (cs.fontFamily || '').split(',')[0].trim().replace(/^["']|["']$/g, '');
          svgData.textWordSpace = pxNum(cs.letterSpacing);
          svgData.box = elBox;   // shape spans the whole card (svg fills it)
          svgData.textInset = [inset, inset, inset, inset];
          out.push(svgData);
          return;
        }
        if (svgData.kind === 'svgimage') {
          const gid = nextGroupId();
          svgData.box = elBox;
          svgData.groupId = gid;
          out.push(svgData);
          // Text box positioned over the text's own union box, grouped with svg.
          const tb = tbox || elBox;
          out.push({
            kind: 'text', box: tb,
            content: serializeRuns(el),
            richContent: content,
            align: horizontalAlign,
            vAlign: valign,
            lineHeight,
            defaultColor: rgbToHex(cs.color),
            defaultFontName: (cs.fontFamily || '').split(',')[0].trim().replace(/^["']|["']$/g, ''),
            groupId: gid,
            inset: [inset, inset, inset, inset],
            wordSpace: pxNum(cs.letterSpacing),
            opacity: finiteNum(cs.opacity, 1),
          });
          return;
        }
      }
    }

    // A component containing exactly one referenced text item may use a direct,
    // fill-parent SVG as its visual background (for example a colored title
    // bar). It remains an editable background shape plus a text box spanning the
    // component. Finding the reference anywhere inside avoids coupling this
    // conversion rule to the model's exact wrapper depth.
    const contentRef = soleContentRef(el);
    if (elBox.w > 0 && elBox.h > 0 && contentRef
        && el.getAttribute('data-group') !== '1'
        && el.querySelectorAll('img, table').length === 0
        && el.textContent && el.textContent.trim()) {
      const bgSvg = findCoverBgSvg(el, elBox);
      if (bgSvg && el.querySelectorAll('svg').length === 1) {
        // Keep the outer element as the background/text box geometry, but read
        // alignment from a nested layout container such as a centered grid.
        // A nested text leaf has no useful container-level alignment of its own.
        const layoutStyle = isTextLeaf(contentRef) ? cs : getComputedStyle(contentRef);
        out.push(readSvg(bgSvg, origin));
        const inset = textPaddingScalar(el, bgSvg);
        const lh = pxNum(cs.lineHeight);
        const fs = pxNum(cs.fontSize);
        out.push({
          kind: 'text', box: elBox,
          content: serializeRuns(el),
          richContent: serializeParagraphs(el, bgSvg, horizontalAlignFromStyle(cs)),
          align: horizontalAlignFromStyle(cs),
          vAlign: verticalAlignFromStyle(layoutStyle),
          lineHeight: (lh && fs) ? +(lh / fs).toFixed(3) : 1.2,
          defaultColor: rgbToHex(cs.color),
          defaultFontName: (cs.fontFamily || '').split(',')[0].trim().replace(/^["']|["']$/g, ''),
          inset: [inset, inset, inset, inset],
          wordSpace: pxNum(cs.letterSpacing),
          opacity: finiteNum(cs.opacity, 1),
        });
        return;
      }
    }

    if (isTextLeaf(el)) {
      const box = relBox(el, origin);
      if (box.w > 0 && box.h > 0) {
        const lh = pxNum(cs.lineHeight);
        const fs = pxNum(cs.fontSize);
        out.push({
          kind: 'text', box,
          content: serializeRuns(el),
          align: horizontalAlignFromStyle(cs),
          vAlign: verticalAlignFromStyle(cs),
          lineHeight: (lh && fs) ? +(lh / fs).toFixed(3) : 1.2,
          defaultColor: rgbToHex(cs.color),
          defaultFontName: (cs.fontFamily || '').split(',')[0].trim().replace(/^["']|["']$/g, ''),
          inset: [paddingScalar(cs), paddingScalar(cs), paddingScalar(cs), paddingScalar(cs)],
          wordSpace: pxNum(cs.letterSpacing),
          opacity: finiteNum(cs.opacity, 1),
        });
      }
      return;
    }

    emitDirectTextNodes(el, origin, out);

    for (const c of el.children) walk(c, origin, out);
  };

  const pages = [];
  const pageEls = document.querySelectorAll('.page, .baseline-page');
  for (const pageEl of pageEls) {
    const r = pageEl.getBoundingClientRect();
    const origin = { left: r.left, top: r.top, width: r.width, height: r.height };
    const cs = getComputedStyle(pageEl);
    const bg = isTransparent(cs.backgroundColor) ? '#ffffff' : rgbToHex(cs.backgroundColor);
    const out = [];
    for (const c of pageEl.children) walk(c, origin, out);
    pages.push({ width: r.width, height: r.height, background: bg, elements: out });
  }
  return pages;
}
"""


# The fixed-size page contract from the pipeline's css_library.css. Every step3
# page depends on it, but a standalone chunk (base href="../") can't always load
# that stylesheet, so we re-assert the invariant before measuring. Applies to all
# pages uniformly; it is the base contract, not page-specific tuning.
#
# We also force PPTist's own text font stack ($textElementFont) onto every node.
# Rationale: converted text elements carry no font-family (defaultFontName is ""),
# so PPTist renders them with $textElementFont regardless of what css_library used.
# Measuring with a different stack (e.g. Calibri) bakes a height that no longer
# matches PPTist's render, so a fixedHeight box would clip. Measuring with the same
# stack keeps the bake WYSIWYG with the editor.
_PPTIST_TEXT_FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, '
    '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif, '
    "'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'"
)
_PAGE_CONTRACT_CSS = f"""
html, body {{ margin: 0; padding: 0; }}
* {{ box-sizing: border-box; }}
.page {{ width: 959.76pt; height: 540pt; margin: 0; overflow: hidden; }}
.page {{ font-family: {_PPTIST_TEXT_FONT}; }}
/* Blast-radius guard: a decorative SVG (divider/background) whose absolute
   positioning failed to apply (e.g. a selector that doesn't match the DOM)
   drops into flow. With preserveAspectRatio="none" + a tall viewBox +
   height:100%, an undefined-height parent makes it resolve to viewBox ratio and
   balloon to thousands of px, shoving every following element off the page. We
   can't restore the intended layout here (that's step3's job), but capping any
   in-page svg to the page height keeps a single bad svg from exploding the whole
   page geometry to ~20000px. */
.page svg {{ max-height: 540pt; }}
"""


def measure_html(html_path: str | Path) -> list[dict]:
    """Render the HTML file and return baked visual primitives per page."""
    html_path = Path(html_path).resolve()
    url = html_path.as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.add_style_tag(content=_PAGE_CONTRACT_CSS)
        # Give web fonts / images a beat so auto-height boxes settle.
        page.wait_for_timeout(200)
        pages = page.evaluate(_EXTRACT_JS)
        browser.close()
    return pages


# Rebuilds each standalone text element's DOM like PPTist's TextElement/index.vue
# (`.element-content` with the element's lineHeight ratio + inset padding, holding
# a `.ProseMirror-static { font-size:16px }` that renders the content HTML), and
# reports the real content height. This is the authority we bake into
# `fixedHeight` boxes so the editor never re-measures to a different height.
_TEXT_HEIGHT_JS = r"""
(items) => {
  const out = {};
  for (const it of items) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'position:absolute;left:-99999px;top:0;';
    wrap.style.width = it.width + 'px';

    const content = document.createElement('div');
    // Mirror .element-content: relative, break-word, the element font stack and
    // the element's (compensated) unitless lineHeight; padding == inset.
    content.style.cssText = 'position:relative;word-break:break-word;box-sizing:border-box;';
    content.style.width = it.width + 'px';
    content.style.fontFamily = it.font;
    content.style.lineHeight = String(it.lineHeight);
    content.style.letterSpacing = (it.wordSpace || 0) + 'px';
    const ins = it.inset || [0, 0, 0, 0];
    content.style.padding = `${ins[0]}px ${ins[1]}px ${ins[2]}px ${ins[3]}px`;

    // Mirror .ProseMirror-static: fixed 16px block font, wrapping rules.
    const pm = document.createElement('div');
    pm.className = 'ProseMirror-static';
    pm.style.fontSize = '16px';
    pm.style.overflowWrap = 'break-word';
    pm.style.wordBreak = 'normal';
    pm.style.whiteSpace = 'normal';
    // Prosemirror paragraph margins come from --paragraphSpace (default 5px, or
    // whatever the shape text set). Plain text elements have no paragraphSpace
    // override, so keep the PPTist default of 5px between <p>s.
    pm.style.setProperty('--paragraphSpace', (it.paragraphSpace ?? 5) + 'px');
    pm.innerHTML = it.content;
    // Apply the p-margin rule ProseMirror uses (first child 0, rest paragraphSpace).
    const ps = pm.querySelectorAll('p, ul, ol, li');
    ps.forEach((p, i) => { p.style.margin = '0'; if (i > 0) p.style.marginTop = 'var(--paragraphSpace)'; });

    content.appendChild(pm);
    wrap.appendChild(content);
    document.body.appendChild(wrap);
    out[it.id] = { height: content.getBoundingClientRect().height };
    document.body.removeChild(wrap);
  }
  return out;
}
"""


def measure_pptist_text_heights(doc: dict) -> dict[str, dict]:
    """Return PPTist-rendered heights for standalone text elements."""
    items: list[dict] = []
    for slide in doc["slides"]:
        for el in slide["elements"]:
            if el["type"] == "text" and el.get("content"):
                items.append({
                    "id": el["id"],
                    "font": el.get("defaultFontName") or _PPTIST_TEXT_FONT,
                    "width": el["width"],
                    "content": el["content"],
                    "lineHeight": el.get("lineHeight", 1.2),
                    "wordSpace": el.get("wordSpace", 0),
                    "inset": el.get("inset") or [0, 0, 0, 0],
                    "paragraphSpace": 5,
                })
    if not items:
        return {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        page.set_content("<!doctype html><html><head><meta charset='utf-8'></head><body></body></html>")
        page.add_style_tag(content=_PAGE_CONTRACT_CSS)
        page.wait_for_timeout(50)
        rendered = page.evaluate(_TEXT_HEIGHT_JS, items)
        browser.close()
    return rendered


if __name__ == "__main__":
    import sys

    result = measure_html(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
