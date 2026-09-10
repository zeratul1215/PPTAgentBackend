"""
Step 4 (MVP): non-LLM overflow QA + autoshrink repair on Step 3 bundle.

Vendored into `agent_backend` so the LangGraph pipeline is self-contained.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from agent_backend.agent.tools.heavy_tool_impl.single_page_edition.step_3_reassemble.step3_reassemble_mvp import (
    _assemble_preview_html,
    _extract_page_div_from_chunk_html,
)


CHUNK_RE = re.compile(r"chunk_(\d+)_(\d+)\.html$", flags=re.I)


AUTO_SHRINK_JS = r"""
async (opts) => {
  const minFontPt = Number(opts?.minFontPt ?? 8.0);
  const maxIters = Math.max(1, Math.min(16, Number(opts?.maxIters ?? 8)));
  const minScale = Math.max(0.2, Math.min(1.0, Number(opts?.minScale ?? 0.5)));
  const minLineHeightMul = Math.max(
    1.0,
    Math.min(2.0, Number(opts?.minLineHeightMul ?? 1.15)),
  );
  const PT_TO_PX = 96 / 72;
  const minPx = minFontPt * PT_TO_PX;

  function nextFrame() {
    return new Promise((r) => requestAnimationFrame(() => r()));
  }

  function hasDirectText(el) {
    for (const n of (el.childNodes || [])) {
      if (n.nodeType === Node.TEXT_NODE && (n.textContent || "").trim()) return true;
    }
    return false;
  }

  function isInTable(el) {
    return !!(el.closest && el.closest("table,td,th"));
  }

  function isNonEmptyTextElement(el) {
    if (!el || !el.tagName) return false;
    const tag = el.tagName.toLowerCase();
    if (tag === "script" || tag === "style" || tag === "noscript") return false;
    if (isInTable(el)) return false;
    if (!hasDirectText(el)) return false;
    const rect = el.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return false;
    return true;
  }

  function collectItems(pg) {
    const items = [];
    const groups = new Map();
    const fallbackId = "__page_fallback__";
    for (const el of pg.querySelectorAll("*")) {
      if (!isNonEmptyTextElement(el)) continue;
      const groupEl = el.closest("[data-autofit-group]");
      const groupId = groupEl
        ? String(groupEl.getAttribute("data-autofit-group") || "").trim() || fallbackId
        : fallbackId;
      const sync = groupEl
        ? String(groupEl.getAttribute("data-autofit-sync") || "").trim()
        : "";
      if (!groups.has(groupId)) {
        groups.set(groupId, {
          id: groupId,
          sync,
          element: groupEl || pg,
          indices: [],
        });
      }
      const cs = getComputedStyle(el);
      let lh = 0;
      if (cs.lineHeight && cs.lineHeight !== "normal") {
        const parsed = parseFloat(cs.lineHeight);
        lh = Number.isFinite(parsed) ? parsed : 0;
      }
      groups.get(groupId).indices.push(items.length);
      items.push({
        el,
        groupId,
        origFontPx: parseFloat(cs.fontSize || "0") || 0,
        origLineHeightPx: lh,
      });
    }
    return { items, groups: Array.from(groups.values()) };
  }

  function applyScales(items, groupScales) {
    for (const item of items) {
      const scale = Math.max(minScale, Math.min(1.0, Number(groupScales[item.groupId] ?? 1.0)));
      let newFontPx = null;
      if (item.origFontPx > 0) {
        const px = Math.max(minPx, item.origFontPx * scale);
        newFontPx = Math.min(item.origFontPx, px);
        item.el.style.fontSize = `${newFontPx}px`;
      }
      if (item.origLineHeightPx > 0) {
        const fontRef = newFontPx ?? item.origFontPx;
        const lhFloor = fontRef > 0 ? fontRef * minLineHeightMul : 0;
        const lhScaled = item.origLineHeightPx * scale;
        const newLh = Math.min(item.origLineHeightPx, Math.max(lhFloor, lhScaled));
        item.el.style.lineHeight = `${newLh}px`;
      }
    }
  }

  function unionRects(rects) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const r of rects) {
      if (!r || r.width <= 0 || r.height <= 0) continue;
      minX = Math.min(minX, r.left);
      minY = Math.min(minY, r.top);
      maxX = Math.max(maxX, r.right);
      maxY = Math.max(maxY, r.bottom);
    }
    if (minX === Infinity) return null;
    return { left: minX, top: minY, right: maxX, bottom: maxY, width: maxX - minX, height: maxY - minY };
  }

  function itemTextRect(item) {
    const rects = [];
    for (const node of item.el.childNodes || []) {
      if (node.nodeType !== Node.TEXT_NODE || !(node.textContent || "").trim()) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      for (const r of range.getClientRects()) rects.push(r);
    }
    return unionRects(rects) || item.el.getBoundingClientRect();
  }

  function overlapArea(a, b) {
    if (!a || !b) return 0;
    const x = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
    const y = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    return x * y;
  }

  function safety(pg, items, groups) {
    const pageRect = pg.getBoundingClientRect();
    const pageOverflowX = pg.scrollWidth > pg.clientWidth + 1;
    const pageOverflowY = pg.scrollHeight > pg.clientHeight + 1;
    const textOutOfPage = [];
    const localOverflow = [];
    const groupRects = new Map();

    for (const group of groups) {
      const rects = group.indices.map(i => itemTextRect(items[i]));
      const u = unionRects(rects);
      if (u) groupRects.set(group.id, u);
      const el = group.element;
      if (el && el !== pg) {
        const cs = getComputedStyle(el);
        const constrained = cs.overflow !== "visible" || cs.overflowX !== "visible" || cs.overflowY !== "visible";
        if (constrained && (el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1)) {
          localOverflow.push({
            group: group.id,
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth,
            scrollHeight: el.scrollHeight,
            clientHeight: el.clientHeight,
          });
        }
      }
    }

    for (let i = 0; i < items.length; i++) {
      const r = itemTextRect(items[i]);
      if (r.left < pageRect.left - 1 || r.top < pageRect.top - 1
          || r.right > pageRect.right + 1 || r.bottom > pageRect.bottom + 1) {
        textOutOfPage.push({ index: i, group: items[i].groupId });
      }
    }

    const overlaps = [];
    for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        const a = groupRects.get(groups[i].id);
        const b = groupRects.get(groups[j].id);
        const area = overlapArea(a, b);
        if (area > 2) overlaps.push({ a: groups[i].id, b: groups[j].id, area: Math.round(area) });
      }
    }

    const ok = !pageOverflowX && !pageOverflowY && !textOutOfPage.length && !localOverflow.length && !overlaps.length;
    return {
      ok,
      pageOverflowX,
      pageOverflowY,
      scrollWidth: pg.scrollWidth,
      clientWidth: pg.clientWidth,
      scrollHeight: pg.scrollHeight,
      clientHeight: pg.clientHeight,
      textOutOfPage,
      localOverflow,
      overlaps,
    };
  }

  function cloneScales(scales) {
    return Object.assign({}, scales);
  }

  const pages = Array.from(document.querySelectorAll(".page"));
  if (!pages.length) return { pages: 0, adjusted: 0, perPage: [] };

  let adjusted = 0;
  const perPage = [];
  await nextFrame();

  for (let idx = 0; idx < pages.length; idx++) {
    const pg = pages[idx];
    const { items, groups } = collectItems(pg);
    if (!items.length) {
      perPage.push({ idx, adjusted: false, scale: 1.0, reason: "no_text_leaves" });
      continue;
    }

    const originalScales = {};
    for (const group of groups) originalScales[group.id] = 1.0;
    applyScales(items, originalScales);
    await nextFrame();
    const initialSafety = safety(pg, items, groups);
    if (initialSafety.ok) {
      perPage.push({ idx, adjusted: false, scale: 1.0, baselineSafe: true, groups: [] });
      continue;
    }

    let lo = minScale;
    let hi = 1.0;
    let best = minScale;
    let bestSafety = null;
    for (let it = 0; it < maxIters; it++) {
      const mid = (lo + hi) / 2;
      const candidate = {};
      for (const group of groups) candidate[group.id] = mid;
      applyScales(items, candidate);
      await nextFrame();
      const s = safety(pg, items, groups);
      if (s.ok) {
        best = mid;
        bestSafety = s;
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const scales = {};
    for (const group of groups) scales[group.id] = best;
    applyScales(items, scales);
    await nextFrame();
    let baselineSafety = bestSafety || safety(pg, items, groups);
    let unresolved = !baselineSafety.ok;

    const buckets = [];
    const seenBucket = new Set();
    for (const group of groups) {
      const bucketId = group.sync ? `sync:${group.sync}` : `group:${group.id}`;
      if (seenBucket.has(bucketId)) continue;
      seenBucket.add(bucketId);
      buckets.push({
        id: bucketId,
        sync: group.sync || "",
        groups: groups.filter(g => (group.sync ? g.sync === group.sync : g.id === group.id)).map(g => g.id),
        frozen: unresolved,
      });
    }

    if (!unresolved) {
      let changed = true;
      while (changed) {
        changed = false;
        for (const bucket of buckets) {
          if (bucket.frozen) continue;
          const cur = Math.min.apply(null, bucket.groups.map(g => scales[g]));
          if (cur >= 0.999) {
            bucket.frozen = true;
            continue;
          }
          const next = Math.min(1.0, cur + 0.05);
          const candidate = cloneScales(scales);
          for (const gid of bucket.groups) candidate[gid] = next;
          applyScales(items, candidate);
          await nextFrame();
          const s = safety(pg, items, groups);
          if (s.ok) {
            for (const gid of bucket.groups) scales[gid] = next;
            changed = true;
          } else {
            applyScales(items, scales);
            await nextFrame();
            bucket.frozen = true;
          }
        }
      }

      for (const bucket of buckets) {
        const start = Math.min.apply(null, bucket.groups.map(g => scales[g]));
        if (start >= 0.999) continue;
        let blo = start;
        let bhi = 1.0;
        let bbest = start;
        for (let it = 0; it < 6; it++) {
          const mid = (blo + bhi) / 2;
          const candidate = cloneScales(scales);
          for (const gid of bucket.groups) candidate[gid] = mid;
          applyScales(items, candidate);
          await nextFrame();
          const s = safety(pg, items, groups);
          if (s.ok) {
            bbest = mid;
            blo = mid;
          } else {
            bhi = mid;
          }
        }
        for (const gid of bucket.groups) scales[gid] = bbest;
        applyScales(items, scales);
        await nextFrame();
      }
    }

    applyScales(items, scales);
    await nextFrame();
    const finalSafety = safety(pg, items, groups);
    const didAdjust = Object.values(scales).some(v => v < 0.999);
    if (didAdjust) adjusted += 1;
    perPage.push({
      idx,
      adjusted: didAdjust,
      scale: Math.min.apply(null, Object.values(scales)),
      globalScale: best,
      baselineSafe: !unresolved,
      finalSafe: finalSafety.ok,
      unresolved,
      groups: groups.map(g => ({
        id: g.id,
        sync: g.sync || "",
        scale: Number((scales[g.id] ?? 1).toFixed(4)),
        textCount: g.indices.length,
      })),
      syncGroups: buckets.filter(b => b.sync).map(b => ({
        sync: b.sync,
        groups: b.groups,
        scale: Number((scales[b.groups[0]] ?? 1).toFixed(4)),
      })),
      remaining: finalSafety.ok ? null : finalSafety,
    });
  }

  return { pages: pages.length, adjusted, perPage: perPage };
}
"""


PAGE_OVERFLOW_QA_JS = r"""
() => {
  const pages = Array.from(document.querySelectorAll(".page"));
  const bad = [];
  for (let idx = 0; idx < pages.length; idx++) {
    const pg = pages[idx];
    const overH = pg.scrollHeight > pg.clientHeight + 1;
    const overW = pg.scrollWidth > pg.clientWidth + 1;
    if (!overH && !overW) continue;
    bad.push({
      idx,
      id: pg.id || "",
      scrollHeight: pg.scrollHeight,
      clientHeight: pg.clientHeight,
      scrollWidth: pg.scrollWidth,
      clientWidth: pg.clientWidth,
    });
  }
  return { pages: pages.length, bad };
}
"""


SERIALIZE_PAGES_JS = r"""
() => {
  const pages = Array.from(document.querySelectorAll(".page"));
  return pages.map((pg) => pg.outerHTML);
}
"""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _chunk_page_no(chunk_path: Path) -> int | None:
    m = CHUNK_RE.search(chunk_path.name)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a != b:
        return None
    return a


def _replace_page_block_in_chunk_html(*, chunk_html: str, new_page_div_outer: str) -> str:
    new_block = "<!-- PAGE_START -->\n" + new_page_div_outer.strip() + "\n<!-- PAGE_END -->"
    if not re.search(r"(?is)<!--\s*PAGE_START\s*-->", chunk_html or ""):
        raise ValueError("PAGE_START not found in chunk")
    if not re.search(r"(?is)<!--\s*PAGE_END\s*-->", chunk_html or ""):
        raise ValueError("PAGE_END not found in chunk")
    return re.sub(
        r"(?is)<!--\s*PAGE_START\s*-->.*?<!--\s*PAGE_END\s*-->",
        lambda _m: new_block,
        chunk_html,
        count=1,
    )


def _qa_overflow(*, pw_page: Any) -> list[dict[str, Any]]:
    result = pw_page.evaluate(PAGE_OVERFLOW_QA_JS) or {}
    bad = result.get("bad") if isinstance(result, dict) else []
    return bad if isinstance(bad, list) else []


def _run_autoshrink(*, pw_page: Any, opts: dict[str, Any]) -> dict[str, Any]:
    result = pw_page.evaluate(AUTO_SHRINK_JS, opts) or {}
    return result if isinstance(result, dict) else {}


def _serialize_pages(*, pw_page: Any) -> list[str]:
    out = pw_page.evaluate(SERIALIZE_PAGES_JS) or []
    return [s for s in out if isinstance(s, str)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4 (MVP): non-LLM overflow QA + autoshrink repair on Step 3 bundle.",
    )
    p.add_argument(
        "bundle_dir",
        type=str,
        help="Step 3 output bundle directory (must contain chunks/ and css_library.css).",
    )
    p.add_argument(
        "--no-shrink",
        action="store_true",
        help="Detection only. Do not run autoshrink; do not modify chunks.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="PPTAgent",
        help="<title> used when index.html is rebuilt (default: PPTAgent).",
    )
    p.add_argument("--autoshrink-min-font-pt", type=float, default=8.0)
    p.add_argument("--autoshrink-min-scale", type=float, default=0.5)
    p.add_argument("--autoshrink-max-iters", type=int, default=8)
    p.add_argument("--autoshrink-min-line-height-mul", type=float, default=1.15)
    return p.parse_args()


def _run_qa_on_index(
    *,
    pw_page: Any,
    index_html_path: Path,
    autoshrink_opts: dict[str, Any] | None,
) -> dict[str, Any]:
    pw_page.goto(index_html_path.as_uri(), wait_until="load")
    pw_page.evaluate("() => (document.fonts ? document.fonts.ready : true)")

    before = _qa_overflow(pw_page=pw_page)
    shrink_info: dict[str, Any] = {}
    after: list[dict[str, Any]] = before
    serialised: list[str] = []

    if autoshrink_opts is not None:
        shrink_info = _run_autoshrink(pw_page=pw_page, opts=autoshrink_opts)
        after = _qa_overflow(pw_page=pw_page)
        serialised = _serialize_pages(pw_page=pw_page)
    else:
        serialised = _serialize_pages(pw_page=pw_page)

    return {
        "before": before,
        "after": after,
        "shrink_info": shrink_info,
        "serialised_pages": serialised,
    }


def _index_pages_in_bundle(bundle_dir: Path) -> list[tuple[int, Path]]:
    chunks_dir = bundle_dir / "chunks"
    if not chunks_dir.exists():
        raise SystemExit(f"chunks/ not found under: {bundle_dir}")
    pairs: list[tuple[int, Path]] = []
    for cf in sorted(chunks_dir.glob("chunk_*.html")):
        pno = _chunk_page_no(cf)
        if pno is None:
            continue
        try:
            html_text = _read_text(cf)
        except Exception:
            continue
        if not _extract_page_div_from_chunk_html(html_text):
            continue
        pairs.append((pno, cf))
    if not pairs:
        raise SystemExit(f"No usable chunk_*.html files under: {chunks_dir}")
    return pairs


def _build_report(
    *,
    bundle_dir: Path,
    pages: list[tuple[int, Path]],
    qa: dict[str, Any],
    autoshrink_opts: dict[str, Any] | None,
    wrote_chunks: list[int],
) -> dict[str, Any]:
    before = qa.get("before") or []
    after = qa.get("after") or []
    shrink_info = qa.get("shrink_info") or {}

    def _bad_idx_set(bad: list[Any]) -> set[int]:
        out: set[int] = set()
        for x in bad:
            if isinstance(x, dict) and isinstance(x.get("idx"), int):
                out.add(int(x["idx"]))
        return out

    before_idx = _bad_idx_set(before if isinstance(before, list) else [])
    after_idx = _bad_idx_set(after if isinstance(after, list) else [])

    per_page_info: list[dict[str, Any]] = []
    per_page_shrink: list[dict[str, Any]] = (
        shrink_info.get("perPage") if isinstance(shrink_info.get("perPage"), list) else []
    )
    per_page_shrink_by_idx: dict[int, dict[str, Any]] = {
        int(x["idx"]): x for x in per_page_shrink if isinstance(x, dict) and isinstance(x.get("idx"), int)
    }

    for order_idx, (page_no, chunk_path) in enumerate(pages):
        overflow_before = order_idx in before_idx
        overflow_after = order_idx in after_idx
        sh = per_page_shrink_by_idx.get(order_idx) or {}
        per_page_info.append(
            {
                "page_no": page_no,
                "chunk": chunk_path.name,
                "overflow_before": overflow_before,
                "overflow_after_shrink": overflow_after,
                "shrink_applied": bool(sh.get("adjusted")) if sh else False,
                "shrink_scale": float(sh.get("scale")) if isinstance(sh.get("scale"), (int, float)) else None,
                "global_scale": float(sh.get("globalScale")) if isinstance(sh.get("globalScale"), (int, float)) else None,
                "baseline_safe": sh.get("baselineSafe") if isinstance(sh.get("baselineSafe"), bool) else None,
                "final_safe": sh.get("finalSafe") if isinstance(sh.get("finalSafe"), bool) else None,
                "unresolved": bool(sh.get("unresolved")) if sh else False,
                "autofit_groups": sh.get("groups") if isinstance(sh.get("groups"), list) else [],
                "autofit_sync_groups": sh.get("syncGroups") if isinstance(sh.get("syncGroups"), list) else [],
                "remaining": sh.get("remaining") if isinstance(sh.get("remaining"), dict) else None,
                "chunk_rewritten": page_no in set(wrote_chunks),
            }
        )

    return {
        "bundle_dir": str(bundle_dir),
        "autoshrink": autoshrink_opts,
        "summary": {
            "pages": len(pages),
            "overflow_before": len(before_idx),
            "overflow_after_shrink": len(after_idx) if autoshrink_opts is not None else None,
            "shrink_applied_pages": int(shrink_info.get("adjusted") or 0) if autoshrink_opts is not None else 0,
        },
        "pages": per_page_info,
    }


def main() -> int:
    args = _parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists():
        raise SystemExit(f"bundle_dir not found: {bundle_dir}")
    css_path = bundle_dir / "css_library.css"
    if not css_path.exists():
        raise SystemExit(f"css_library.css not found under bundle_dir: {css_path}")

    pages = _index_pages_in_bundle(bundle_dir)

    autoshrink_opts: dict[str, Any] | None = None
    if not bool(args.no_shrink):
        autoshrink_opts = {
            "minFontPt": float(args.autoshrink_min_font_pt),
            "minScale": float(args.autoshrink_min_scale),
            "maxIters": int(args.autoshrink_max_iters),
            "minLineHeightMul": float(args.autoshrink_min_line_height_mul),
        }

    index_html_path = bundle_dir / "index.html"
    if not index_html_path.exists():
        _assemble_preview_html(out_bundle_dir=bundle_dir, title=str(args.title))

    wrote_chunks: list[int] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            pw_page = browser.new_page(viewport={"width": 1280, "height": 720})
            qa = _run_qa_on_index(
                pw_page=pw_page,
                index_html_path=index_html_path,
                autoshrink_opts=autoshrink_opts,
            )
        finally:
            browser.close()

    if autoshrink_opts is not None:
        serialised: list[str] = qa.get("serialised_pages") or []
        shrink_info = qa.get("shrink_info") or {}
        per_page_shrink: list[dict[str, Any]] = (
            shrink_info.get("perPage") if isinstance(shrink_info.get("perPage"), list) else []
        )
        adjusted_by_idx = {
            int(x["idx"]): bool(x.get("adjusted"))
            for x in per_page_shrink
            if isinstance(x, dict) and isinstance(x.get("idx"), int)
        }

        if len(serialised) == len(pages):
            for order_idx, (page_no, chunk_path) in enumerate(pages):
                if not adjusted_by_idx.get(order_idx, False):
                    continue
                page_outer = serialised[order_idx]
                if not isinstance(page_outer, str) or not page_outer.strip():
                    continue
                try:
                    original = _read_text(chunk_path)
                    new_html = _replace_page_block_in_chunk_html(
                        chunk_html=original, new_page_div_outer=page_outer
                    )
                except Exception:
                    continue
                _write_text(chunk_path, new_html)
                page_dir = bundle_dir / f"page_{page_no:03d}"
                if page_dir.exists():
                    new_block = "<!-- PAGE_START -->\n" + page_outer.strip() + "\n<!-- PAGE_END -->\n"
                    _write_text(page_dir / "page_block.html", new_block)
                wrote_chunks.append(page_no)

        if wrote_chunks:
            _assemble_preview_html(out_bundle_dir=bundle_dir, title=str(args.title))

    report = _build_report(
        bundle_dir=bundle_dir,
        pages=pages,
        qa=qa,
        autoshrink_opts=autoshrink_opts,
        wrote_chunks=wrote_chunks,
    )
    report_path = bundle_dir / "qa_report.json"
    _write_json(report_path, report)

    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
