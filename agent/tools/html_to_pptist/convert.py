"""CLI: convert a pipeline HTML file into a PPTist-importable JSON.

    python3 convert.py INPUT.html [-o OUTPUT.json] [--title TITLE]

Use the step4 HTML (chunk_after_step4.html) as INPUT: it already carries the
autoshrink QA results, so text is sized to fit the page before conversion.
Conversion preserves the measured layout and does not apply another page-wide
fit or shrink pass.

INPUT may be a single-page chunk or a multi-page bundle (index.html): every
top-level `.page` / `.baseline-page` becomes one slide. Local <img> srcs are
resolved relative to the HTML file and inlined as data URIs, so the produced
JSON is self-contained and can be handed straight to the PPTist frontend
(importJSON / setSlides) with no extra assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    # As a package inside agent_backend (normal runtime).
    from .mapping import build_pptist, finalize_text_heights
    from .measure import measure_html, measure_pptist_text_heights
except ImportError:  # pragma: no cover
    # As loose scripts run from this directory (standalone CLI use).
    from mapping import build_pptist, finalize_text_heights
    from measure import measure_html, measure_pptist_text_heights


def convert(html_path: str | Path, title: str | None = None,
            assets_dir: str | Path | None = None) -> dict:
    html_path = Path(html_path).resolve()
    pages = measure_html(html_path)
    if not pages:
        raise SystemExit(f"No .page / .baseline-page found in {html_path}")
    base_dir = Path(assets_dir).resolve() if assets_dir else html_path.parent
    doc = build_pptist(pages, base_dir=base_dir, title=title or html_path.stem)
    # Bake standalone text boxes to the height PPTist actually renders.
    # Strut-compensated lineHeight is already applied in the mapping layer.
    rendered = measure_pptist_text_heights(doc)
    finalize_text_heights(doc, rendered)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description="step4 HTML -> PPTist JSON")
    ap.add_argument("input", help="step4 HTML file (single chunk or bundle index.html)")
    ap.add_argument("-o", "--output", help="output JSON path (default: <input>.pptist.json)")
    ap.add_argument("--title", help="presentation title")
    ap.add_argument("--assets", help="base dir for resolving relative <img> src "
                                     "(default: the HTML file's own folder)")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output) if args.output else in_path.with_suffix(".pptist.json")

    doc = convert(in_path, title=args.title, assets_dir=args.assets)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    n_el = sum(len(s["elements"]) for s in doc["slides"])
    print(f"Wrote {out_path}")
    print(f"  slides={len(doc['slides'])}  elements={n_el}  size={doc['width']}x{doc['height']}")


if __name__ == "__main__":
    main()
