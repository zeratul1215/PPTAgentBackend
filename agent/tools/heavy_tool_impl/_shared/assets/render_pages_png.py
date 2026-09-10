#!/usr/bin/env python3
"""
Render every page of a PDF to standalone PNG files (full page, PDF-accurate).

Output files:
  <out_dir>/page_001.png
  <out_dir>/page_002.png
  ...
  <out_dir>/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF


def render_pdf_pages(
    pdf_path: Path,
    out_dir: Path,
    *,
    dpi: float,
    alpha: bool,
) -> int:
    pdf_path = pdf_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    zoom = float(dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)

    meta_pages: list[dict[str, float | int | str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=alpha)
        name = f"page_{i + 1:03d}.png"
        out_path = out_dir / name
        pix.save(out_path.as_posix())
        meta_pages.append(
            {
                "file": name,
                "pdf_page_1based": i + 1,
                "pdf_page_0based": i,
                "width_px": pix.width,
                "height_px": pix.height,
                "width_pt": round(float(page.rect.width), 4),
                "height_pt": round(float(page.rect.height), 4),
            }
        )

    manifest = {
        "source_render_pdf": str(pdf_path),
        "dpi": float(dpi),
        "alpha": bool(alpha),
        "page_count": doc.page_count,
        "pages": meta_pages,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n = doc.page_count
    doc.close()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Render every page of a PDF to PNG files.")
    p.add_argument("pdf", type=str, help="Input PDF path")
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--dpi", type=float, default=150.0, help="DPI for rendering (default: 150)")
    p.add_argument("--alpha", action="store_true", help="Preserve alpha channel (default: false)")

    args = p.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out)
    n = render_pdf_pages(pdf_path, out_dir, dpi=float(args.dpi), alpha=bool(args.alpha))
    print(f"Wrote {n} pages to: {out_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
