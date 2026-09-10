"""
Step 1 (merged understand + recompose) — single Claude multimodal pass.

Vendored into `agent_backend` so the LangGraph pipeline is self-contained.

Backend: a strong multimodal model via an OpenAI-compatible proxy. Set
PPT_LLM_BASE_URL + PPT_LLM_API_KEY (or pass --api-key); model via --model /
PPT_LLM_MODEL.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = os.getenv("PPT_LLM_MODEL", "claude-opus-4-8")

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _reasoning_effort(default: str = "minimal") -> str | None:
    """OpenAI-style `reasoning_effort` sent with the step1.understand call.

    step1 is the most vision-reasoning-heavy stage (it rebuilds table structure
    and text grouping from the rendered page image), so `minimal` is a latency
    trade-off, not a free win. Controlled by `PPT_STEP1_REASONING_EFFORT`, which
    falls back to the shared `PPT_LLM_REASONING_EFFORT`. Set it to an empty
    string / `none` / `off` / `default` to omit the field entirely and use the
    proxy's default reasoning."""
    raw = os.getenv("PPT_STEP1_REASONING_EFFORT")
    if raw is None:
        raw = os.getenv("PPT_LLM_REASONING_EFFORT")
    if raw is None:
        return default
    val = raw.strip()
    if not val or val.lower() in {"none", "off", "default"}:
        return None
    return val


def _normalize_model_output(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return ""
    m = _FENCE_RE.search(t)
    if m:
        return (m.group(1) or "").strip()
    t = re.sub(r"(?is)^\s*```(?:json)?", "", t).strip()
    t = re.sub(r"(?is)```\s*$", "", t).strip()
    return t


def _resolve_backend(api_key: str | None) -> tuple[str, str]:
    base_url = (os.getenv("PPT_LLM_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError(
            "PPT_LLM_BASE_URL is not set (OpenAI-compatible endpoint, e.g. https://eoeo.xyz/v1)."
        )
    key = (api_key or os.getenv("PPT_LLM_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing API key. Provide --api-key or set PPT_LLM_API_KEY.")
    return base_url, key


def _downscale_jpeg(raw: bytes, *, max_edge: int = 1568, quality: int = 85) -> bytes:
    """Resize an image so its long edge is <= max_edge and re-encode as JPEG,
    to keep the multimodal request within the token/context . Falls back
    to the original bytes if PIL is unavailable or anything fails."""
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


def _call_claude_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    images: list[bytes] | None = None,
    labeled_images: list[tuple[str, bytes]] | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 2,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Returns (raw_text, parsed_obj_or_none, error)."""
    import httpx

    def _img_part(b: bytes) -> dict[str, Any]:
        jb = _downscale_jpeg(b)
        b64 = base64.b64encode(jb).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b in images or []:
        content.append(_img_part(b))
    for label, b in labeled_images or []:
        content.append({"type": "text", "text": label})
        content.append(_img_part(b))

    body = {
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
    url = base_url.rstrip("/") + "/chat/completions"

    last_err: str | None = None
    for attempt in range(max(0, int(retries)) + 1):
        t0 = time.monotonic()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=_llm_timeout_s())
            _log_llm_timing(tag="step1.understand", model=model, attempt=attempt, elapsed=time.monotonic() - t0, status=resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                raw = str(data["choices"][0]["message"]["content"] or "")
                try:
                    obj = json.loads(_normalize_model_output(raw))
                    return raw, obj if isinstance(obj, dict) else None, None
                except Exception as e:
                    return raw, None, f"json_parse_error: {type(e).__name__}: {e}"
            last_err = f"http_{resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as e:
            _log_llm_timing(tag="step1.understand", model=model, attempt=attempt, elapsed=time.monotonic() - t0, status=type(e).__name__)
            last_err = f"{type(e).__name__}: {e}"
        if attempt < int(retries):
            time.sleep(min(30.0, 2.0 * (2.0 ** attempt)))
    return "", None, f"model_call_failed: {last_err}"


def _safe_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def _iter_text_items_from_plan_page(plan_page: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    next_id = 0

    def emit(*, kind: str, text: str) -> None:
        nonlocal next_id
        t = (text or "").strip()
        if not t:
            return
        out.append(
            {
                "id": f"t{next_id}",
                "kind": str(kind or "body"),
                "text": t,
            }
        )
        next_id += 1

    def blocks_from(blk: dict[str, Any]) -> Iterable[tuple[str, str]]:
        kind = str(blk.get("kind") or "body")
        if kind == "bullets":
            for it in (blk.get("items") or []):
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    yield ("bullet_item", str(it.get("text") or ""))
        else:
            yield (kind, str(blk.get("text") or ""))

    for blk in (plan_page.get("blocks") or []):
        if not isinstance(blk, dict):
            continue
        for kind, txt in blocks_from(blk):
            emit(kind=kind, text=txt)

    return out


def _strip_ws(s: str) -> str:
    return _WS_RE.sub("", s or "")


def _validate_paragraphs(*, paragraphs: Any, items_by_id: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not isinstance(paragraphs, list) or not paragraphs:
        return False, ["paragraphs_missing_or_empty"]
    seen: set[str] = set()
    expected = set(items_by_id.keys())
    for idx, para in enumerate(paragraphs):
        if not isinstance(para, dict):
            reasons.append(f"para[{idx}]_not_object")
            continue
        sources = para.get("sources")
        if not isinstance(sources, list) or not sources:
            reasons.append(f"para[{idx}]_sources_missing")
            continue
        valid: list[str] = []
        for s in sources:
            if not isinstance(s, str) or s not in expected:
                reasons.append(f"para[{idx}]_unknown_source: {s!r}")
                continue
            if s in seen:
                reasons.append(f"para[{idx}]_duplicate_source: {s}")
                continue
            seen.add(s)
            valid.append(s)
        if not valid:
            reasons.append(f"para[{idx}]_no_valid_sources")
            continue
        text = para.get("text")
        if not isinstance(text, str) or not text.strip():
            reasons.append(f"para[{idx}]_text_empty")
            continue

        segs_any = para.get("segments")
        segments: list[str]
        if isinstance(segs_any, list) and all(isinstance(x, str) for x in segs_any) and segs_any:
            segments = [str(x) for x in segs_any]
        else:
            # Backward compatible: treat missing/invalid segments as a single segment.
            segments = [text]

        if not any(s.strip() for s in segments):
            reasons.append(f"para[{idx}]_segments_empty")
            continue

        concat_src = "".join(_strip_ws(str(items_by_id[s].get("text") or "")) for s in valid)
        concat_out = "".join(_strip_ws(s) for s in segments)
        if concat_src != concat_out:
            reasons.append(f"para[{idx}]_char_mismatch")
    missing = sorted(expected - seen)
    if missing:
        reasons.append(f"missing_ids: {missing[:20]}")
    return (not reasons), reasons


def _identity_paragraphs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": str(it.get("kind") or ""),
            "sources": [str(it.get("id") or "")],
            "text": str(it.get("text") or ""),
            "segments": [str(it.get("text") or "")],
        }
        for it in items
        if isinstance(it, dict) and str(it.get("id") or "")
    ]


def _paragraph_index_to_text_id(paragraphs: list[dict[str, Any]]) -> dict[int, str]:
    """Map each paragraph's index (into `paragraphs`) to the final `tN` id.

    Must mirror `_apply_paragraphs_to_texts`: a `tN` is assigned in order and ONLY
    to paragraphs that have at least one source id, so skipped paragraphs shift
    the counter. Returns {paragraph_index: "tN"} for the emitted paragraphs only.
    """
    mapping: dict[int, str] = {}
    new_idx = 0
    for p_idx, para in enumerate(paragraphs):
        if not isinstance(para, dict):
            continue
        sources = [s for s in (para.get("sources") or []) if isinstance(s, str)]
        if not sources:
            continue
        mapping[p_idx] = f"t{new_idx}"
        new_idx += 1
    return mapping


def _build_tables(
    *, raw_tables: Any, paragraphs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the model's `tables[]` and remap each cell's `para` (an index into
    `paragraphs`) to the corresponding final text id (`tN`).

    Drops malformed tables/cells (non-object, missing/negative dims, out-of-range
    or unmapped `para`, duplicate row/col) and reports why via warnings. Returns
    ([] , warnings) when there are no valid tables.
    """
    warnings: list[str] = []
    if not isinstance(raw_tables, list) or not raw_tables:
        return [], warnings
    idx_to_ref = _paragraph_index_to_text_id(paragraphs)
    out: list[dict[str, Any]] = []
    for t_i, tbl in enumerate(raw_tables):
        if not isinstance(tbl, dict):
            warnings.append(f"table[{t_i}]_not_object")
            continue
        try:
            rows = int(tbl.get("rows"))
            cols = int(tbl.get("cols"))
        except (TypeError, ValueError):
            warnings.append(f"table[{t_i}]_bad_dims")
            continue
        if rows <= 0 or cols <= 0:
            warnings.append(f"table[{t_i}]_nonpositive_dims")
            continue
        # A single-column "table" is just a list and carries no table meaning;
        # drop it so an over-eager TASK 1B can't leak a degenerate 1-col grid.
        if cols < 2:
            warnings.append(f"table[{t_i}]_single_column_dropped")
            continue
        raw_cells = tbl.get("cells")
        if not isinstance(raw_cells, list) or not raw_cells:
            warnings.append(f"table[{t_i}]_no_cells")
            continue
        seen_rc: set[tuple[int, int]] = set()
        cells_out: list[dict[str, Any]] = []
        for c_i, cell in enumerate(raw_cells):
            if not isinstance(cell, dict):
                warnings.append(f"table[{t_i}]_cell[{c_i}]_not_object")
                continue
            try:
                r = int(cell.get("row"))
                c = int(cell.get("col"))
                para = int(cell.get("para"))
            except (TypeError, ValueError):
                warnings.append(f"table[{t_i}]_cell[{c_i}]_bad_index")
                continue
            if not (0 <= r < rows and 0 <= c < cols):
                warnings.append(f"table[{t_i}]_cell[{c_i}]_out_of_bounds")
                continue
            ref = idx_to_ref.get(para)
            if not ref:
                warnings.append(f"table[{t_i}]_cell[{c_i}]_unmapped_para: {para}")
                continue
            if (r, c) in seen_rc:
                warnings.append(f"table[{t_i}]_cell[{c_i}]_duplicate_rc: {(r, c)}")
                continue
            seen_rc.add((r, c))
            cells_out.append({"row": r, "col": c, "ref": ref})
        if not cells_out:
            warnings.append(f"table[{t_i}]_no_valid_cells")
            continue
        out.append(
            {
                "id": str(tbl.get("id") or f"tbl{len(out)}"),
                "rows": rows,
                "cols": cols,
                "cells": cells_out,
            }
        )
    return out, warnings


def _apply_paragraphs_to_texts(*, items: list[dict[str, Any]], paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(it.get("id") or ""): it for it in items if isinstance(it, dict)}
    out: list[dict[str, Any]] = []
    for new_idx, para in enumerate(paragraphs):
        sources = [s for s in (para.get("sources") or []) if isinstance(s, str)]
        if not sources:
            continue
        head = by_id.get(sources[0]) or {}
        text = str(para.get("text") or "")
        segs_any = para.get("segments")
        segments: list[str]
        if isinstance(segs_any, list) and all(isinstance(x, str) for x in segs_any) and segs_any:
            segments = [str(x) for x in segs_any]
        else:
            segments = [text]
        item: dict[str, Any] = {
            "id": f"t{new_idx}",
            # `kind` is decided by step1 from the rendered image (TASK 1 RE-TYPE);
            # the build-plan geometric guess is no longer fed to the model nor used
            # as a fallback (symmetry with the pptx/JSON path, which has no kind at
            # all). Default to a neutral "body" if step1 omits it for a paragraph.
            "kind": str(para.get("kind") or "body"),
            # Keep `text` as a flat string. When segments are present (a real
            # multi-item list), text is defined as their concatenation.
            "text": "".join(segments),
            "sources": sources,
        }
        # Only carry `segments` when it encodes real internal structure (>=2
        # items). A length-1 segments list duplicates `text` and adds no value,
        # so it is dropped here as a deterministic backstop regardless of what
        # the model emitted. Downstream `_get_segments` falls back to `[text]`.
        if len(segments) > 1:
            item["segments"] = segments
        out.append(item)
    return out


_SYSTEM_PROMPT = """You are the "UNDERSTAND" stage of a single-page PPT-editing pipeline. You look at ONE rendered slide and produce a clean structured understanding of it. You do FOUR things in one pass.

INPUTS
- `page_png`: the full-page render of the slide (look at it).
- `asset_images`: the extracted raw images on the page, each preceded by its image id.
- `fragments`: the text on the page, already extracted deterministically from the source. Each fragment has an id (t0, t1, ...) and its text. These fragments are often OVER-SPLIT: one sentence/paragraph may be broken across several fragments because of line breaks. (Fragments carry NO role/kind hint — YOU assign every paragraph's `kind` purely by looking at the slide.)

TASK 1 — RECOMPOSE + RE-TYPE THE TEXT (`paragraphs`)
Merge the fragments back into natural reading paragraphs, and assign each paragraph the correct role by LOOKING at the slide.
- Merge fragments that clearly belong to one sentence/paragraph (e.g. a sentence split across a line break). Fragments are a flat, ordered list with no grouping; decide what belongs together purely by reading the slide.
- Judge `kind` from the visual (e.g. "title", "subheading", "body", "bullet_item", "caption", "label", "date"): decide it purely from what the slide shows. Fragments carry no kind hint.
- ABSOLUTE RULE — do NOT change any characters. You may only re-group and re-order-at-join existing text. When you merge fragment A then B, `text` MUST equal A's text followed by B's text (a single space at the join is allowed; nothing else added, removed, or substituted). Never translate, rewrite, fix typos, or change punctuation.
- Every fragment id MUST appear in EXACTLY ONE paragraph's `sources`. No missing, no duplicate, no invented ids. `sources[0]` is the head fragment.
- If two fragments are clearly separate lines/labels/titles, keep them as separate paragraphs (do NOT force-merge).

Within a paragraph, preserve its internal visual line/item structure via `segments` ONLY when the paragraph is a real multi-item list:
- `segments` is an array of strings that splits the paragraph into the ORIGINAL visible lines/items in reading order.
- OMIT the `segments` field entirely for a normal sentence/paragraph (anything that is a single line/item). A single-segment `segments` carries no information — do NOT emit it. In that case output only `text`.
- ONLY emit `segments` for a clear enumerated list (timeline/date list, numbered list, bullet list) that has TWO OR MORE items. Then use one segment per item/row so the structure survives reread, and do NOT collapse the list into one giant segment.
- When present, `segments` MUST have length >= 2, and `text` MUST equal the EXACT concatenation of all `segments` (no separators; do not insert newlines). Every character must come from the input fragments; you may only choose segmentation boundaries.

TASK 1B — DETECT TABLES (`tables`)
Decide, PURELY by LOOKING at the image, whether the slide contains a real TABLE. Be CONSERVATIVE: only emit a table when the slide shows UNMISTAKABLE table structure. When in doubt, output NO table.
- A real table REQUIRES clear VISUAL grid evidence, such as: visible row/column separator lines or cell borders, a ruled grid, a shaded header row or alternating row-band fills, or a genuine matrix of values organized into labelled rows AND columns. Visual evidence outweighs the text content — never infer a table from what the words say.
- A plain LIST is NOT a table. Ordinary bullet lists, numbered lists, and timelines (e.g. "2012.06 ... / 2013.10 ...") are LISTS even when their items line up neatly row-by-row. Row-by-row alignment ALONE is NOT enough; do NOT treat aligned text as a table just because it looks tidy or because you could split each line into a date + description.
- NEVER output a single-column table (`cols` == 1): one column is just a list, so it carries no table meaning. A table must have `cols` >= 2 with genuinely distinct columns.
- Do NOT rely on coordinates or fragment order to guess a grid; rely on the grid lines / cell borders / shading you actually SEE.
- For each table, output: an `id` (e.g. "tbl0"), `rows` (int), `cols` (int >= 2), and `cells` — a row-major list of { "row": r, "col": c, "para": <index into `paragraphs`> } (all 0-based). Omit empty cells.
- If ONE paragraph is a column of many rows (you emitted one `segment` per row in TASK 1), emit one cell PER ROW all pointing to that SAME `para` index; the `row` disambiguates which segment goes in which row. Prefer combining parallel columns (e.g. a date column + a text column + its translation column) into ONE table with multiple `col` values, not several separate tables.
- If there are NO tables (the common case), output `"tables": []`.

TASK 2 — DESCRIBE EACH IMAGE (`images`), English
- One English description per image id in `asset_images`. Cover: what it shows (high level), its role on the page (hero / supporting / icon / logo / chart / background), and a rough relative placement.
- Do NOT OCR or transcribe text inside images.

TASK 3 — DESCRIBE THE ORIGINAL LAYOUT (`original_layout_description_en`), English
- One short paragraph (3-6 sentences) on the original composition: structure, relative placement (top/bottom/left/right/center, bands, columns, cards, whitespace), and the dominant COLOUR TONE per distinct region in plain English ("warm orange top band", "near-white body"). No hex codes. No OCR of exact text.
- ALSO mention, BRIEFLY, any purely DECORATIVE graphics you see — color blocks/bands, cards or rounded rectangles behind text, circles/dots/nodes, connector lines, timeline axes, dividers, arrows, and similar simple shapes — noting their rough placement and color tone. Keep this secondary and understated: these shapes are decoration, not the meaning-bearing content. The genuinely important elements are the text, tables, and images; decorations are just context for how the page looks. Do NOT over-describe them or let them dominate the paragraph.

OUTPUT — STRICT JSON only, no markdown, no commentary:
{
  "paragraphs": [ { "kind": "body", "sources": ["t3","t2"], "text": "..." },
                  { "kind": "body", "sources": ["t5"], "text": "...", "segments": ["item 1","item 2"] } ],
  "tables": [ { "id": "tbl0", "rows": 6, "cols": 2, "cells": [ {"row":0,"col":0,"para":0}, {"row":0,"col":1,"para":1} ] } ],
  "images": [ { "id": "p1_i0", "description_en": "..." } ],
  "original_layout_description_en": "..."
}
"""


def understand_step(
    *,
    input_obj: dict[str, Any],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict[str, Any]:
    warnings: list[str] = []

    page_num = int(input_obj.get("page_num") or 0)
    if page_num <= 0:
        raise ValueError("page_num must be a positive 1-based integer")
    plan_page = input_obj.get("plan_page")
    if not isinstance(plan_page, dict):
        raise ValueError("plan_page must be an object")

    page_id = str(plan_page.get("page_id") or "")
    palette = plan_page.get("palette") if isinstance(plan_page.get("palette"), dict) else {}
    page_size_pt = (
        input_obj.get("page_size_pt") if isinstance(input_obj.get("page_size_pt"), dict) else {"w": None, "h": None}
    )

    bundle_dir_s = str(input_obj.get("bundle_dir") or "").strip()
    bundle_dir = Path(bundle_dir_s).expanduser().resolve() if bundle_dir_s else None
    page_png_path_s = str(input_obj.get("page_png_path") or "").strip()
    page_png_path = Path(page_png_path_s).expanduser().resolve() if page_png_path_s else None

    # 1) Deterministic text extraction (integrity anchor).
    items = _iter_text_items_from_plan_page(plan_page)
    items_by_id = {str(it["id"]): it for it in items}

    # 2) Images baseline from plan_page.
    images_out: list[dict[str, Any]] = []
    asset_list: list[tuple[str, Path]] = []
    for im in (plan_page.get("images") or []):
        if not isinstance(im, dict):
            continue
        iid = str(im.get("id") or "").strip()
        src = str(im.get("src") or "").strip()
        if not iid or not src:
            continue
        dw = _safe_float(im.get("display_w_pt"))
        dh = _safe_float(im.get("display_h_pt"))
        images_out.append(
            {
                "id": iid,
                "src": src,
                **({"display_w_pt": dw} if dw is not None else {}),
                **({"display_h_pt": dh} if dh is not None else {}),
                "description_en": "",
            }
        )
        if bundle_dir is not None:
            asset_list.append((iid, (bundle_dir / src).resolve()))

    original_layout_description_en = ""
    used_paragraphs: list[dict[str, Any]]
    tables_out: list[dict[str, Any]] = []

    if dry_run:
        warnings.append("dry_run_identity_recompose")
        used_paragraphs = _identity_paragraphs(items)
    else:
        base_url, key = _resolve_backend(api_key)

        labeled: list[tuple[str, bytes]] = []
        page_imgs: list[bytes] = []
        if page_png_path is not None and page_png_path.exists():
            page_imgs.append(page_png_path.read_bytes())
        else:
            warnings.append("missing_page_png")
        for iid, p in asset_list:
            if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                labeled.append((f"asset_image_for_id: {iid}", p.read_bytes()))
            else:
                warnings.append(f"missing_or_unsupported_asset: {iid}")

        payload = {
            "page_size_pt": page_size_pt,
            "palette": palette,
            "image_ids": [im["id"] for im in images_out],
            "fragments": [
                {"id": it["id"], "text": it["text"]}
                for it in items
            ],
        }
        user_text = "page_png is the first image. asset_images follow, each preceded by its id.\n\n" + json.dumps(
            payload, ensure_ascii=False, indent=2
        )

        _raw, obj, err = _call_claude_json(
            base_url=base_url,
            api_key=key,
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_text=user_text,
            images=page_imgs,
            labeled_images=labeled,
            max_tokens=8192,
        )
        if err or not isinstance(obj, dict):
            warnings.append(f"understand_call_error: {err or 'invalid_json'}")
            used_paragraphs = _identity_paragraphs(items)
        else:
            paragraphs = obj.get("paragraphs")
            ok, reasons = _validate_paragraphs(
                paragraphs=paragraphs if isinstance(paragraphs, list) else [], items_by_id=items_by_id
            )
            if not ok:
                warnings.append(f"recompose_validation_failed: {reasons[:8]}")
                used_paragraphs = _identity_paragraphs(items)
            else:
                used_paragraphs = [p for p in paragraphs if isinstance(p, dict)]

            tables_out, table_warns = _build_tables(
                raw_tables=obj.get("tables"), paragraphs=used_paragraphs
            )
            warnings.extend(table_warns)

            desc_by_id: dict[str, str] = {}
            for it in (obj.get("images") or []):
                if (
                    isinstance(it, dict)
                    and isinstance(it.get("id"), str)
                    and isinstance(it.get("description_en"), str)
                ):
                    desc_by_id[it["id"]] = it["description_en"].strip()
            for im in images_out:
                im["description_en"] = desc_by_id.get(str(im["id"]), "")
            missing_desc = [im["id"] for im in images_out if not im["description_en"]]
            if missing_desc:
                warnings.append(f"missing_image_descriptions_for_ids: {missing_desc[:30]}")

            layout_desc = obj.get("original_layout_description_en")
            if isinstance(layout_desc, str) and layout_desc.strip():
                original_layout_description_en = layout_desc.strip()
            else:
                warnings.append("original_layout_description_missing_or_empty")

    new_texts = _apply_paragraphs_to_texts(items=items, paragraphs=used_paragraphs)

    out: dict[str, Any] = {
        "schema_version": "understand_output_v1",
        "page_num": page_num,
        "page_id": page_id,
        "page_size_pt": page_size_pt,
        "palette": palette,
        "page_png_path": str(page_png_path) if page_png_path is not None else None,
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
        "texts": new_texts,
        "tables": tables_out,
        "images": images_out,
        "original_layout_description_en": original_layout_description_en,
        "recompose": {
            "schema_version": "recompose_v1",
            "input_fragment_count": len(items),
            "output_paragraph_count": len(new_texts),
            "merged_paragraph_count": max(0, len(items) - len(new_texts)),
        },
        "warnings": warnings,
    }
    return out


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("input JSON must be an object")
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Step1 merged understand+recompose (single Claude multimodal pass).")
    p.add_argument("--input", required=True, help="Path to an understand_input_v1 JSON file.")
    p.add_argument("--out", required=True, help="Path to write understand_output_v1 JSON.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--api-key", default="", help="Proxy API key (or env PPT_LLM_API_KEY).")
    p.add_argument("--dry-run", action="store_true", help="Skip the model; identity recompose + empty descriptions.")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")
    out_obj = understand_step(
        input_obj=_read_json(in_path),
        api_key=(args.api_key or None),
        model=str(args.model),
        dry_run=bool(args.dry_run),
    )
    _write_json(Path(args.out).expanduser().resolve(), out_obj)
    print(str(Path(args.out).expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

