"""
Beautify reference-image generation (Step 2.5).

When Step 2's resolved `visual_intent.enabled` is true, we ask an image model
(gpt-image-2 by default, via an OpenAI-compatible `/images/edits` proxy) to
render a high-quality reference image of the redesigned slide. Step 3 then uses
that image as a strong visual target, instead of free-styling the layout.

Design notes:
- The reference image is a *visual target*, not the source of truth for text /
  images. Step 3 still pulls verbatim text from `texts[]` and real images from
  their `src`. Colors follow the reference image (palette is the default tone,
  but a color the user explicitly requested overrides it).
- We feed the image model the user's ORIGINAL request (the design intent) plus
  the authoritative text blocks, the original page render, and the real embedded
  content images, and let the image model choose the composition itself. We do
  NOT send a pre-computed design brief or a structured layout tree (both only
  constrained the model without helping).
- Backend is resolved from PPT_IMAGE_BASE_URL / PPT_IMAGE_API_KEY (falling back
  to OPENAI_BASE_URL / OPENAI_API_KEY). Missing config raises RuntimeError so
  the caller can degrade gracefully rather than crashing the turn.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_IMAGE_MODEL = os.getenv("PPT_IMAGE_MODEL", "gpt-image-2")
DEFAULT_IMAGE_SIZE = os.getenv("PPT_IMAGE_SIZE", "1536x864")
# The reference image is only a layout/visual target for step 3 (which re-inserts
# all text verbatim from texts[] and never OCRs the image), so a lower quality
# tier is safe and meaningfully faster/cheaper upstream. Benchmarked on real page
# data: quality=low + downscaled webp inputs cut a turn from ~115s to ~87s with no
# usable-layout loss. Override via PPT_IMAGE_QUALITY if you want richer references.
DEFAULT_IMAGE_QUALITY = os.getenv("PPT_IMAGE_QUALITY", "low")
DEFAULT_IMAGE_FIELD = os.getenv("PPT_IMAGE_FIELD", "image[]")
# Return format for the generated reference image. Left as png by default: the
# output file is written to a fixed `beautify_reference.png` path, and the
# benchmark showed webp output barely shrinks the (base64) download body vs the
# real wins below (input downscaling + quality). Setting PPT_IMAGE_OUTPUT_FORMAT
# to "webp"/"jpeg" is supported (step3 sniffs content, not extension) but will
# write those bytes into the .png-named file, so only do it knowingly.
DEFAULT_IMAGE_OUTPUT_FORMAT = os.getenv("PPT_IMAGE_OUTPUT_FORMAT", "")
DEFAULT_IMAGE_OUTPUT_COMPRESSION = int(os.getenv("PPT_IMAGE_OUTPUT_COMPRESSION", "80"))
DEFAULT_MAX_CONTENT_IMAGES = 15
DEFAULT_MAX_TEXT_CHARS = 800
DEFAULT_MAX_STYLE_CHARS = 360
DEFAULT_MAX_ORIGINAL_LAYOUT_CHARS = 700
DEFAULT_MAX_IMAGE_DESC_CHARS = 280
DEFAULT_MAX_DEFAULT_DETAIL_CHARS = 220

# Input-image downscaling. The dominant cost of the /images/edits round trip is
# uploading the input images (a raw embedded photo can be 7MB+). The reference
# image only needs them as a visual guide, so we downscale to a sane long edge
# and re-encode as webp before upload. Benchmarked: 8.51MB -> 0.49MB upload,
# roughly halving the non-generation (transport) portion of the call. Set
# PPT_IMAGE_INPUT_MAX_EDGE=0 to disable and send originals untouched.
DEFAULT_INPUT_MAX_EDGE = int(os.getenv("PPT_IMAGE_INPUT_MAX_EDGE", "1536"))
DEFAULT_INPUT_WEBP_QUALITY = int(os.getenv("PPT_IMAGE_INPUT_WEBP_QUALITY", "80"))

# Transient-failure retry policy for the image backend. Runtime deployments
# should set PPT_IMAGE_TIMEOUT_S/PPT_IMAGE_MAX_RETRIES explicitly: too much retry
# time is painful for chat UX, while too little timeout can abort slow upstream
# generations before they return a usable image.
DEFAULT_IMAGE_MAX_RETRIES = int(os.getenv("PPT_IMAGE_MAX_RETRIES", "1"))
DEFAULT_IMAGE_RETRY_BACKOFF_S = float(os.getenv("PPT_IMAGE_RETRY_BACKOFF_S", "2.0"))
DEFAULT_IMAGE_TIMEOUT_S = float(os.getenv("PPT_IMAGE_TIMEOUT_S", "420.0"))
DEFAULT_IMAGE_URL_DOWNLOAD_TIMEOUT_S = float(os.getenv("PPT_IMAGE_URL_DOWNLOAD_TIMEOUT_S", "30.0"))
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _resolve_image_backend() -> tuple[str, str]:
    base_url = (os.environ.get("PPT_IMAGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
    api_key = (os.environ.get("PPT_IMAGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not base_url:
        raise RuntimeError(
            "Missing PPT_IMAGE_BASE_URL (OpenAI-compatible image endpoint, e.g. https://.../v1)."
        )
    if not api_key:
        raise RuntimeError("Missing PPT_IMAGE_API_KEY.")
    return base_url, api_key


def _mime_for(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def _compact_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _understand_modified(step2_output: dict[str, Any]) -> dict[str, Any]:
    compile_obj = step2_output.get("compile") if isinstance(step2_output.get("compile"), dict) else {}
    um = compile_obj.get("understand_modified") if isinstance(compile_obj.get("understand_modified"), dict) else {}
    return um if isinstance(um, dict) else {}


def _visual_requirements_text(step2_output: dict[str, Any]) -> str:
    """Return the planner's verbatim visual requirement from the resolved
    top-level `visual_intent`. Returns "" when absent."""
    vi = step2_output.get("visual_intent")
    if isinstance(vi, dict):
        req = vi.get("requirements_text")
        if isinstance(req, str) and req.strip():
            return req.strip()
    return ""


def _default_layout_details(step2_output: dict[str, Any]) -> list[str]:
    """Return the reference-image phrasing (`image_text`) of every built-in
    default layout a skill injected because the user gave no visual requirement
    for that intent (e.g. the default bilingual layout). Empty when none."""
    out: list[str] = []
    vi = step2_output.get("visual_intent")
    if isinstance(vi, dict):
        for d in vi.get("default_details") or []:
            if isinstance(d, dict):
                txt = d.get("image_text")
                if isinstance(txt, str) and txt.strip():
                    out.append(txt.strip())
    return out


def _palette_hex(palette: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(palette, dict):
        p = palette.get("primary")
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
        for c in palette.get("fills") or []:
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
    return [x for x in out if not (x in seen or seen.add(x))]


def _build_tables_prompt_block(
    tables: list[dict[str, Any]] | None,
    texts: list[dict[str, Any]],
) -> str:
    """Describe declared `tables[]` for the image model so it draws them as
    aligned multi-column tables instead of scattering the raw text.

    Returns "" when there are no valid tables. Each table is summarised as its
    rows x cols plus, per column, the ids that populate it (a column is usually
    one text node repeated down its rows)."""
    if not isinstance(tables, list) or not tables:
        return ""
    lines: list[str] = []
    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        data = tbl.get("data")
        if not isinstance(data, list) or not data:
            continue
        col_desc = []
        for r, row in enumerate(data):
            if not isinstance(row, list): continue
            col_desc.append("row %d: %s" % (r, " | ".join(str(c.get("text") or "") for c in row if isinstance(c, dict))) )
        lines.append(
            f'- table "{tbl.get("id") or ""}": {len(data)} rows x {len(data[0]) if isinstance(data[0], list) else 0} columns; '
            + "; ".join(col_desc)
        )
    if not lines:
        return ""
    body = os.linesep.join(lines)
    return (
        "Table structure (authoritative native matrix): render each entry as ONE aligned table, preserving row/column order and merged spans. Do not split a table into loose columns."
        + os.linesep
        + body
    )


def build_image_prompt(
    *,
    page_size_pt: dict[str, Any],
    palette: dict[str, Any],
    user_request: str,
    texts: list[dict[str, Any]],
    images: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
    default_layout_details: list[str] | None = None,
    original_layout_description: str = "",
    has_original_page_reference: bool = True,
    deck_style: dict[str, Any] | None = None,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    creation_mode: bool = False,
) -> str:
    """Build the image-generation prompt.

    The image model decides the composition itself from the user's ORIGINAL
    intent plus the authoritative content. We no longer feed a pre-computed
    natural-language design brief; the image model's own design sense drives the
    layout, while the hard content constraints (verbatim text, place every image,
    palette as the default color scheme unless the user asked for a specific
    color) keep it faithful.
    """
    w = float(page_size_pt.get("w") or 0.0) if isinstance(page_size_pt, dict) else 0.0
    h = float(page_size_pt.get("h") or 0.0) if isinstance(page_size_pt, dict) else 0.0
    aspect = (w / h) if (w > 0 and h > 0) else (16 / 9)

    deck_style = deck_style if isinstance(deck_style, dict) else None
    if deck_style:
        colors = deck_style.get("colors") if isinstance(deck_style.get("colors"), dict) else {}
        top3 = [c for c in colors.get("top3") or [] if isinstance(c, str) and c.strip()]
        palette_hex = top3[:5] or _palette_hex(palette)
    else:
        palette_hex = _palette_hex(palette)
    palette_text = ", ".join(palette_hex) if palette_hex else "(none)"

    text_blocks: list[str] = []
    for t in texts:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "").strip()
        if not tid:
            continue
        content = str(t.get("text") or "")
        if len(content) > max_text_chars:
            content = content[: max_text_chars - 3] + "..."
        text_blocks.append(f'- [{tid}] kind={str(t.get("kind") or "")!r}: "{content}"')
    text_block = os.linesep.join(text_blocks) if text_blocks else "(no text)"

    img_blocks: list[str] = []
    for im in images or []:
        if not isinstance(im, dict):
            continue
        iid = str(im.get("id") or "").strip()
        if not iid:
            continue
        desc = _compact_text(im.get("description_en") or "", limit=DEFAULT_MAX_IMAGE_DESC_CHARS)
        dw = im.get("display_w_pt")
        dh = im.get("display_h_pt")
        if isinstance(dw, (int, float)) and isinstance(dh, (int, float)) and dw > 0 and dh > 0:
            size_hint = f" ({int(round(dw))}x{int(round(dh))} soft size)"
        else:
            size_hint = ""
        img_blocks.append(f'- [{iid}]{size_hint}: {desc}')
    img_block = os.linesep.join(img_blocks)

    user_request = str(user_request or "").strip() or "(no explicit instruction; redesign this slide to be clean, modern, and well-balanced.)"

    tables_block = _build_tables_prompt_block(tables, texts)
    tables_section = (os.linesep + os.linesep + tables_block) if tables_block else ""

    details = [d for d in (default_layout_details or []) if isinstance(d, str) and d.strip()]
    if details:
        defaults_section = (
            os.linesep + os.linesep
            + "Default layout hints (use unless the user says otherwise):" + os.linesep
            + os.linesep.join(
                f"- {_compact_text(d, limit=DEFAULT_MAX_DEFAULT_DETAIL_CHARS)}" for d in details
            )
        )
    else:
        defaults_section = ""

    orig_desc = _compact_text(original_layout_description, limit=DEFAULT_MAX_ORIGINAL_LAYOUT_CHARS)
    if orig_desc:
        original_layout_section = (
            os.linesep + os.linesep
            + "Original layout note (secondary): use for content relationships and rough composition only; keep it only when it helps the user request." + os.linesep
            + orig_desc
        )
    else:
        original_layout_section = ""
    secondary_reference_phrase = (
        "the original slide render, the original page description"
        if has_original_page_reference
        else "the content image references, the original page description"
    )

    deck_style_section = ""
    if deck_style:
        mood = deck_style.get("mood") if isinstance(deck_style.get("mood"), dict) else {}
        colors = deck_style.get("colors") if isinstance(deck_style.get("colors"), dict) else {}
        style_feel = _compact_text(mood.get("feel") or "", limit=180)
        overall_style = _compact_text(deck_style.get("overallStyle") or "", limit=DEFAULT_MAX_STYLE_CHARS)
        deck_style_section = f"""

Whole-deck style (use after the user's explicit visual request):
- Mood: {mood.get("name") or mood.get("id") or ""}{f" — {style_feel}" if style_feel else ""}
- Main colors: {", ".join(str(c) for c in (colors.get("top3") or [])[:5])}
- Layout feel: {overall_style}
""".rstrip()

    if creation_mode:
        reference_images_section = """
Reference images you receive:
- No input images are provided. This is intentional for a blank-page creation run.
- If content images are listed below, draw only neutral placeholders for their future positions; Step3 will place the real files.
""".strip()
    elif has_original_page_reference:
        reference_images_section = """
Reference images you receive:
- FIRST image: original slide render. Use it for content relationships, visual weight, and image aspect ratios; redesign when useful.
- ADDITIONAL images: real embedded photos/figures. Place each as-is, resized/cropped only to fit.
""".strip()
    elif images:
        reference_images_section = """
Reference images you receive:
- Input images are real embedded photos/figures only; place each as-is.
- No original slide render is provided; design from the user intent, content, and palette.
""".strip()
    else:
        reference_images_section = """
Reference images you receive:
- No input images are provided. Design from the user intent, authoritative text, and palette.
""".strip()

    content_images_section = (
        f"""

Content images to place (place every one; choose size/position):
{img_block}
""".rstrip()
        if img_block
        else ""
    )

    creation_mode_section = ""
    if creation_mode:
        creation_mode_section = """

Creation-mode image policy:
- This is a new page being created from a blank slide. No original slide screenshot is provided or useful.
- If content images are listed, reserve one neutral solid-color rectangle for each image according to its description, use, order, and aspect ratio.
- Do NOT simulate, redraw, invent, or fill in the real image content.
- Image placeholders must contain no text, logos, icons, pictograms, or decorative detail.
- Make placeholders visually plain so Step3 can replace them with the real image files.
""".rstrip()

    return f"""
You are generating a SINGLE presentation slide image to be used as a HIGH-FIDELITY layout reference.

Hard constraints:
- Aspect ratio must match the slide canvas (about {aspect:.4f}).
- Palette/default tone: {palette_text}. Stay within it unless the user names another color; honor requested colors and keep the rest cohesive. Do not add unrelated colors.
- Render the EXACT text content provided below (Chinese and/or English) with correct characters. Do NOT paraphrase, translate, reorder, or add/remove characters.
- Include EVERY text block and, if provided, EVERY content image; omit nothing.
- Do NOT add any extra text, labels, logos, watermarks, page numbers, UI chrome, or decorative paragraphs.
- Do NOT use any background photo; keep backgrounds as clean color fields / bands / cards only.
- Do NOT invent clipart, stock photos, fetched images, icons, glyphs, pictograms, or symbols. Only provided image files may appear as photos/icons; render them as-is. Plain bullet dots and small color accents are allowed.
- This image is a visual layout reference, not final artwork. It will be reconstructed as a single-slide HTML composition and then converted into editable presentation elements.
- Prefer designs expressible with standard text boxes, rectangles, rounded rectangles, straight lines, single-layer circles, and solid color blocks.
- Avoid decorative details that require custom paths, masks, precise multi-layer overlap, concentric micro-decoration, irregular cutouts/notches, complex nested geometry, or many tiny decorative parts. Make the slide beautiful through typography, whitespace, alignment, hierarchy, and palette usage.
- Lines must be straight and solid. Do NOT use arrowheads, dashed/dotted lines, vertical dotted stems, brackets, curved connectors, or multi-segment connector paths. For timelines and processes, use one plain solid axis with single-layer filled circles; separate stages through spacing, text hierarchy, and color.
- Do NOT construct compound graphic symbols from several touching micro-shapes. Every decorative object should read as one basic primitive. Keep labels comfortably legible instead of shrinking text to make room for decoration.

{reference_images_section}

Priority: editable/reconstructable layout and complete content > user's explicit visual request > whole-deck style when provided > old page look.

User request (top design intent; you choose the best composition). Other references ({secondary_reference_phrase}) are secondary:
{user_request}{deck_style_section}{defaults_section}{original_layout_section}
{content_images_section}{creation_mode_section}

Text content blocks (authoritative; render every one exactly, arrange them yourself into the best layout for the intent above):
{text_block}{tables_section}

Design a clean, legible, well-balanced slide with clear hierarchy, enough whitespace, one focal path, and correct palette use.
""".strip()


def _understanding_has_visible_content(understanding: dict[str, Any] | None) -> bool:
    if not isinstance(understanding, dict):
        return True
    for t in understanding.get("texts") or []:
        if isinstance(t, dict) and str(t.get("text") or "").strip():
            return True
    if any(isinstance(im, dict) for im in (understanding.get("images") or [])):
        return True
    if any(isinstance(tb, dict) for tb in (understanding.get("tables") or [])):
        return True
    return bool(str(understanding.get("original_layout_description_en") or "").strip())


def _collect_reference_images(
    um: dict[str, Any],
    *,
    max_content_images: int,
    include_page_render: bool,
) -> tuple[list[Path], list[str]]:
    """Return (image paths, notes). First path is the original page render;
    the rest are the real embedded content images (resolved via bundle_dir)."""
    paths: list[Path] = []
    notes: list[str] = []

    if include_page_render:
        page_png = str(um.get("page_png_path") or "").strip()
        if page_png:
            p = Path(page_png).expanduser()
            if p.exists():
                paths.append(p)
            else:
                notes.append("page_png_missing")
        else:
            notes.append("no_page_png_path")
    else:
        notes.append("omitted_blank_original_page_render")

    bundle_dir_s = str(um.get("bundle_dir") or "").strip()
    bundle_dir = Path(bundle_dir_s).expanduser() if bundle_dir_s else None
    if bundle_dir is not None:
        count = 0
        for im in um.get("images") or []:
            if not isinstance(im, dict):
                continue
            src = str(im.get("src") or "").strip()
            if not src:
                continue
            candidates = [(bundle_dir / src).resolve()]
            # A staged chat image may still be in the run-isolated upload
            # directory if promotion into ``source`` was interrupted.
            if bundle_dir.name == "source":
                candidates.append((bundle_dir.parent / "uploads" / src).resolve())
            ip = next((candidate for candidate in candidates if candidate.is_file()), None)
            if ip is not None:
                paths.append(ip)
                count += 1
            else:
                notes.append(f"content_image_missing[{im.get('id')}]")
            if count >= max_content_images:
                break

    return paths, notes


def _prepare_input_images(
    image_paths: list[Path],
    *,
    max_edge: int = DEFAULT_INPUT_MAX_EDGE,
    webp_quality: int = DEFAULT_INPUT_WEBP_QUALITY,
) -> list[tuple[str, bytes, str]]:
    """Load input images, optionally downscaling to `max_edge` and re-encoding as
    webp to shrink the upload body. Returns (filename, bytes, mime) tuples.

    On any failure (or when max_edge<=0, or Pillow is unavailable) we fall back to
    sending the original file bytes untouched, so this can only ever help, never
    break the call."""
    prepared: list[tuple[str, bytes, str]] = []
    for p in image_paths:
        raw = p.read_bytes()
        if max_edge <= 0:
            prepared.append((p.name, raw, _mime_for(p)))
            continue
        try:
            from PIL import Image  # local import: keep module import cheap/optional

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            w, h = img.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / longest
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=webp_quality, method=4)
            blob = buf.getvalue()
            # Only adopt the re-encode if it actually shrank the payload.
            if blob and len(blob) < len(raw):
                prepared.append((p.stem + ".webp", blob, "image/webp"))
            else:
                prepared.append((p.name, raw, _mime_for(p)))
        except Exception:
            prepared.append((p.name, raw, _mime_for(p)))
    return prepared


def _post_images_edits_once(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    data: dict[str, str],
    image_field: str,
    inputs: list[tuple[str, bytes, str]],
) -> dict[str, Any]:
    """Single attempt. `inputs` are already-materialized (name, bytes, mime)
    tuples so each retry sends a fresh, identical body."""
    files = [(image_field, (name, blob, mime)) for (name, blob, mime) in inputs]
    resp = client.post(url, headers=headers, data=data, files=files)
    resp.raise_for_status()
    payload = resp.json()
    _raise_image_api_payload_error(payload)
    return payload


def _post_images_generations_once(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    data: dict[str, str],
) -> dict[str, Any]:
    resp = client.post(url, headers=headers, json=data)
    resp.raise_for_status()
    payload = resp.json()
    _raise_image_api_payload_error(payload)
    return payload


def _raise_image_api_payload_error(payload: dict[str, Any]) -> None:
    err = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(err, dict):
        return
    message = err.get("message")
    err_type = err.get("type")
    detail = str(message).strip() if isinstance(message, str) and message.strip() else "unknown error"
    if isinstance(err_type, str) and err_type.strip():
        detail = f"{detail} (type={err_type.strip()})"
    raise RuntimeError(f"image API returned error payload: {detail}")


def _download_image_url(url: str, *, timeout_s: float = DEFAULT_IMAGE_URL_DOWNLOAD_TIMEOUT_S) -> bytes:
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        blob = resp.content
    if not blob:
        raise RuntimeError("image API response data[0].url downloaded empty content")
    return blob


def _decode_image_payload(payload: dict[str, Any]) -> bytes:
    _raise_image_api_payload_error(payload)
    arr = payload.get("data") if isinstance(payload, dict) else None
    data0 = arr[0] if isinstance(arr, list) and arr and isinstance(arr[0], dict) else None
    if not isinstance(data0, dict):
        raise RuntimeError("image API response missing data[0]")
    b64 = data0.get("b64_json")
    if isinstance(b64, str) and b64.strip():
        return base64.b64decode(b64)
    image_url = data0.get("url")
    if isinstance(image_url, str) and image_url.strip():
        return _download_image_url(image_url.strip())
    raise RuntimeError("image API response missing data[0].b64_json/url")


def _call_images_edits(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    size: str,
    quality: str,
    image_field: str,
    output_format: str | None = DEFAULT_IMAGE_OUTPUT_FORMAT,
    output_compression: int | None = DEFAULT_IMAGE_OUTPUT_COMPRESSION,
    input_max_edge: int = DEFAULT_INPUT_MAX_EDGE,
    input_webp_quality: int = DEFAULT_INPUT_WEBP_QUALITY,
    timeout_s: float = DEFAULT_IMAGE_TIMEOUT_S,
    max_retries: int = DEFAULT_IMAGE_MAX_RETRIES,
    retry_backoff_s: float = DEFAULT_IMAGE_RETRY_BACKOFF_S,
) -> bytes:
    url = base_url.rstrip("/") + "/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "prompt": prompt, "n": "1", "size": size, "quality": quality}
    if output_format:
        data["output_format"] = output_format
        # output_compression only applies to lossy webp/jpeg outputs.
        if output_compression is not None and output_format.lower() in ("webp", "jpeg", "jpg"):
            data["output_compression"] = str(output_compression)

    inputs = _prepare_input_images(
        image_paths, max_edge=input_max_edge, webp_quality=input_webp_quality
    )

    payload: dict[str, Any] | None = None
    last_err: Exception | None = None
    attempts = max(1, max_retries + 1)
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(attempts):
            try:
                payload = _post_images_edits_once(
                    client=client,
                    url=url,
                    headers=headers,
                    data=data,
                    image_field=image_field,
                    inputs=inputs,
                )
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                last_err = e
                if status not in _RETRYABLE_STATUS or attempt == attempts - 1:
                    raise
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = e
                if attempt == attempts - 1:
                    raise
            time.sleep(retry_backoff_s * (2 ** attempt))

    if payload is None:
        raise RuntimeError(f"image API call failed after {attempts} attempts: {last_err}")

    return _decode_image_payload(payload)


def _call_images_generations(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    output_format: str | None = DEFAULT_IMAGE_OUTPUT_FORMAT,
    output_compression: int | None = DEFAULT_IMAGE_OUTPUT_COMPRESSION,
    timeout_s: float = DEFAULT_IMAGE_TIMEOUT_S,
    max_retries: int = DEFAULT_IMAGE_MAX_RETRIES,
    retry_backoff_s: float = DEFAULT_IMAGE_RETRY_BACKOFF_S,
) -> bytes:
    url = base_url.rstrip("/") + "/images/generations"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "prompt": prompt, "n": 1, "size": size, "quality": quality}
    if output_format:
        data["output_format"] = output_format
        if output_compression is not None and output_format.lower() in ("webp", "jpeg", "jpg"):
            data["output_compression"] = str(output_compression)

    payload: dict[str, Any] | None = None
    last_err: Exception | None = None
    attempts = max(1, max_retries + 1)
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(attempts):
            try:
                payload = _post_images_generations_once(
                    client=client,
                    url=url,
                    headers=headers,
                    data=data,
                )
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                last_err = e
                if status not in _RETRYABLE_STATUS or attempt == attempts - 1:
                    raise
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = e
                if attempt == attempts - 1:
                    raise
            time.sleep(retry_backoff_s * (2 ** attempt))

    if payload is None:
        raise RuntimeError(f"image API call failed after {attempts} attempts: {last_err}")
    return _decode_image_payload(payload)


def generate_beautify_reference_image(
    *,
    step2_output: dict[str, Any],
    out_path: Path,
    original_understanding: dict[str, Any] | None = None,
    deck_style: dict[str, Any] | None = None,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str = DEFAULT_IMAGE_SIZE,
    quality: str = DEFAULT_IMAGE_QUALITY,
    image_field: str = DEFAULT_IMAGE_FIELD,
    output_format: str = DEFAULT_IMAGE_OUTPUT_FORMAT,
    output_compression: int | None = DEFAULT_IMAGE_OUTPUT_COMPRESSION,
    input_max_edge: int = DEFAULT_INPUT_MAX_EDGE,
    input_webp_quality: int = DEFAULT_INPUT_WEBP_QUALITY,
    max_content_images: int = DEFAULT_MAX_CONTENT_IMAGES,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    creation_mode: bool = False,
    force_generation: bool = False,
) -> dict[str, Any]:
    """Generate a beautify reference image from a Step 2 output and write it to
    `out_path`. Returns metadata (path, attached images, notes). Raises
    RuntimeError on backend/config/API failure so the caller can degrade."""
    um = _understand_modified(step2_output)
    if not um:
        raise RuntimeError("step2_output missing compile.understand_modified")

    # Prefer the planner's verbatim visual requirement (visual_intent.
    # requirements_text) over the raw user_request: it isolates the user's
    # visual/layout/style words from any content-editing instructions. Fall
    # back to the full user_request when no visual text was captured.
    user_request = _visual_requirements_text(step2_output) or str(step2_output.get("user_request") or "")
    page_size_pt = um.get("page_size_pt") if isinstance(um.get("page_size_pt"), dict) else {}
    palette = um.get("palette") if isinstance(um.get("palette"), dict) else {}
    if isinstance(deck_style, dict):
        colors = deck_style.get("colors") if isinstance(deck_style.get("colors"), dict) else {}
        top3 = [c for c in colors.get("top3") or [] if isinstance(c, str)]
        if top3:
            palette = {"primary": top3[0], "fills": top3[1:5]}
    texts = [t for t in (um.get("texts") or []) if isinstance(t, dict)]
    images = [im for im in (um.get("images") or []) if isinstance(im, dict)]
    tables = [tb for tb in (um.get("tables") or []) if isinstance(tb, dict)]
    original_layout_description = str(um.get("original_layout_description_en") or "")
    include_page_render = _understanding_has_visible_content(original_understanding)

    prompt = build_image_prompt(
        page_size_pt=page_size_pt,
        palette=palette,
        user_request=user_request,
        texts=texts,
        images=images,
        tables=tables,
        default_layout_details=_default_layout_details(step2_output),
        original_layout_description=original_layout_description,
        has_original_page_reference=include_page_render,
        deck_style=deck_style if isinstance(deck_style, dict) else None,
        max_text_chars=max_text_chars,
        creation_mode=bool(creation_mode),
    )

    image_paths, notes = _collect_reference_images(
        um,
        max_content_images=max_content_images,
        include_page_render=include_page_render,
    )

    base_url, api_key = _resolve_image_backend()
    if force_generation:
        image_paths = []

    if image_paths:
        generation_mode = "edit"
        img_bytes = _call_images_edits(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            size=size,
            quality=quality,
            image_field=image_field,
            output_format=output_format or None,
            output_compression=output_compression,
            input_max_edge=input_max_edge,
            input_webp_quality=input_webp_quality,
        )
    else:
        generation_mode = "generation"
        img_bytes = _call_images_generations(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format=output_format or None,
            output_compression=output_compression,
        )

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img_bytes)

    return {
        "reference_image_path": str(out_path),
        "attached_images": [str(p) for p in image_paths],
        "generation_mode": generation_mode,
        "creation_mode": bool(creation_mode),
        "force_generation": bool(force_generation),
        "prompt": prompt,
        "model": model,
        "size": size,
        "quality": quality,
        "output_format": output_format or "png",
        "input_max_edge": input_max_edge,
        "notes": notes,
    }
