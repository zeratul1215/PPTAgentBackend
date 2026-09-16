"""Deck-level visual style analysis, storage, and agent gate."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from agent_backend.agent.models import build_chat_model
from agent_backend.agent.tools.context import (
    emit,
    require_project_id,
    require_user_id,
    workspace_for,
)
from agent_backend.workspace import pageorder, repo
from agent_backend.workspace.db import db_enabled
from agent_backend.workspace.paths import WorkspacePaths, read_json


MOOD_CATALOG: dict[str, dict[str, str]] = {
    "professional_rational": {
        "name": "专业理性",
        "feel": "清晰、稳健、可信，强调信息层级、对齐和克制的商务感。",
    },
    "warm_approachable": {
        "name": "温和亲近",
        "feel": "柔和、友好、易读，用较轻的对比和舒适留白降低距离感。",
    },
    "energetic_creative": {
        "name": "活力创意",
        "feel": "明快、有节奏、视觉更有动势，但仍保持页面结构清楚。",
    },
    "premium_restrained": {
        "name": "高级克制",
        "feel": "简洁、精致、低噪声，依靠留白、比例和少量强调色形成品质感。",
    },
    "tech_futuristic": {
        "name": "科技未来",
        "feel": "冷静、现代、数字化，常用清晰网格、线性元素和高对比强调。",
    },
}

VALID_SOURCES = {"ai_detected", "user_preset", "user_edited"}
BUILTIN_FONT_VALUES = {
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
SYSTEM_FONT_VALUES = {
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
PPTIST_FONT_VALUES = BUILTIN_FONT_VALUES | SYSTEM_FONT_VALUES
FONT_ALIASES = {
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


class DeckStyleGateCancelled(RuntimeError):
    """Raised internally when the user cancels a Deck Style gate."""


def normalize_color(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"transparent", "none", "inherit"}:
        return None
    if re.fullmatch(r"#[0-9A-Fa-f]{3}", text):
        return "#" + "".join(ch * 2 for ch in text[1:]).upper()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return text.upper()
    m = re.fullmatch(r"rgba?\(([^)]+)\)", text, re.I)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 3:
            try:
                if len(parts) >= 4 and float(parts[3]) <= 0:
                    return None
                rgb = [max(0, min(255, int(float(p)))) for p in parts[:3]]
                return "#{:02X}{:02X}{:02X}".format(*rgb)
            except Exception:
                return None
    return None


def normalize_font_family(value: Any) -> str | None:
    text = str(value or "").strip().strip("'\"")
    if not text:
        return None
    first = text.split(",")[0].strip().strip("'\"")
    if first in PPTIST_FONT_VALUES:
        return first
    lowered = re.sub(r"\s+", " ", first).lower()
    alias = FONT_ALIASES.get(lowered)
    if alias:
        return alias
    if re.fullmatch(r"[\w\s\-\u4e00-\u9fff.()]+", first, re.U) and len(first) <= 80:
        return first
    return None


def validate_style(raw: dict[str, Any], *, source: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("style_json must be an object")
    mood_raw = raw.get("mood") if isinstance(raw.get("mood"), dict) else {}
    mood_id = str(raw.get("moodId") or mood_raw.get("id") or "").strip()
    if mood_id not in MOOD_CATALOG:
        raise ValueError(f"invalid moodId: {mood_id or '(empty)'}")
    mood = {"id": mood_id, **MOOD_CATALOG[mood_id]}
    src = str(source or raw.get("source") or "user_edited").strip()
    if src not in VALID_SOURCES:
        src = "user_edited"
    colors_raw = raw.get("colors") if isinstance(raw.get("colors"), dict) else {}
    top3: list[str] = []
    for c in colors_raw.get("top3") or raw.get("top3") or []:
        norm = normalize_color(c)
        if norm and norm not in top3:
            top3.append(norm)
        if len(top3) >= 5:
            break
    typography_raw = raw.get("typography") if isinstance(raw.get("typography"), dict) else {}

    def _font(name: str) -> dict[str, Any]:
        obj = typography_raw.get(name) if isinstance(typography_raw.get(name), dict) else {}
        return {"fontFamily": normalize_font_family(obj.get("fontFamily"))}

    return {
        "schemaVersion": "deck_style_v1",
        "source": src,
        "mood": mood,
        "colors": {"top3": top3},
        "typography": {"title": _font("title"), "body": _font("body")},
        "overallStyle": str(raw.get("overallStyle") or "").strip(),
    }


def public_style_row(project_id: str) -> dict[str, Any]:
    row = repo.get_deck_style(project_id) if db_enabled() else None
    if not row:
        return {
            "project_id": project_id,
            "status": "failed",
            "style_json": None,
            "source": None,
            "revision": 0,
            "error": "deck style has not been initialized",
        }
    return {
        "project_id": project_id,
        "status": row.get("status"),
        "style_json": row.get("style_json"),
        "source": row.get("source"),
        "revision": int(row.get("revision") or 0),
        "analysis_run_id": row.get("analysis_run_id"),
        "error": row.get("error"),
    }


def save_user_style(project_id: str, style_json: dict[str, Any], *, source: str) -> dict[str, Any]:
    style = validate_style(style_json, source=source)
    row = repo.upsert_deck_style(
        project_id=project_id,
        status="ready",
        style_json=style,
        source=style["source"],
        analysis_run_id=None,
        error=None,
        bump_revision=True,
    )
    if row is None:
        raise RuntimeError("deck style requires database")
    emit(project_id, {"type": "deck_style_status", **public_style_row(project_id)})
    return public_style_row(project_id)


def require_ready_style(project_id: str, *, interrupt_when_unready: bool = True) -> dict[str, Any]:
    row = public_style_row(project_id)
    if row.get("status") == "ready" and isinstance(row.get("style_json"), dict):
        return row
    if not interrupt_when_unready:
        raise RuntimeError(f"deck style is not ready: {row.get('status')}")
    value = {
        "kind": "deck_style_gate",
        "project_id": project_id,
        "status": row.get("status"),
        "revision": row.get("revision"),
        "error": row.get("error"),
        "interrupt_id": f"style_gate_{uuid.uuid4().hex[:10]}",
    }
    resume = interrupt(value)
    if isinstance(resume, dict) and resume.get("cancel"):
        raise DeckStyleGateCancelled("deck style task was cancelled by user")
    row = public_style_row(project_id)
    if row.get("status") == "ready" and isinstance(row.get("style_json"), dict):
        return row
    raise RuntimeError(f"deck style is still not ready: {row.get('status')}")


@tool
def get_deck_style(runtime: ToolRuntime, require_ready: bool = False) -> dict[str, Any]:
    """Read the active deck's visual style.

    With `require_ready=false`, this returns the current status and any available
    style. With `require_ready=true`, it returns a ready authoritative style or
    enters the existing style gate. It describes whole-deck style, not page-local
    appearance.
    """
    pid = require_project_id(runtime)
    if require_ready:
        try:
            row = require_ready_style(pid, interrupt_when_unready=True)
        except DeckStyleGateCancelled:
            return {
                "project_id": pid,
                "ok": False,
                "cancelled": True,
                "status": "cancelled",
                "revision": None,
                "style": None,
                "error": None,
            }
    else:
        row = public_style_row(pid)
    return {
        "project_id": pid,
        "ok": True,
        "status": row.get("status"),
        "revision": row.get("revision"),
        "style": row.get("style_json") if row.get("status") == "ready" else None,
        "error": row.get("error"),
    }


def _style_weight_add(weights: dict[str, float], color: Any, weight: float) -> None:
    norm = normalize_color(color)
    if norm and weight > 0:
        weights[norm] += float(weight)


def _style_dict(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(style or "").split(";"):
        k, sep, v = part.partition(":")
        if sep and k.strip() and v.strip():
            out[k.strip().lower()] = v.strip()
    return out


def _iter_html_runs(content: str):
    try:
        from lxml import html

        root = html.fragment_fromstring(f"<div>{content or ''}</div>", create_parent=False)
        for node in root.iter():
            if not isinstance(node.tag, str):
                continue
            text = " ".join(node.text_content().split())
            if not text:
                continue
            style = _style_dict(node.get("style") or "")
            yield text, style
    except Exception:
        return


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("px", "").replace("pt", "").strip())
    except Exception:
        return default


def _facts_for_slide(slide: dict[str, Any]) -> dict[str, Any]:
    color_weights: dict[str, float] = defaultdict(float)
    title_fonts: dict[str, float] = defaultdict(float)
    body_fonts: dict[str, float] = defaultdict(float)
    bg = slide.get("background") if isinstance(slide.get("background"), dict) else {}
    _style_weight_add(color_weights, bg.get("color"), 100000)
    runs: list[tuple[str, str, float]] = []
    for el in slide.get("elements") or []:
        if not isinstance(el, dict):
            continue
        area = max(1.0, _float(el.get("width")) * _float(el.get("height")))
        etype = el.get("type")
        if etype == "shape":
            _style_weight_add(color_weights, el.get("fill"), area)
            text = el.get("text") if isinstance(el.get("text"), dict) else {}
            content = text.get("content") if isinstance(text, dict) else None
        elif etype == "text":
            content = el.get("content")
        elif etype == "line":
            _style_weight_add(color_weights, el.get("color") or el.get("style"), max(_float(el.get("width")), _float(el.get("height")), 1.0))
            content = None
        else:
            content = None
        if isinstance(content, str):
            for text, style in _iter_html_runs(content):
                size = _float(style.get("font-size"), 16.0)
                weight = max(1, len(text)) * max(1.0, size)
                _style_weight_add(color_weights, style.get("color"), weight)
                font = style.get("font-family") or el.get("defaultFontName")
                if isinstance(font, str) and font.strip():
                    runs.append((font.strip().strip("'\""), text, size))
    if runs:
        max_size = max(size for _, _, size in runs)
        for font, text, size in runs:
            target = title_fonts if size >= max_size * 0.9 else body_fonts
            target[font] += max(1, len(text)) * max(1.0, size)
    return {
        "colors": dict(color_weights),
        "title_fonts": dict(title_fonts),
        "body_fonts": dict(body_fonts),
    }


def _merge_facts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    colors: dict[str, float] = defaultdict(float)
    title_fonts: dict[str, float] = defaultdict(float)
    body_fonts: dict[str, float] = defaultdict(float)
    for s in samples:
        for k, v in (s.get("colors") or {}).items():
            colors[k] += float(v or 0)
        for k, v in (s.get("title_fonts") or {}).items():
            title_fonts[k] += float(v or 0)
        for k, v in (s.get("body_fonts") or {}).items():
            body_fonts[k] += float(v or 0)
    top3 = [c for c, _ in sorted(colors.items(), key=lambda kv: kv[1], reverse=True)[:5]]

    def top_font(values: dict[str, float]) -> str | None:
        return max(values.items(), key=lambda kv: kv[1])[0] if values else None

    return {
        "top3": top3,
        "titleFont": top_font(title_fonts),
        "bodyFont": top_font(body_fonts),
    }


def _data_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


STYLE_ANALYSIS_PROMPT = """You analyze the shared visual style of a PPT deck.

You receive up to five rendered slide screenshots and deterministic style facts
extracted from PPTist JSON. Return STRICT JSON only:
{
  "moodId": "professional_rational|warm_approachable|energetic_creative|premium_restrained|tech_futuristic",
  "overallStyle": ""
}

Mood catalog:
- professional_rational: clear, steady, trustworthy, business-like.
- warm_approachable: soft, friendly, comfortable, easy to read.
- energetic_creative: bright, rhythmic, expressive, dynamic but clear.
- premium_restrained: refined, quiet, minimal, high-quality through spacing and proportion.
- tech_futuristic: modern, digital, grid/line oriented, high contrast accents.

Rules:
- Choose exactly one moodId from the catalog.
- Do not output colors as palette; the five deterministic colors will be merged by code.
- Do not describe fonts; deterministic title/body fonts will be merged by code.
- Describe reusable visual tendencies, not page-specific layout.
- overallStyle must be concise: max 80 English words or 160 Chinese characters.
- Focus on overall layout feel: hierarchy, spacing, alignment, background treatment, simple shape/card tendencies, and decoration level when visible.
- Mention if the deck is inconsistent, but still infer a practical unifying style.
"""


def _call_style_model(*, facts: dict[str, Any], pngs: list[Path]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps({"styleFacts": facts}, ensure_ascii=False)}
    ]
    for p in pngs:
        content.append({"type": "image_url", "image_url": {"url": _data_url(p)}})
    response = build_chat_model().invoke([SystemMessage(content=STYLE_ANALYSIS_PROMPT), HumanMessage(content=content)])
    raw = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if m:
        raw = m.group(1)
    return json.loads(raw.strip())


def _collect_samples(paths: WorkspacePaths) -> tuple[list[Path], dict[str, Any]]:
    pngs: list[Path] = []
    slide_facts: list[dict[str, Any]] = []
    for entry in pageorder.ordered_entries(paths)[:5]:
        slot = int(entry["slot"])
        png = paths.page_png(slot)
        if not png.exists():
            png = paths.reread_page_png(slot)
        if png.exists():
            pngs.append(png)
        slide_path = paths.pptist_slide_json(slot)
        if slide_path.exists():
            try:
                slide = read_json(slide_path)
                if isinstance(slide, dict):
                    slide_facts.append(_facts_for_slide(slide))
            except Exception:
                pass
    if not pngs:
        raise RuntimeError("no valid screenshots for deck style analysis")
    facts = _merge_facts(slide_facts)
    return pngs, facts


def _analysis_worker(project_id: str, analysis_run_id: str) -> None:
    paths = workspace_for(project_id)
    try:
        pngs, facts = _collect_samples(paths)
        model_obj = _call_style_model(facts=facts, pngs=pngs)
        style = validate_style(
            {
                "source": "ai_detected",
                "moodId": model_obj.get("moodId"),
                "colors": {"top3": facts.get("top3") or []},
                "typography": {
                    "title": {"fontFamily": facts.get("titleFont")},
                    "body": {"fontFamily": facts.get("bodyFont")},
                },
                "overallStyle": model_obj.get("overallStyle") or "",
            },
            source="ai_detected",
        )
        row = repo.upsert_deck_style(
            project_id=project_id,
            status="ready",
            style_json=style,
            source="ai_detected",
            analysis_run_id=analysis_run_id,
            error=None,
            bump_revision=True,
            expected_analysis_run_id=analysis_run_id,
            expected_status="analyzing",
        )
        if row:
            emit(project_id, {"type": "deck_style_status", **public_style_row(project_id)})
    except Exception as exc:  # noqa: BLE001
        row = repo.upsert_deck_style(
            project_id=project_id,
            status="failed",
            style_json=None,
            source=None,
            analysis_run_id=None,
            error=f"{type(exc).__name__}: {exc}",
            bump_revision=False,
            expected_analysis_run_id=analysis_run_id,
            expected_status="analyzing",
        )
        if row:
            emit(project_id, {"type": "deck_style_status", **public_style_row(project_id)})


def start_style_analysis(project_id: str) -> dict[str, Any]:
    if not db_enabled():
        raise RuntimeError("deck style requires database")
    run_id = f"style_{uuid.uuid4().hex[:12]}"
    row = repo.upsert_deck_style(
        project_id=project_id,
        status="analyzing",
        style_json=None,
        source=None,
        analysis_run_id=run_id,
        error=None,
        bump_revision=True,
    )
    if row is None:
        raise RuntimeError("deck style requires database")
    emit(project_id, {"type": "deck_style_status", **public_style_row(project_id)})
    threading.Thread(
        target=_analysis_worker,
        args=(project_id, run_id),
        daemon=True,
        name=f"deck-style-{project_id[:16]}",
    ).start()
    return public_style_row(project_id)


def bootstrap_style_analysis_async(project_id: str) -> None:
    def _run() -> None:
        try:
            start_style_analysis(project_id)
        except Exception as exc:  # noqa: BLE001
            try:
                repo.upsert_deck_style(
                    project_id=project_id,
                    status="failed",
                    style_json=None,
                    source=None,
                    analysis_run_id=None,
                    error=f"{type(exc).__name__}: {exc}",
                    bump_revision=True,
                )
                emit(project_id, {"type": "deck_style_status", **public_style_row(project_id)})
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True, name=f"deck-style-start-{project_id[:16]}").start()


__all__ = [
    "DeckStyleGateCancelled",
    "MOOD_CATALOG",
    "validate_style",
    "public_style_row",
    "save_user_style",
    "require_ready_style",
    "start_style_analysis",
    "bootstrap_style_analysis_async",
    "get_deck_style",
]
