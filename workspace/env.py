"""Minimal, dependency-free `.env` loader.

The pipeline reads all configuration straight from `os.environ` (LLM /
image backend URLs, keys, model names). To let users keep those in a
`agent_backend/.env` file instead of exporting them by hand,
we load that file once at process startup.

Kept intentionally tiny (no python-dotenv dependency): supports
`KEY=VALUE` lines, `#` comments, optional surrounding quotes, and
inline `# ...` comments on unquoted values. Existing environment
variables always win, so an explicit `export` still overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import PROJECT_ROOT


def _strip_inline_comment(value: str) -> str:
    # Only strip inline comments for unquoted values; quoted values are
    # returned verbatim by the caller before this is reached.
    out: list[str] = []
    for i, ch in enumerate(value):
        if ch == "#" and (i == 0 or value[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export "):
        raw = raw[len("export "):].strip()
    if "=" not in raw:
        return None
    key, _, value = raw.partition("=")
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        # Fully quoted: take verbatim (no inline-comment stripping).
        value = value[1:-1]
    elif value[:1] in ("'", '"'):
        # Quoted value with trailing content (e.g. an inline comment):
        # take everything up to the matching closing quote.
        quote = value[0]
        end = value.find(quote, 1)
        value = value[1:end] if end != -1 else value[1:]
    else:
        value = _strip_inline_comment(value)
    return key, value


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load `KEY=VALUE` pairs from a `.env` file into `os.environ`.

    Defaults to `agent_backend/.env`. Returns the mapping that
    was applied. Existing env vars are preserved unless `override=True`.
    Missing file is a no-op.
    """
    env_path = Path(path).expanduser() if path else (PROJECT_ROOT / ".env")
    if not env_path.is_file():
        return {}

    applied: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
