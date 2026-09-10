"""Convert Office presentations to PDF via LibreOffice (headless).

Users upload .ppt/.pptx. The generated PDF is a temporary rasterization source
for baseline PNGs only; PDF upload is not a supported product path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class ConversionError(RuntimeError):
    pass


_PRESENTATION_SUFFIXES = {".ppt", ".pptx"}


def _find_soffice() -> str | None:
    p = shutil.which("soffice")
    if p:
        return p
    mac = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if Path(mac).exists():
        return mac
    return None


def is_supported_source(path: Path) -> bool:
    return path.suffix.lower() in _PRESENTATION_SUFFIXES


def convert_to_pdf(*, src: Path, outdir: Path) -> Path:
    """Convert a .ppt/.pptx file to PDF, returning the temporary PDF path."""
    if not is_supported_source(src):
        raise ConversionError(f"unsupported source format: {src.suffix}")

    soffice = _find_soffice()
    if not soffice:
        raise ConversionError(
            "LibreOffice `soffice` not found. Install LibreOffice (macOS: "
            "LibreOffice.app) or put `soffice` on PATH."
        )

    src = src.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"input not found: {src}")

    outdir.mkdir(parents=True, exist_ok=True)

    # Isolate the LO user profile to dodge first-run prompts / cross-process locks.
    profile_dir = Path(tempfile.mkdtemp(prefix="lo_profile_", dir=str(outdir)))
    profile_uri = profile_dir.as_uri()

    try:
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--norestore",
            f"-env:UserInstallation={profile_uri}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(src),
        ]
        env = dict(os.environ)
        env.setdefault("HOME", str(outdir))

        print("\n$ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ConversionError(
                f"soffice exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

        expected = outdir / f"{src.stem}.pdf"
        if not expected.exists():
            # LO occasionally normalizes casing / naming; find the best match.
            candidates = sorted(
                outdir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            for c in candidates:
                if c.stem == src.stem or c.name.lower() == f"{src.stem.lower()}.pdf":
                    expected = c
                    break
        if not expected.exists():
            raise ConversionError(
                f"conversion finished but no PDF was produced in {outdir} for {src}"
            )
        return expected
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


__all__ = [
    "ConversionError",
    "is_supported_source",
    "convert_to_pdf",
]
