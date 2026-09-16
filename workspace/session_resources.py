"""Filesystem side of the session-scoped multimodal resource store.

The database owns resource metadata; this module owns immutable payload files.
No user supplied path is ever used directly.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from agent_backend.workspace.paths import session_resource_dir, session_resources_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_resource_dir(user_id: str, session_id: str, resource_ref: str) -> Path:
    """Create an empty resource directory atomically and return its path."""
    parent = session_resources_root() / str(user_id) / str(session_id)
    parent.mkdir(parents=True, exist_ok=True)
    target = session_resource_dir(user_id, session_id, resource_ref)
    if target.exists():
        raise FileExistsError(str(target))
    temp = Path(tempfile.mkdtemp(prefix=f".{resource_ref}.", dir=str(parent)))
    try:
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def copy_into_resource(
    *,
    user_id: str,
    session_id: str,
    resource_ref: str,
    source: Path,
    filename: str,
) -> tuple[Path, str, int]:
    """Copy one immutable payload into a newly-created resource directory."""
    if not source.is_file():
        raise FileNotFoundError(str(source))
    root = atomic_resource_dir(user_id, session_id, resource_ref)
    safe_name = Path(filename).name or "original.bin"
    dst = root / safe_name
    try:
        shutil.copy2(source, dst)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return dst, sha256_file(dst), int(dst.stat().st_size)


def write_resource_files(
    *,
    user_id: str,
    session_id: str,
    resource_ref: str,
    files: Iterable[tuple[str, bytes]],
) -> dict[str, tuple[Path, str, int]]:
    """Atomically write multiple resource payloads such as structure+preview."""
    root = atomic_resource_dir(user_id, session_id, resource_ref)
    out: dict[str, tuple[Path, str, int]] = {}
    try:
        for name, data in files:
            safe_name = Path(name).name
            if not safe_name:
                raise ValueError("resource filename is empty")
            dst = root / safe_name
            dst.write_bytes(data)
            out[safe_name] = (dst, sha256_file(dst), len(data))
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return out


def resource_path(user_id: str, session_id: str, resource_ref: str, relative_path: str) -> Path:
    root = session_resource_dir(user_id, session_id, resource_ref).resolve()
    path = (root / Path(relative_path).name).resolve()
    if root not in path.parents:
        raise ValueError("resource path escapes resource directory")
    return path


def delete_session_resource_dir(user_id: str, session_id: str, resource_ref: str) -> None:
    shutil.rmtree(session_resource_dir(user_id, session_id, resource_ref), ignore_errors=True)


def delete_session_resource_root(user_id: str, session_id: str) -> None:
    shutil.rmtree(session_resources_root() / str(user_id) / str(session_id), ignore_errors=True)


__all__ = [
    "atomic_resource_dir",
    "copy_into_resource",
    "delete_session_resource_dir",
    "delete_session_resource_root",
    "resource_path",
    "sha256_file",
    "write_resource_files",
]
