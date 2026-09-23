"""Storage-safe and atomic file utilities for evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
EVALUATION_ROOT = DATA_ROOT / "evaluation"
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class StorageError(RuntimeError):
    """A requested path violates the permanent storage contract."""


def require_within(
    path: Path, root: Path, *, label: str, must_exist: bool = True
) -> Path:
    try:
        resolved_root = root.resolve(strict=must_exist)
        resolved = path.resolve(strict=must_exist)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise StorageError(f"{label} must stay below {root}: {path}") from error
    return resolved


def evaluation_directory(identifier: str) -> Path:
    if not IDENTIFIER.fullmatch(identifier):
        raise StorageError(
            "evaluation id must be 1-80 letters, digits, dot, underscore or hyphen"
        )
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    require_within(EVALUATION_ROOT, DATA_ROOT, label="evaluation root")
    destination = EVALUATION_ROOT / identifier
    require_within(destination, EVALUATION_ROOT, label="evaluation directory", must_exist=False)
    destination.mkdir(parents=False, exist_ok=True)
    return require_within(destination, EVALUATION_ROOT, label="evaluation directory")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path = DATA_ROOT) -> dict[str, Any]:
    resolved = require_within(path, root, label="input artifact")
    if not resolved.is_file():
        raise StorageError(f"input artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    parent = require_within(path.parent, EVALUATION_ROOT, label="report directory")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()

