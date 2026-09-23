"""Small deterministic I/O helpers; all callers pass data-root destinations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigurationError

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def validate_identifier(value: str, *, label: str) -> str:
    if not SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ConfigurationError(
            f"{label} must match {SAFE_ID.pattern!r}; received {value!r}"
        )
    return value


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def object_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    display = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    return {
        "path": display,
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON durably using a temporary file beside the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read JSON {path}: {exc}") from exc


def manifest_for_files(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    return [file_record(path, relative_to=relative_to) for path in sorted(paths)]
