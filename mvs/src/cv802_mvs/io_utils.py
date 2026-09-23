"""Small, dependency-free integrity and atomic-I/O helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file and atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inventory_files(root: Path, *, hashes: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        record: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": stat.st_size,
        }
        if hashes:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def inventory_summary(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    items = list(records)
    return {
        "file_count": len(items),
        "total_bytes": sum(int(item["size_bytes"]) for item in items),
    }

