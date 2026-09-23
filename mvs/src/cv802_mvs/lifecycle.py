"""Recoverable experiment lifecycle operations (never recursive deletion)."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

from .errors import StageConflictError


GENERATED_NAMES = (
    "inputs",
    "work",
    "outputs",
    "receipts",
    "logs",
    "manifests",
    "runtime",
    "run_state.json",
)


def archive_existing(root: Path, names: tuple[str, ...] = GENERATED_NAMES, *, reason: str) -> Path | None:
    """Atomically move known experiment artifacts to a sibling archive folder."""
    existing = [root / name for name in names if (root / name).exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "archive" / f"{reason}-{stamp}-{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=True, exist_ok=False)
    for source in existing:
        os.replace(source, archive / source.name)
    return archive


def assert_clean_run_area(root: Path) -> None:
    conflicts = [
        root / name
        for name in ("work", "outputs", "receipts", "logs", "runtime", "run_state.json")
        if (root / name).exists()
    ]
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise StageConflictError(
            f"Run artifacts already exist ({rendered}); use --resume or --overwrite"
        )


def archive_run_artifacts(root: Path) -> Path | None:
    """Archive a prior run while preserving independently staged inputs."""
    run_names = ("work", "outputs", "receipts", "logs", "runtime", "run_state.json")
    archive = archive_existing(root, run_names, reason="overwrite-run")
    generated_manifests = (
        "run_manifest.json",
        "input_validation.json",
        "runtime_validation.json",
        "masked_images.json",
        "final_validation.json",
    )
    existing_manifests = [
        root / "manifests" / name
        for name in generated_manifests
        if (root / "manifests" / name).exists()
    ]
    if existing_manifests:
        if archive is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive = root / "archive" / f"overwrite-run-{stamp}-{uuid.uuid4().hex[:8]}"
            archive.mkdir(parents=True, exist_ok=False)
        destination = archive / "manifests"
        destination.mkdir(parents=True, exist_ok=True)
        for source in existing_manifests:
            os.replace(source, destination / source.name)
    return archive
