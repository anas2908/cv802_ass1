#!/usr/bin/env python3
"""Verify the method-local all-290 dark input and persist its copy lineage.

This is a read-only audit except for its small JSON receipt. It verifies both
source and destination bytes against the already frozen MVS copy manifest; the
VGGSfM inference itself depends only on its method-local destination.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256
from vggsfm_engine.paths import production_layout, require_within


SOURCE_ROOT = Path(
    "/l/users/anas.khan/cv_802_ass1/mvs/experiments/"
    "dark_e3_colmap_mvs_1024_raw_v1/inputs/images"
)
SOURCE_MANIFEST = SOURCE_ROOT.parent.parent / "manifests" / "input_provenance.json"
DATASET = "dark_shirt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("dark input audit must run inside the active Slurm allocation")
    layout = production_layout()
    destination = layout.assert_member(
        layout.dataset_root(DATASET), label="dark VGGSfM input", must_exist=True
    )
    source_root = SOURCE_ROOT.resolve(strict=True)
    source_manifest = SOURCE_MANIFEST.resolve(strict=True)
    receipt_path = destination / "input_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    provenance = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (
        receipt.get("dataset") != DATASET
        or receipt.get("selection") != "all_images"
        or receipt.get("image_count") != 290
        or receipt.get("original_image_count") != 290
        or receipt.get("mask_count") != 0
    ):
        raise RuntimeError("VGGSfM input receipt does not prove raw all-290 selection")

    copied = {}
    for row in receipt.get("files", []):
        relative = Path(row["path"])
        if not relative.parts or relative.parts[0] != "images":
            raise RuntimeError(f"Unexpected VGGSfM input role/path: {relative}")
        copied[Path(*relative.parts[1:]).as_posix()] = (row["bytes"], row["sha256"])
    upstream = {}
    for row in provenance.get("files", []):
        if row.get("role") != "registered_image":
            continue
        prefix = Path("inputs/images")
        relative = Path(row["destination_relative"])
        try:
            key = relative.relative_to(prefix).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Unexpected source manifest path: {relative}") from exc
        upstream[key] = (row["size_bytes"], row["sha256"])
    if copied != upstream or len(copied) != 290:
        raise RuntimeError("VGGSfM receipt and source manifest records differ")

    rows = []
    dimensions: Counter[str] = Counter()
    total_bytes = 0
    for relative, (expected_bytes, expected_hash) in sorted(copied.items()):
        source = require_within(source_root / relative, source_root,
                                label="dark source image", must_exist=True)
        target = require_within(destination / "images" / relative, destination,
                                label="dark copied image", must_exist=True)
        if source.is_symlink() or target.is_symlink():
            raise RuntimeError("Dark input contains a symlink")
        source_stat_before, target_stat_before = source.stat(), target.stat()
        source_hash, target_hash = sha256(source), sha256(target)
        source_stat_after, target_stat_after = source.stat(), target.stat()
        if (
            source_stat_before.st_size != source_stat_after.st_size
            or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
            or target_stat_before.st_size != target_stat_after.st_size
            or target_stat_before.st_mtime_ns != target_stat_after.st_mtime_ns
        ):
            raise RuntimeError(f"Image changed during hashing: {relative}")
        if (
            source_stat_after.st_size != expected_bytes
            or target_stat_after.st_size != expected_bytes
            or source_hash != expected_hash
            or target_hash != expected_hash
        ):
            raise RuntimeError(f"Size/SHA mismatch: {relative}")
        with Image.open(target) as image:
            width, height = image.size
        dimensions[f"{width}x{height}"] += 1
        total_bytes += expected_bytes
        rows.append({
            "relative_path": relative,
            "bytes": expected_bytes,
            "sha256": expected_hash,
            "source_path": str(source),
            "destination_path": str(target),
            "width": width,
            "height": height,
        })
    if dimensions != Counter({"3024x4032": 183, "1080x1920": 107}):
        raise RuntimeError(f"Unexpected dark image dimensions: {dict(dimensions)}")

    report = {
        "schema_version": 1,
        "status": "complete",
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "selection": "all_images",
        "image_count": 290,
        "mask_count": 0,
        "total_bytes": total_bytes,
        "dimensions": dict(sorted(dimensions.items())),
        "source_root": str(source_root),
        "destination_root": str(destination / "images"),
        "source_manifest": file_record(source_manifest),
        "copy_receipt": file_record(receipt_path),
        "record_sets_equal": True,
        "source_and_destination_sha256_verified": True,
        "source_and_destination_regular_files": True,
        "no_symlinks": True,
        "raw_unmasked_to_retain_crutches": True,
        "rows_digest_sha256": object_sha256(rows),
        "rows": rows,
        "runtime_independence": (
            "Inference reads only the copied VGGSfM destination; the MVS path is provenance, "
            "not a runtime dependency."
        ),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "audit_source": file_record(Path(__file__).resolve()),
    }
    atomic_write_json(destination / "source_provenance_audit.json", report)
    return report


def main() -> int:
    result = audit()
    print(json.dumps({key: result[key] for key in (
        "status", "dataset", "image_count", "total_bytes", "dimensions",
        "rows_digest_sha256", "record_sets_equal", "no_symlinks",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
