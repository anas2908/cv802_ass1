#!/usr/bin/env python3
"""Export the tracked VGGSfM core as a read-only-derived review PLY.

This is not another reconstruction and not a replacement for the published
full cloud. It selects exactly the existing COLMAP points whose track length
is positive and preserves their float32 XYZ/RGB PLY records.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine import colmap
from vggsfm_engine.errors import OutputValidationError
from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256, read_json
from vggsfm_engine.paths import production_layout


SOURCE_RUN_ID = "light-shirt-vggsfm-v1"
DERIVED_RUN_ID = "light-shirt-vggsfm-v1-tracked-only-review-v1"
EXPECTED_TRACKED_POINTS = 11_362
EXPECTED_TOTAL_POINTS = 414_208


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def absolute_record(path: Path, *, declared: Path | None = None) -> dict[str, object]:
    path = path.resolve(strict=True)
    before = path.stat()
    record = file_record(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise OutputValidationError(f"File changed while hashing: {path}")
    record["path"] = str((declared or path).resolve(strict=False))
    return record


def point_record(point: colmap.Point3D) -> bytes:
    if point.track_length <= 0:
        raise ValueError("tracked-only PLY cannot contain a trackless point")
    return struct.pack(
        "<fffBBB", point.x, point.y, point.z, point.red, point.green, point.blue
    )


def write_tracked_ply(points: Iterable[colmap.Point3D], destination: Path) -> dict[str, object]:
    selected = [point for point in points if point.track_length > 0]
    if not selected:
        raise OutputValidationError("Source model has no tracked points")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment derived review: existing VGGSfM points with track_length > 0\n"
        f"element vertex {len(selected)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bounds_min = [math.inf, math.inf, math.inf]
    bounds_max = [-math.inf, -math.inf, -math.inf]
    ids: list[int] = []
    expected_records: list[bytes] = []
    with destination.open("wb") as stream:
        stream.write(header)
        for point in selected:
            record = point_record(point)
            stream.write(record)
            expected_records.append(record)
            ids.append(point.point_id)
            for axis, value in enumerate((point.x, point.y, point.z)):
                bounds_min[axis] = min(bounds_min[axis], value)
                bounds_max[axis] = max(bounds_max[axis], value)

    payload = destination.read_bytes()
    marker = b"end_header\n"
    offset = payload.find(marker)
    if offset < 0:
        raise OutputValidationError("Written PLY has no header terminator")
    body = payload[offset + len(marker) :]
    if len(body) != 15 * len(selected):
        raise OutputValidationError("Written PLY body size does not match selected points")
    if body != b"".join(expected_records):
        raise OutputValidationError("Written PLY XYZ/RGB records changed after serialization")
    return {
        "point_count": len(selected),
        "point_ids_sha256": object_sha256(ids),
        "point_ids_unique": len(ids) == len(set(ids)),
        "selection_predicate": "COLMAP point3D.track_length > 0",
        "xyz_rgb_source": "unchanged existing source COLMAP points3D records",
        "xyz_rgb_float32_records_verified": True,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "ply_format": "binary_little_endian",
        "has_rgb": True,
    }


def export() -> dict[str, object]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("tracked-only export must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    source = layout.assert_member(
        layout.output_root(SOURCE_RUN_ID), label="source output", must_exist=True
    )
    destination = layout.assert_member(
        layout.output_root(DERIVED_RUN_ID), label="tracked-only output"
    )
    if destination.exists():
        raise OutputValidationError(f"Refusing to overwrite {destination}")
    source_manifest_path = source / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    if source_manifest.get("status") != "complete" or source_manifest.get("run_id") != SOURCE_RUN_ID:
        raise OutputValidationError("Source is not the expected complete light VGGSfM run")
    metrics = source_manifest.get("point_cloud", {})
    if (
        metrics.get("point_count") != EXPECTED_TOTAL_POINTS
        or metrics.get("tracked_point_count") != EXPECTED_TRACKED_POINTS
    ):
        raise OutputValidationError("Source manifest point counts differ from reviewed evidence")

    declared: list[Path] = []
    for row in source_manifest.get("files", []):
        path = layout.assert_member(source / row["path"], label="declared source file", must_exist=True)
        current = file_record(path, relative_to=source)
        if current != row:
            raise OutputValidationError(f"Source checksum differs from manifest: {path}")
        declared.append(path)
    expected_names = {
        "colmap/sparse/0/cameras.bin", "colmap/sparse/0/images.bin",
        "colmap/sparse/0/points3D.bin", "point_cloud.ply",
    }
    if {path.relative_to(source).as_posix() for path in declared} != expected_names:
        raise OutputValidationError("Source manifest does not declare the exact reviewed files")
    frozen = [source_manifest_path, *declared]
    before = [absolute_record(path) for path in frozen]

    model = source / "colmap" / "sparse" / "0"
    validation = colmap.validate_binary_model(model)
    count, points = colmap.iter_points3d(model / "points3D.bin")
    if count != EXPECTED_TOTAL_POINTS:
        raise OutputValidationError("Validated points3D count differs from source manifest")

    staging = layout.outputs / f".{DERIVED_RUN_ID}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    layout.assert_member(staging, label="tracked-only staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        output_ply = staging / "point_cloud.ply"
        selection = write_tracked_ply(points, output_ply)
        if selection["point_count"] != EXPECTED_TRACKED_POINTS:
            raise OutputValidationError("Tracked-only selection count is not exactly 11,362")
        after = [absolute_record(path) for path in frozen]
        if before != after:
            raise OutputValidationError("Published source changed during derived export")
        declared_ply = destination / "point_cloud.ply"
        output_record = absolute_record(output_ply, declared=declared_ply)
        receipt: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "method": "derived_vggsfm_tracked_only_review",
            "source_run_id": SOURCE_RUN_ID,
            "derived_run_id": DERIVED_RUN_ID,
            "created_at": utc_now(),
            "scope": "CPU-only diagnostic subset of existing points; no inference, triangulation, bundle adjustment or cleanup.",
            "inference_rerun": False,
            "replacement_for_full_output": False,
            "source_total_point_count": EXPECTED_TOTAL_POINTS,
            "source_trackless_point_count": EXPECTED_TOTAL_POINTS - EXPECTED_TRACKED_POINTS,
            "selection": selection,
            "source_binary_validation": validation,
            "source_files_before": before,
            "source_files_after": after,
            "source_files_before_after_equal": True,
            "source_manifest": absolute_record(source_manifest_path),
            "output_files": [output_record],
            "execution": {
                "python": sys.executable,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        manifest = {
            **receipt,
            "manifest_kind": "derived_tracked_only_review",
            "point_count": selection["point_count"],
            "files": [output_record],
        }
        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    final = read_json(destination / "manifest.json")
    current = absolute_record(destination / "point_cloud.ply")
    if final.get("status") != "complete" or current != final["files"][0]:
        raise OutputValidationError("Published tracked-only output failed checksum validation")
    return read_json(destination / "receipt.json")


def main() -> int:
    result = export()
    print(json.dumps({
        "status": result["status"],
        "source_run_id": result["source_run_id"],
        "derived_run_id": result["derived_run_id"],
        "inference_rerun": result["inference_rerun"],
        "selection": result["selection"],
        "output_files": result["output_files"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
