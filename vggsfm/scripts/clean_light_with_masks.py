#!/usr/bin/env python3
"""Derive a mask-projection-cleaned light VGGSfM cloud.

Every existing raw VGGSfM point is projected with the independently estimated
VGGSfM camera pose and SIMPLE_RADIAL intrinsics into the matching native-size
person mask. A projection is usable only with positive camera Z, finite pixel
coordinates and an in-bounds nearest pixel. The default keeps points with at
least six usable projections and >=90% mask foreground agreement. This is a
projection consensus, not historical COLMAP track visibility or an occlusion
test; trackless official grid extras are labelled separately in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine import colmap
from vggsfm_engine.errors import OutputValidationError
from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256, read_json
from vggsfm_engine.paths import production_layout


SOURCE_RUN = "light-shirt-vggsfm-v1"
CAMERA_RUN = SOURCE_RUN + "-colmap-consistent-v1"
DERIVED_RUN = SOURCE_RUN + "-mask-projection-clean-v1"
MASK_DATASET = "light_shirt_cleanup_masks"
EXPECTED_POINTS = 414_208
MASK_THRESHOLD = 128


def absolute_record(path: Path, *, declared: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    before = path.stat()
    record = file_record(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise OutputValidationError(f"file changed while hashing: {path}")
    record["path"] = str((declared or path).resolve(strict=False))
    return record


def foreground_values(mask: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Nearest-pixel foreground rule for native uint8 confidence mattes."""

    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("mask must be a 2D uint8 array")
    return mask[y, x] >= MASK_THRESHOLD


def select_by_support(
    foreground: np.ndarray,
    usable: np.ndarray,
    *,
    minimum_usable: int,
    agreement_threshold: float,
) -> np.ndarray:
    if foreground.shape != usable.shape:
        raise ValueError("foreground and usable arrays must have identical shape")
    if minimum_usable < 1 or not 0.0 <= agreement_threshold <= 1.0:
        raise ValueError("invalid support rule")
    foreground_i = foreground.astype(np.int64, copy=False)
    usable_i = usable.astype(np.int64, copy=False)
    return (usable_i >= minimum_usable) & (
        foreground_i >= np.ceil(agreement_threshold * usable_i - 1e-12).astype(np.int64)
    )


def projection_votes(
    reconstruction: Any,
    point_ids: np.ndarray,
    xyz: np.ndarray,
    official_to_source: dict[str, str],
    mask_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Count in-bounds projections and mask foreground votes for every point."""

    usable = np.zeros(len(point_ids), dtype=np.uint16)
    foreground = np.zeros(len(point_ids), dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        if image.name not in official_to_source:
            raise OutputValidationError(f"registered image absent from frozen map: {image.name}")
        source_name = official_to_source[image.name]
        mask_path = mask_root / Path(source_name).with_suffix(".png")
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened, dtype=np.uint8)
        camera = reconstruction.cameras[image.camera_id]
        if mask.shape != (int(camera.height), int(camera.width)):
            raise OutputValidationError(
                f"mask/camera dimensions disagree for {source_name}: "
                f"{mask.shape[::-1]} vs {(camera.width, camera.height)}"
            )
        camera_xyz = np.asarray(image.cam_from_world * xyz, dtype=np.float64)
        positive = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0.0)
        positive_indices = np.flatnonzero(positive)
        if positive_indices.size:
            normalized = (
                camera_xyz[positive_indices, :2]
                / camera_xyz[positive_indices, 2, None]
            )
            pixels = np.asarray(camera.img_from_cam(normalized), dtype=np.float64)
            # Extremely distant finite projections can exceed int64 during
            # nearest-pixel conversion. They cannot be in bounds, so reject
            # them before casting without clamping a usable coordinate.
            finite = np.isfinite(pixels).all(axis=1) & (
                np.abs(pixels) < float(2**62)
            ).all(axis=1)
            rounded = np.zeros_like(pixels, dtype=np.int64)
            rounded[finite] = np.rint(pixels[finite]).astype(np.int64)
            inside = finite & (
                (rounded[:, 0] >= 0)
                & (rounded[:, 0] < int(camera.width))
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < int(camera.height))
            )
            indices = positive_indices[inside]
            sampled = rounded[inside]
            usable[indices] += 1
            foreground[indices] += foreground_values(
                mask, sampled[:, 1], sampled[:, 0]
            ).astype(np.uint16)
        rows.append({
            "image_id": int(image_id),
            "official_name": image.name,
            "source_name": source_name,
            "camera_id": int(image.camera_id),
            "camera_model": str(camera.model).rsplit(".", 1)[-1],
            "width": int(camera.width),
            "height": int(camera.height),
            "positive_z_count": int(positive.sum()),
            "usable_in_bounds_count": int(inside.sum()) if positive_indices.size else 0,
        })
    return foreground, usable, rows


def support_summary(foreground: np.ndarray, usable: np.ndarray) -> dict[str, Any]:
    variants: dict[str, int] = {}
    for minimum in (3, 6):
        for threshold in (0.7, 0.8, 0.9):
            key = f"min{minimum}_agreement{int(threshold * 100)}"
            variants[key] = int(select_by_support(
                foreground, usable, minimum_usable=minimum,
                agreement_threshold=threshold,
            ).sum())
    usable_histogram = {
        "0": int((usable == 0).sum()),
        "1-2": int(((usable >= 1) & (usable <= 2)).sum()),
        "3-5": int(((usable >= 3) & (usable <= 5)).sum()),
        "6-15": int(((usable >= 6) & (usable <= 15)).sum()),
        "16-31": int(((usable >= 16) & (usable <= 31)).sum()),
        "32-63": int(((usable >= 32) & (usable <= 63)).sum()),
        "64+": int((usable >= 64).sum()),
    }
    agreement = np.divide(
        foreground.astype(np.float64), usable,
        out=np.zeros(len(usable), dtype=np.float64), where=usable > 0,
    )
    agreement_histogram = {}
    eligible = usable >= 3
    for lower in range(0, 100, 10):
        upper = lower + 10
        values = eligible & (agreement >= lower / 100.0)
        values &= agreement <= 1.0 if upper == 100 else agreement < upper / 100.0
        agreement_histogram[f"{lower:02d}-{upper:03d}%"] = int(values.sum())
    return {
        "candidate_counts": variants,
        "usable_projection_histogram": usable_histogram,
        "agreement_histogram_for_minimum3": agreement_histogram,
        "maximum_usable_views": int(usable.max()),
        "minimum_foreground_votes": int(foreground.min()),
        "maximum_foreground_votes": int(foreground.max()),
    }


def _ply_body(path: Path, expected_count: int) -> tuple[bytes, bytes]:
    payload = path.read_bytes()
    marker = b"end_header\n"
    offset = payload.find(marker)
    if offset < 0:
        raise OutputValidationError("source PLY has no end_header")
    header, body = payload[: offset + len(marker)], payload[offset + len(marker) :]
    if len(body) != expected_count * 15:
        raise OutputValidationError("source PLY is not the expected compact XYZ/RGB layout")
    return header, body


def write_subset_ply(
    points_bin: Path,
    source_ply: Path,
    selected_ids: set[int],
    destination: Path,
    *,
    expected_count: int = EXPECTED_POINTS,
) -> dict[str, Any]:
    count, points = colmap.iter_points3d(points_bin)
    if count != expected_count:
        raise OutputValidationError("raw COLMAP point count changed")
    _, source_body = _ply_body(source_ply, count)
    selected_records: list[bytes] = []
    selected_point_ids: list[int] = []
    tracked = trackless = 0
    bounds_min = [math.inf, math.inf, math.inf]
    bounds_max = [-math.inf, -math.inf, -math.inf]
    for index, point in enumerate(points):
        packed = struct.pack(
            "<fffBBB", point.x, point.y, point.z,
            point.red, point.green, point.blue,
        )
        source_record = source_body[index * 15 : (index + 1) * 15]
        if packed != source_record:
            raise OutputValidationError("raw PLY record differs from raw COLMAP XYZ/RGB")
        if point.point_id in selected_ids:
            selected_records.append(source_record)
            selected_point_ids.append(point.point_id)
            tracked += int(point.track_length > 0)
            trackless += int(point.track_length == 0)
            for axis, value in enumerate((point.x, point.y, point.z)):
                bounds_min[axis] = min(bounds_min[axis], value)
                bounds_max[axis] = max(bounds_max[axis], value)
    if len(selected_records) != len(selected_ids):
        raise OutputValidationError("selected point IDs do not exactly match raw COLMAP")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment derived VGGSfM native-mask projection consensus; raw XYZ/RGB unchanged\n"
        f"element vertex {len(selected_records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    destination.write_bytes(header + b"".join(selected_records))
    return {
        "point_count": len(selected_records),
        "tracked_point_count": tracked,
        "trackless_point_count": trackless,
        "point_ids_sha256": object_sha256(selected_point_ids),
        "source_ply_records_copied_byte_identically": True,
        "xyz_rgb_unchanged": True,
        "all_finite_xyz": all(math.isfinite(x) for x in (*bounds_min, *bounds_max)),
        "has_rgb": True,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "ply_format": "binary_little_endian",
    }


def derive(*, minimum_usable: int = 6, agreement_threshold: float = 0.9) -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cleanup must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    raw_root = layout.output_root(SOURCE_RUN)
    camera_root = layout.output_root(CAMERA_RUN)
    destination = layout.output_root(DERIVED_RUN)
    if destination.exists():
        raise OutputValidationError(f"refusing to overwrite {destination}")
    raw_manifest_path = raw_root / "manifest.json"
    camera_receipt_path = camera_root / "review_export_receipt.json"
    camera_manifest_path = camera_root / "manifest.json"
    mask_receipt_path = layout.inputs / MASK_DATASET / "receipt.json"
    request_path = layout.experiment_root(SOURCE_RUN) / "request.json"
    raw_manifest = read_json(raw_manifest_path)
    camera_receipt = read_json(camera_receipt_path)
    mask_receipt = read_json(mask_receipt_path)
    request = read_json(request_path)
    if raw_manifest.get("status") != "complete" or raw_manifest.get("run_id") != SOURCE_RUN:
        raise OutputValidationError("raw VGGSfM source is not complete")
    if raw_manifest.get("point_cloud", {}).get("point_count") != EXPECTED_POINTS:
        raise OutputValidationError("raw VGGSfM point count changed")
    if (
        camera_receipt.get("status") != "complete"
        or camera_receipt.get("source_run_id") != SOURCE_RUN
        or not camera_receipt.get("invariants", {}).get("point_ids_xyz_rgb_tracks_unchanged")
    ):
        raise OutputValidationError("coordinate-consistent camera model lacks invariant proof")
    if mask_receipt.get("status") != "complete" or mask_receipt.get("mask_count") != 125:
        raise OutputValidationError("method-local mask receipt is incomplete")

    mask_rows = {row["image_relative"]: row for row in mask_receipt["rows"]}
    mask_root = layout.inputs / MASK_DATASET / "masks"
    for name, row in mask_rows.items():
        path = mask_root / Path(name).with_suffix(".png")
        current = file_record(path, relative_to=layout.inputs / MASK_DATASET)
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            raise OutputValidationError(f"method-local mask changed: {name}")

    official_to_source = {
        row["official"]: row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    if set(official_to_source.values()) != set(mask_rows):
        raise OutputValidationError("mask receipt does not exactly cover frozen input map")

    raw_model = raw_root / "colmap" / "sparse" / "0"
    camera_model = camera_root / "colmap" / "sparse" / "0"
    raw_validation = colmap.validate_binary_model(raw_model)
    camera_validation = colmap.validate_binary_model(camera_model)
    if raw_validation["point_count"] != camera_validation["point_count"]:
        raise OutputValidationError("raw and camera-review model point counts differ")

    frozen_paths = [
        raw_manifest_path, raw_root / "point_cloud.ply",
        raw_model / "cameras.bin", raw_model / "images.bin", raw_model / "points3D.bin",
        camera_manifest_path, camera_receipt_path,
        camera_model / "cameras.bin", camera_model / "images.bin", camera_model / "points3D.bin",
        mask_receipt_path, request_path,
    ]
    source_before = [absolute_record(path) for path in frozen_paths]

    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(camera_model))
    ordered_ids = np.asarray(sorted(reconstruction.points3D), dtype=np.uint64)
    xyz = np.stack(
        [np.asarray(reconstruction.points3D[int(point_id)].xyz) for point_id in ordered_ids]
    ).astype(np.float64, copy=False)
    foreground, usable, projection_rows = projection_votes(
        reconstruction, ordered_ids, xyz, official_to_source, mask_root
    )
    selected = select_by_support(
        foreground, usable,
        minimum_usable=minimum_usable,
        agreement_threshold=agreement_threshold,
    )
    selected_ids = {int(value) for value in ordered_ids[selected]}
    if not selected_ids or len(selected_ids) >= EXPECTED_POINTS:
        raise OutputValidationError("cleanup selected an implausible empty/full result")

    staging = layout.outputs / f".{DERIVED_RUN}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        output_ply = staging / "point_cloud.ply"
        selected_metrics = write_subset_ply(
            raw_model / "points3D.bin", raw_root / "point_cloud.ply",
            selected_ids, output_ply,
        )
        if selected_metrics["point_count"] + (EXPECTED_POINTS - len(selected_ids)) != EXPECTED_POINTS:
            raise OutputValidationError("retained/removed point conservation failed")
        summary = support_summary(foreground, usable)
        source_after = [absolute_record(path) for path in frozen_paths]
        if source_before != source_after:
            raise OutputValidationError("raw/evidence source changed during cleanup")
        output_record = absolute_record(
            output_ply, declared=destination / "point_cloud.ply"
        )
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "run_id": DERIVED_RUN,
            "source_run_id": SOURCE_RUN,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": "derived_vggsfm_native_mask_projection_consensus",
            "inference_rerun": False,
            "triangulation_rerun": False,
            "raw_output_replaced": False,
            "uses_e10_poses": False,
            "camera_source": {
                "role": "independently estimated VGGSfM cameras with original image dimensions",
                "run_id": CAMERA_RUN,
                "model_path": str(camera_model),
                "manifest": absolute_record(camera_manifest_path),
                "receipt": absolute_record(camera_receipt_path),
                "camera_count": int(reconstruction.num_cameras()),
                "registered_image_count": int(reconstruction.num_reg_images()),
                "camera_models": sorted({
                    str(camera.model).rsplit(".", 1)[-1]
                    for camera in reconstruction.cameras.values()
                }),
                "radial_distortion_applied_by_camera_img_from_cam": True,
            },
            "mask_input": {
                "receipt": absolute_record(mask_receipt_path),
                "mask_count": 125,
                "rows_sha256": mask_receipt["rows_sha256"],
                "native_original_image_dimensions": True,
                "polarity": "uint8 value >=128 is person foreground",
            },
            "projection_rule": {
                "coordinates": "VGGSfM world XYZ -> cam_from_world -> normalized xy -> distorted original-image pixels via camera.img_from_cam",
                "usable": "finite camera coordinate, positive Z, finite projected pixel, nearest-pixel coordinate in camera/mask bounds",
                "nearest_pixel": "NumPy rint",
                "foreground": f"native uint8 person mask value >= {MASK_THRESHOLD}",
                "minimum_usable_views": minimum_usable,
                "minimum_foreground_agreement": agreement_threshold,
                "occlusion_test": False,
                "interpretation": (
                    "Projection-based consensus across usable cameras. Trackless extra points "
                    "have no true visibility, so usable projection is not historical track support."
                ),
            },
            "support_statistics": summary,
            "projection_rows_sha256": object_sha256(projection_rows),
            "projection_image_count": len(projection_rows),
            "source_point_count": EXPECTED_POINTS,
            "retained_point_count": len(selected_ids),
            "removed_point_count": EXPECTED_POINTS - len(selected_ids),
            "point_count_conservation": len(selected_ids) + (EXPECTED_POINTS - len(selected_ids)) == EXPECTED_POINTS,
            "selected_point_cloud": selected_metrics,
            "source_files_before": source_before,
            "source_files_after": source_after,
            "source_files_before_after_equal": True,
            "output_files": [output_record],
            "roles": {
                "raw_full": str(raw_root / "point_cloud.ply"),
                "cleaned_derived": str(destination / "point_cloud.ply"),
            },
            "execution": {
                "python": sys.executable,
                "pycolmap_version": pycolmap.__version__,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        manifest = {
            **receipt,
            "manifest_kind": "derived_mask_projection_cleanup",
            "point_cloud": {
                **selected_metrics,
                "selection_rule": receipt["projection_rule"],
            },
            "files": [output_record],
        }
        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_json(staging / "cleanup_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return read_json(destination / "cleanup_receipt.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-usable", type=int, default=6)
    parser.add_argument("--agreement", type=float, default=0.9)
    args = parser.parse_args()
    result = derive(
        minimum_usable=args.minimum_usable,
        agreement_threshold=args.agreement,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "run_id", "source_point_count", "retained_point_count",
        "removed_point_count", "support_statistics", "selected_point_cloud",
        "output_files",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
