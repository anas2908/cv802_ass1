#!/usr/bin/env python3
"""Derive a crutch-aware dark-shirt cleanup from the raw dark VGGSfM cloud.

Body membership is a 90% native-mask projection consensus with at least six
usable views.  Crutch membership is independent image-space evidence: a raw
VGGSfM XYZ must project inside reviewed corridors in a triple of annotated
views whose camera-to-point rays are pairwise separated by at least 15 degrees.
The output is their union.  No E1/MVS world geometry, IDs or poses are used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import time
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
from vggsfm_engine.io_utils import (
    atomic_write_json,
    file_record,
    object_sha256,
    read_json,
)
from vggsfm_engine.paths import production_layout
from vggsfm_engine.projection_cleanup import (
    corridor_hits,
    project_original_pixels,
    separated_view_confirmation,
)


SOURCE_RUN = "dark-shirt-vggsfm-all290-a10040-fit-v2"
DERIVED_RUN = SOURCE_RUN + "-mask-crutch-clean-v1"
INPUT_DATASET = "dark_shirt_cleanup_inputs"
EXPECTED_POINTS = 187_308
EXPECTED_IMAGES = 290
MASK_THRESHOLD = 128


def absolute_record(path: Path, *, declared: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    record = file_record(resolved)
    after = resolved.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise OutputValidationError(f"file changed while hashing: {resolved}")
    record["path"] = str((declared or resolved).resolve(strict=False))
    return record


def select_by_support(
    foreground: np.ndarray,
    usable: np.ndarray,
    *,
    minimum_usable: int,
    agreement_threshold: float,
) -> np.ndarray:
    if foreground.shape != usable.shape or minimum_usable < 1:
        raise ValueError("invalid projection-support arrays/rule")
    if not 0.0 <= agreement_threshold <= 1.0:
        raise ValueError("agreement threshold must be in [0,1]")
    required = np.ceil(agreement_threshold * usable - 1e-12).astype(np.int64)
    return (usable >= minimum_usable) & (foreground >= required)


def support_summary(foreground: np.ndarray, usable: np.ndarray) -> dict[str, Any]:
    candidates = {}
    for minimum in (3, 6):
        for threshold in (0.7, 0.8, 0.9):
            candidates[f"min{minimum}_agreement{int(threshold * 100)}"] = int(
                select_by_support(
                    foreground,
                    usable,
                    minimum_usable=minimum,
                    agreement_threshold=threshold,
                ).sum()
            )
    return {
        "candidate_counts": candidates,
        "usable_views": {
            "minimum": int(usable.min()),
            "maximum": int(usable.max()),
            "zero": int((usable == 0).sum()),
            "one_to_five": int(((usable >= 1) & (usable <= 5)).sum()),
            "six_to_thirty_one": int(((usable >= 6) & (usable <= 31)).sum()),
            "thirty_two_to_sixty_three": int(
                ((usable >= 32) & (usable <= 63)).sum()
            ),
            "sixty_four_plus": int((usable >= 64).sum()),
        },
        "foreground_votes": {
            "minimum": int(foreground.min()),
            "maximum": int(foreground.max()),
        },
    }


def _ply_body(path: Path, expected_count: int) -> bytes:
    payload = path.read_bytes()
    marker = b"end_header\n"
    offset = payload.find(marker)
    if offset < 0:
        raise OutputValidationError("source PLY has no end_header")
    body = payload[offset + len(marker) :]
    if len(body) != expected_count * 15:
        raise OutputValidationError("raw PLY is not compact float32 XYZ + uint8 RGB")
    return body


def write_subset_ply(
    points_bin: Path,
    source_ply: Path,
    selected_ids: set[int],
    destination: Path,
) -> dict[str, Any]:
    count, points = colmap.iter_points3d(points_bin)
    if count != EXPECTED_POINTS:
        raise OutputValidationError("raw COLMAP point count changed")
    body = _ply_body(source_ply, count)
    records: list[bytes] = []
    ids: list[int] = []
    tracked = 0
    bounds_min = [math.inf] * 3
    bounds_max = [-math.inf] * 3
    for index, point in enumerate(points):
        expected = struct.pack(
            "<fffBBB", point.x, point.y, point.z,
            point.red, point.green, point.blue,
        )
        raw_record = body[index * 15 : (index + 1) * 15]
        if raw_record != expected:
            raise OutputValidationError("raw PLY record differs from raw COLMAP XYZ/RGB")
        if point.point_id not in selected_ids:
            continue
        records.append(raw_record)
        ids.append(point.point_id)
        tracked += int(point.track_length > 0)
        for axis, value in enumerate((point.x, point.y, point.z)):
            bounds_min[axis] = min(bounds_min[axis], value)
            bounds_max[axis] = max(bounds_max[axis], value)
    if len(records) != len(selected_ids) or not records:
        raise OutputValidationError("selected point IDs do not exactly match raw COLMAP")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment derived dark VGGSfM body-mask plus separated crutch corridors; raw XYZ/RGB unchanged\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    destination.write_bytes(header + b"".join(records))
    return {
        "point_count": len(records),
        "tracked_point_count": tracked,
        "trackless_point_count": len(records) - tracked,
        "point_ids_sha256": object_sha256(ids),
        "source_ply_records_copied_byte_identically": True,
        "xyz_rgb_unchanged": True,
        "all_finite_xyz": all(math.isfinite(x) for x in (*bounds_min, *bounds_max)),
        "has_rgb": True,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "ply_format": "binary_little_endian",
    }


def _verify_manifest_files(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise OutputValidationError("raw manifest has no declared files")
    for row in rows:
        declared = Path(row["path"])
        path = declared if declared.is_absolute() else root / declared
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        actual = file_record(resolved, relative_to=root)
        if actual["bytes"] != row.get("bytes") or actual["sha256"] != row.get("sha256"):
            raise OutputValidationError(f"raw manifest file changed: {resolved}")
        verified.append(absolute_record(resolved))
    return verified


def derive(
    *, minimum_usable: int = 6,
    body_agreement: float = 0.9,
    crutch_minimum_views: int = 3,
    crutch_minimum_angle_degrees: float = 15.0,
) -> dict[str, Any]:
    started = time.monotonic()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cleanup must run inside an active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    raw_root = layout.output_root(SOURCE_RUN)
    destination = layout.output_root(DERIVED_RUN)
    if destination.exists():
        raise OutputValidationError(f"refusing to overwrite {destination}")
    raw_manifest_path = raw_root / "manifest.json"
    raw_manifest = read_json(raw_manifest_path)
    input_root = layout.inputs / INPUT_DATASET
    input_receipt_path = input_root / "receipt.json"
    input_receipt = read_json(input_receipt_path)
    pixel_identity_path = input_root / "pixel_identity_audit.json"
    pixel_identity = read_json(pixel_identity_path)
    request_path = layout.experiment_root(SOURCE_RUN) / "request.json"
    request = read_json(request_path)
    corridor_path = input_root / "crutches" / "corridors.json"
    corridor_evidence = read_json(corridor_path)
    if (
        raw_manifest.get("status") != "complete"
        or raw_manifest.get("run_id") != SOURCE_RUN
        or raw_manifest.get("point_cloud", {}).get("point_count") != EXPECTED_POINTS
    ):
        raise OutputValidationError("raw dark VGGSfM source is not the expected complete run")
    if (
        input_receipt.get("status") != "complete"
        or input_receipt.get("mask_count") != EXPECTED_IMAGES
        or input_receipt.get("no_cross_method_geometry") is not True
    ):
        raise OutputValidationError("method-local dark cleanup inputs are incomplete")
    if (
        pixel_identity.get("status") != "complete"
        or pixel_identity.get("image_count") != EXPECTED_IMAGES
        or pixel_identity.get("all_declared_and_actual_image_hashes_equal") is not True
        or pixel_identity.get("all_mask_hashes_and_native_dimensions_match") is not True
    ):
        raise OutputValidationError("dark cleanup pixel-identity audit is incomplete")
    if corridor_evidence.get("explicit_exclusions") is None:
        raise OutputValidationError("portable corridor evidence lacks geometry exclusions")

    raw_model = raw_root / "colmap" / "sparse" / "0"
    raw_validation = colmap.validate_binary_model(raw_model)
    if (
        raw_validation["point_count"] != EXPECTED_POINTS
        or raw_validation["registered_image_count"] != EXPECTED_IMAGES
    ):
        raise OutputValidationError("raw model counts changed")
    raw_declared_before = _verify_manifest_files(raw_root, raw_manifest)
    frozen_paths = [
        raw_manifest_path, input_receipt_path, pixel_identity_path,
        request_path, corridor_path,
    ]
    frozen_before = [absolute_record(path) for path in frozen_paths]

    mask_rows = {row["image_relative"]: row for row in input_receipt["masks"]}
    if len(mask_rows) != EXPECTED_IMAGES:
        raise OutputValidationError("mask receipt does not contain 290 unique names")
    mask_root = input_root / "masks"
    for name, row in mask_rows.items():
        path = mask_root / Path(row["mask_relative"]).relative_to("masks")
        current = file_record(path, relative_to=input_root)
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            raise OutputValidationError(f"body mask changed: {name}")

    official_to_source = {
        row["official"]: row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    if len(official_to_source) != EXPECTED_IMAGES or set(official_to_source.values()) != set(mask_rows):
        raise OutputValidationError("frozen official-name map and masks do not match")

    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(raw_model))
    ordered_ids = np.asarray(sorted(reconstruction.points3D), dtype=np.uint64)
    xyz = np.stack([
        np.asarray(reconstruction.points3D[int(point_id)].xyz, dtype=np.float64)
        for point_id in ordered_ids
    ])
    foreground = np.zeros(EXPECTED_POINTS, dtype=np.uint16)
    usable = np.zeros(EXPECTED_POINTS, dtype=np.uint16)
    annotations = corridor_evidence["annotated_image_corridors"]
    corridor_hit_rows: list[np.ndarray] = []
    corridor_centers: list[np.ndarray] = []
    projection_rows: list[dict[str, Any]] = []

    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        source_name = official_to_source.get(image.name)
        if source_name is None:
            raise OutputValidationError(f"registered image absent from frozen map: {image.name}")
        camera = reconstruction.cameras[image.camera_id]
        row = mask_rows[source_name]
        mask_path = mask_root / Path(row["mask_relative"]).relative_to("masks")
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened, dtype=np.uint8)
        if mask.shape != (int(camera.height), int(camera.width)):
            raise OutputValidationError(f"mask/camera dimensions disagree: {source_name}")
        pixels, valid = project_original_pixels(image, camera, xyz)
        indices = np.flatnonzero(valid)
        sampled = np.rint(pixels[indices]).astype(np.int64)
        # Continuous pixels are in bounds, but rint at the upper half-pixel can
        # land exactly on width/height; such samples are not nearest-pixel usable.
        nearest_valid = (
            (sampled[:, 0] >= 0) & (sampled[:, 0] < int(camera.width))
            & (sampled[:, 1] >= 0) & (sampled[:, 1] < int(camera.height))
        )
        indices = indices[nearest_valid]
        sampled = sampled[nearest_valid]
        usable[indices] += 1
        foreground[indices] += (
            mask[sampled[:, 1], sampled[:, 0]] >= MASK_THRESHOLD
        ).astype(np.uint16)

        corridor_count = 0
        if source_name in annotations:
            continuous_usable = np.zeros(EXPECTED_POINTS, dtype=bool)
            continuous_usable[np.flatnonzero(valid)] = True
            hits = corridor_hits(
                pixels,
                continuous_usable,
                width=int(camera.width),
                height=int(camera.height),
                groups=annotations[source_name],
            )
            corridor_hit_rows.append(hits)
            center = np.asarray(image.projection_center(), dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise OutputValidationError(f"invalid camera center: {source_name}")
            corridor_centers.append(center)
            corridor_count = int(hits.sum())
        projection_rows.append({
            "image_id": int(image_id),
            "official_name": image.name,
            "source_name": source_name,
            "camera_id": int(image.camera_id),
            "camera_model": str(camera.model).rsplit(".", 1)[-1],
            "width": int(camera.width),
            "height": int(camera.height),
            "body_usable_nearest_count": int(len(indices)),
            "annotated_crutch_corridor": source_name in annotations,
            "crutch_corridor_hit_count": corridor_count,
        })

    if len(corridor_hit_rows) != 6:
        raise OutputValidationError("not all six reviewed crutch views are registered")
    body_selected = select_by_support(
        foreground,
        usable,
        minimum_usable=minimum_usable,
        agreement_threshold=body_agreement,
    )
    crutch_selected = separated_view_confirmation(
        xyz,
        np.stack(corridor_centers),
        np.stack(corridor_hit_rows),
        minimum_views=crutch_minimum_views,
        minimum_pairwise_angle_degrees=crutch_minimum_angle_degrees,
    )
    selected = body_selected | crutch_selected
    selected_ids = {int(value) for value in ordered_ids[selected]}
    if not selected_ids or len(selected_ids) >= EXPECTED_POINTS:
        raise OutputValidationError("dark cleanup selected an implausible empty/full result")

    staging = layout.outputs / f".{DERIVED_RUN}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        projection_path = staging / "projection_rows.json"
        atomic_write_json(projection_path, {
            "schema_version": 1,
            "coordinate_system": "original distorted image pixels from dark VGGSfM cameras",
            "rows_sha256": object_sha256(projection_rows),
            "rows": projection_rows,
        })
        output_ply = staging / "point_cloud.ply"
        selected_metrics = write_subset_ply(
            raw_model / "points3D.bin", raw_root / "point_cloud.ply",
            selected_ids, output_ply,
        )
        raw_declared_after = _verify_manifest_files(raw_root, raw_manifest)
        frozen_after = [absolute_record(path) for path in frozen_paths]
        if raw_declared_before != raw_declared_after or frozen_before != frozen_after:
            raise OutputValidationError("raw/evidence source changed during cleanup")
        output_record = absolute_record(output_ply, declared=destination / "point_cloud.ply")
        projection_record = absolute_record(
            projection_path, declared=destination / "projection_rows.json"
        )
        body_count = int(body_selected.sum())
        crutch_count = int(crutch_selected.sum())
        overlap = int((body_selected & crutch_selected).sum())
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "run_id": DERIVED_RUN,
            "source_run_id": SOURCE_RUN,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": "derived_dark_vggsfm_body_consensus_union_separated_crutch_corridors",
            "inference_rerun": False,
            "triangulation_rerun": False,
            "poses_recalculated": False,
            "xyz_rgb_recalculated": False,
            "raw_output_replaced": False,
            "uses_e10_or_mvs_geometry": False,
            "camera_source": {
                "role": "dark VGGSfM's independently estimated original-resolution cameras",
                "run_id": SOURCE_RUN,
                "model_path": str(raw_model),
                "registered_images": int(reconstruction.num_reg_images()),
                "camera_count": int(reconstruction.num_cameras()),
                "radial_distortion_applied_by_camera_img_from_cam": True,
            },
            "input_evidence": {
                "receipt": absolute_record(input_receipt_path),
                "pixel_identity_audit": absolute_record(pixel_identity_path),
                "pixel_identity_rows_sha256": pixel_identity["rows_sha256"],
                "body_masks": EXPECTED_IMAGES,
                "crutch_corridor_images": len(corridor_hit_rows),
                "crutch_corridors": absolute_record(corridor_path),
                "no_cross_method_geometry": True,
            },
            "body_rule": {
                "usable": "finite, positive-Z, finite original pixel, nearest pixel in bounds",
                "foreground": f"native uint8 body mask value >= {MASK_THRESHOLD}",
                "minimum_usable_views": minimum_usable,
                "minimum_foreground_agreement": body_agreement,
                "occlusion_test": False,
            },
            "crutch_rule": {
                "per_view": "continuous original pixel inside union of reviewed polylines and widths",
                "minimum_distinct_annotated_views": crutch_minimum_views,
                "minimum_pairwise_camera_center_to_point_ray_separation_degrees": crutch_minimum_angle_degrees,
                "occlusion_test": False,
                "uses_world_capsules": False,
                "interpretation": "image-space protection of existing raw XYZ; cannot create absent crutch geometry",
            },
            "union_rule": "body_selected OR crutch_selected",
            "selection": {
                "source_count": EXPECTED_POINTS,
                "body_selected_count": body_count,
                "crutch_selected_count": crutch_count,
                "body_crutch_overlap_count": overlap,
                "crutch_only_protected_count": crutch_count - overlap,
                "retained_count": len(selected_ids),
                "removed_count": EXPECTED_POINTS - len(selected_ids),
                "count_conservation": len(selected_ids) + EXPECTED_POINTS - len(selected_ids) == EXPECTED_POINTS,
            },
            "support_statistics": support_summary(foreground, usable),
            "corridor_per_view_hit_counts": [
                int(row.sum()) for row in corridor_hit_rows
            ],
            "projection_rows": projection_record,
            "selected_point_cloud": selected_metrics,
            "raw_manifest_declared_files_before": raw_declared_before,
            "raw_manifest_declared_files_after": raw_declared_after,
            "raw_manifest_files_unchanged": True,
            "frozen_evidence_before": frozen_before,
            "frozen_evidence_after": frozen_after,
            "frozen_evidence_unchanged": True,
            "output_files": [output_record, projection_record],
            "roles": {
                "raw_full": str(raw_root / "point_cloud.ply"),
                "cleaned_derived": str(destination / "point_cloud.ply"),
            },
            "limitations": [
                "Projection support is not an occlusion test.",
                "Body masks omit crutch shafts; reviewed corridors protect but may include adjacent body/floor points.",
                "Six corridor annotations cannot restore geometry absent from the raw cloud.",
                "Trackless grid extras have projections but no observed feature tracks.",
            ],
            "elapsed_seconds": time.monotonic() - started,
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
            "manifest_kind": "derived_dark_mask_crutch_projection_cleanup",
            "point_cloud": {
                **selected_metrics,
                "body_rule": receipt["body_rule"],
                "crutch_rule": receipt["crutch_rule"],
                "union_rule": receipt["union_rule"],
            },
            "files": [output_record, projection_record],
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
    parser.add_argument("--body-agreement", type=float, default=0.9)
    parser.add_argument("--crutch-minimum-views", type=int, default=3)
    parser.add_argument("--crutch-minimum-angle-degrees", type=float, default=15.0)
    args = parser.parse_args()
    receipt = derive(
        minimum_usable=args.minimum_usable,
        body_agreement=args.body_agreement,
        crutch_minimum_views=args.crutch_minimum_views,
        crutch_minimum_angle_degrees=args.crutch_minimum_angle_degrees,
    )
    print(json.dumps({key: receipt[key] for key in (
        "status", "run_id", "selection", "selected_point_cloud",
        "support_statistics", "corridor_per_view_hit_counts", "elapsed_seconds",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
