#!/usr/bin/env python3
"""Create a coordinate-consistent, read-only-derived COLMAP export.

The pinned official VGGSfM runner rescales cameras back to each original
image size.  In the assignment run its resolved option
``shift_point2d_to_original_res`` is false, so ``images.bin`` retains 2D
observations in padded 1024-pixel coordinates.  This CPU-only command applies
the upstream transform that would have run when that option is true, then
recomputes real track reprojection errors.

It is deliberately a *derived review export*, not inference.  It accepts only
an immutable, checksum-valid, successfully published run, writes a separately
named output, never overwrites anything, and proves that cameras, poses, 3D
geometry, RGB and tracks did not change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Keep source bytecode out of the GitHub-ready tree even if a caller forgets
# ``-B``. Runtime commands still set PYTHONPYCACHEPREFIX to DATA_ROOT.
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine import colmap
from vggsfm_engine.constants import (
    OFFICIAL_COMMIT,
    OPTIONAL_COLMAP_FILES,
    REQUIRED_COLMAP_FILES,
)
from vggsfm_engine.errors import OutputValidationError, ProvenanceError
from vggsfm_engine.io_utils import (
    atomic_write_json,
    file_sha256,
    object_sha256,
    read_json,
    validate_identifier,
)
from vggsfm_engine.paths import Layout, production_layout, require_within


DERIVED_SUFFIX = "-colmap-consistent-v1"
EXPECTED_PYCOLMAP = "3.10.0"
FORMULA = "xy_original=(xy_padded-top_left_abs)*resize_ratio"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def absolute_file_record(path: Path, *, declared_path: Path | None = None) -> dict[str, Any]:
    """Return the audit's exact absolute ``path/bytes/sha256`` file record."""

    path = path.resolve(strict=True)
    stat_before = path.stat()
    digest = file_sha256(path)
    stat_after = path.stat()
    compared = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino")
    if any(getattr(stat_before, key) != getattr(stat_after, key) for key in compared):
        raise OutputValidationError(f"File changed while it was hashed: {path}")
    display = (declared_path or path).resolve(strict=False)
    if not display.is_absolute():
        raise OutputValidationError(f"Audit record path is not absolute: {display}")
    return {"path": str(display), "bytes": stat_after.st_size, "sha256": digest}


def records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [absolute_file_record(path) for path in sorted(set(paths))]


def _record_map(items: list[dict[str, Any]]) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for item in items:
        if set(item) != {"path", "bytes", "sha256"} or not Path(item["path"]).is_absolute():
            raise OutputValidationError("Malformed non-absolute file record")
        result[item["path"]] = (int(item["bytes"]), str(item["sha256"]))
    if len(result) != len(items):
        raise OutputValidationError("Duplicate path in file records")
    return result


def exact_crop_correction(width: int, height: int, img_size: int) -> dict[str, Any]:
    """Reproduce the pinned loader/runner float32 crop transform exactly."""

    if min(width, height, img_size) <= 0:
        raise ValueError("Image dimensions and img_size must be positive")
    length = max(width, height)
    left = (width - length) // 2
    top = (height - length) // 2
    # calculate_crop_parameters performs the division in NumPy float64, then
    # torch.tensor(...).float() rounds the stored bbox to float32.
    bbox_after = np.asarray([left, top], dtype=np.int64) / length * img_size
    top_left_abs = np.abs(np.asarray(bbox_after, dtype=np.float32))
    # crop_params is float32, so max()/img_size is also float32; .item() then
    # exposes that exact rounded value to the NumPy point2D expression.
    resize_ratio = float(np.float32(np.float32(length) / np.float32(img_size)))
    return {
        "original_width": int(width),
        "original_height": int(height),
        "square_length": int(length),
        "left": int(left),
        "top": int(top),
        "top_left_abs": [float(top_left_abs[0]), float(top_left_abs[1])],
        "resize_ratio": resize_ratio,
        "img_size": int(img_size),
    }


def correct_xy(xy: Any, correction: dict[str, Any]) -> np.ndarray:
    source = np.asarray(xy, dtype=np.float64)
    if source.shape != (2,) or not np.all(np.isfinite(source)):
        raise OutputValidationError(f"Invalid point2D coordinate: {source!r}")
    top_left = np.asarray(correction["top_left_abs"], dtype=np.float32)
    ratio = float(correction["resize_ratio"])
    result = (source - top_left) * ratio
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise OutputValidationError("Coordinate correction produced a non-finite point")
    return result


def residual_statistics(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise OutputValidationError("No tracked observations are available for reprojection audit")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise OutputValidationError("Reprojection residuals must be finite and nonnegative")
    return {
        "count": int(array.size),
        "minimum_px": float(array.min()),
        "maximum_px": float(array.max()),
        "mean_px": float(array.mean()),
        "median_px": float(np.median(array)),
        "rmse_px": float(np.sqrt(np.mean(np.square(array)))),
        "p90_px": float(np.percentile(array, 90)),
        "p95_px": float(np.percentile(array, 95)),
        "p99_px": float(np.percentile(array, 99)),
        "all_finite_nonnegative": True,
    }


def _project_point(reconstruction: Any, image: Any, xyz: Any) -> np.ndarray:
    camera = reconstruction.cameras[image.camera_id]
    camera_xyz = np.asarray(image.cam_from_world * np.asarray(xyz, dtype=np.float64))
    if camera_xyz.shape != (3,) or not np.all(np.isfinite(camera_xyz)):
        raise OutputValidationError("Pose projection produced invalid camera coordinates")
    if camera_xyz[2] <= 0:
        raise OutputValidationError("A tracked 3D point is not in front of its observing camera")
    normalized = (camera_xyz[:2] / camera_xyz[2]).reshape(1, 2)
    projected = np.asarray(camera.img_from_cam(normalized), dtype=np.float64).reshape(-1, 2)
    if projected.shape != (1, 2) or not np.all(np.isfinite(projected)):
        raise OutputValidationError("Camera projection produced an invalid pixel")
    return projected[0]


def actual_reprojection_errors(reconstruction: Any) -> tuple[dict[str, Any], dict[int, float]]:
    """Measure every reciprocal track observation using cameras and poses."""

    residuals: list[float] = []
    point_means: dict[int, float] = {}
    tracked_points = 0
    trackless_points = 0
    for point_id in sorted(reconstruction.points3D):
        point = reconstruction.points3D[point_id]
        elements = list(point.track.elements)
        if not elements:
            trackless_points += 1
            continue
        tracked_points += 1
        local: list[float] = []
        for element in elements:
            if element.image_id not in reconstruction.images:
                raise OutputValidationError(f"Point {point_id} track names missing image")
            image = reconstruction.images[element.image_id]
            if element.point2D_idx >= len(image.points2D):
                raise OutputValidationError(f"Point {point_id} track has invalid point2D index")
            observation = image.points2D[element.point2D_idx]
            if int(observation.point3D_id) != int(point_id):
                raise OutputValidationError(f"Point {point_id} track is not reciprocal")
            projected = _project_point(reconstruction, image, point.xyz)
            residual = float(np.linalg.norm(np.asarray(observation.xy) - projected))
            if not math.isfinite(residual) or residual < 0:
                raise OutputValidationError("Invalid measured reprojection residual")
            local.append(residual)
            residuals.append(residual)
        point_means[int(point_id)] = float(np.mean(np.asarray(local, dtype=np.float64)))
    summary = residual_statistics(residuals)
    summary.update({
        "tracked_observation_count": len(residuals),
        "tracked_point_count": tracked_points,
        "trackless_point_count": trackless_points,
        "total_point_count": tracked_points + trackless_points,
    })
    return summary, point_means


def apply_coordinate_correction(
    reconstruction: Any, corrections: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Correct every saved point2D and audit linked observation bounds.

    COLMAP permits observations outside an image.  In particular, VGGSfM can
    retain tracks on the square padding that the exact upstream conversion
    maps outside the original rectangular image.  We must report those rather
    than clamp/delete them, because either operation would violate the exact
    formula and change tracks.
    """

    if set(corrections) != {image.name for image in reconstruction.images.values()}:
        raise OutputValidationError("Crop corrections do not exactly cover registered images")
    all_count = linked_count = changed_count = in_bounds_count = 0
    out_of_bounds: list[dict[str, Any]] = []
    affected_images: set[str] = set()
    minimum_margin = math.inf
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        correction = corrections[image.name]
        camera = reconstruction.cameras[image.camera_id]
        if (int(camera.width), int(camera.height)) != (
            correction["original_width"], correction["original_height"]
        ):
            raise OutputValidationError(f"Camera dimensions differ from source image: {image.name}")
        for observation in image.points2D:
            before = np.asarray(observation.xy, dtype=np.float64).copy()
            after = correct_xy(before, correction)
            observation.xy = after
            all_count += 1
            changed_count += int(not np.array_equal(before, after))
            if observation.has_point3D():
                linked_count += 1
                x, y = (float(after[0]), float(after[1]))
                within = 0.0 <= x < camera.width and 0.0 <= y < camera.height
                in_bounds_count += int(within)
                signed_margin = min(
                    x, y, float(camera.width) - x, float(camera.height) - y
                )
                if not within:
                    affected_images.add(image.name)
                    out_of_bounds.append({
                        "image": image.name,
                        "xy": [x, y],
                        "camera_width": int(camera.width),
                        "camera_height": int(camera.height),
                        "signed_border_margin_px": float(signed_margin),
                        "reason": "Exact conversion of a VGGSfM track on square padding; not clamped or deleted.",
                    })
                minimum_margin = min(minimum_margin, signed_margin)
    outside_count = linked_count - in_bounds_count
    worst_examples = sorted(
        out_of_bounds, key=lambda item: item["signed_border_margin_px"]
    )[:20]
    return {
        "registered_image_count": len(corrections),
        "all_point2d_count": all_count,
        "linked_point2d_count": linked_count,
        "changed_point2d_count": changed_count,
        "all_linked_finite": True,
        "linked_in_camera_bounds_count": in_bounds_count,
        "linked_outside_camera_bounds_count": outside_count,
        "linked_bounds_count_sum_matches_total": in_bounds_count + outside_count == linked_count,
        "out_of_bounds_affected_image_count": len(affected_images),
        "out_of_bounds_affected_images": sorted(affected_images),
        "all_linked_within_camera_bounds": in_bounds_count == linked_count,
        "worst_out_of_bounds_examples": worst_examples,
        "bounds_interpretation": (
            "COLMAP allows out-of-image observations. Exact upstream correction is preserved; "
            "padding tracks are reported and never clamped or deleted."
        ),
        "policy": "preserve_exact_formula_and_tracks_no_clamp",
        "minimum_linked_border_margin_px": float(minimum_margin),
    }


def set_reprojection_errors(reconstruction: Any, point_means: dict[int, float]) -> dict[str, int]:
    tracked = trackless = 0
    for point_id in sorted(reconstruction.points3D):
        point = reconstruction.points3D[point_id]
        if point.track.length() == 0:
            point.error = -1.0
            trackless += 1
        else:
            if int(point_id) not in point_means:
                raise OutputValidationError(f"No measured error for tracked point {point_id}")
            point.error = float(point_means[int(point_id)])
            tracked += 1
    return {"tracked_errors_recomputed": tracked, "trackless_errors_set_to_minus_one": trackless}


def _track_tuple(point: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(item.image_id), int(item.point2D_idx)) for item in point.track.elements)


def semantic_invariants(
    source: Any,
    derived: Any,
    corrections: dict[str, dict[str, Any]],
    expected_errors: dict[int, float],
) -> dict[str, Any]:
    """Fail unless the only semantic changes are point2D.xy and point.error."""

    if set(source.cameras) != set(derived.cameras):
        raise OutputValidationError("Camera IDs changed")
    for camera_id in source.cameras:
        left, right = source.cameras[camera_id], derived.cameras[camera_id]
        if (
            str(left.model) != str(right.model)
            or int(left.width) != int(right.width)
            or int(left.height) != int(right.height)
            or not np.array_equal(np.asarray(left.params), np.asarray(right.params))
        ):
            raise OutputValidationError(f"Camera {camera_id} model/dimensions/intrinsics changed")

    if set(source.images) != set(derived.images):
        raise OutputValidationError("Image IDs changed")
    changed_xy = 0
    for image_id in source.images:
        left, right = source.images[image_id], derived.images[image_id]
        if left.name != right.name or int(left.camera_id) != int(right.camera_id):
            raise OutputValidationError(f"Image {image_id} name or camera reference changed")
        if not np.array_equal(left.cam_from_world.matrix(), right.cam_from_world.matrix()):
            raise OutputValidationError(f"Image {image_id} q/t pose changed")
        if len(left.points2D) != len(right.points2D):
            raise OutputValidationError(f"Image {image_id} point2D count changed")
        correction = corrections[left.name]
        for index, (old, new) in enumerate(zip(left.points2D, right.points2D, strict=True)):
            if int(old.point3D_id) != int(new.point3D_id):
                raise OutputValidationError(f"Image {image_id} point3D reference changed")
            expected = correct_xy(old.xy, correction)
            if not np.array_equal(np.asarray(new.xy), expected):
                raise OutputValidationError(
                    f"Image {image_id} point2D {index} differs from the exact crop formula"
                )
            changed_xy += int(not np.array_equal(np.asarray(old.xy), np.asarray(new.xy)))

    if set(source.points3D) != set(derived.points3D):
        raise OutputValidationError("Point3D IDs changed")
    tracked = trackless = changed_errors = 0
    for point_id in source.points3D:
        left, right = source.points3D[point_id], derived.points3D[point_id]
        if not np.array_equal(np.asarray(left.xyz), np.asarray(right.xyz)):
            raise OutputValidationError(f"Point {point_id} XYZ changed")
        if not np.array_equal(np.asarray(left.color), np.asarray(right.color)):
            raise OutputValidationError(f"Point {point_id} RGB changed")
        if _track_tuple(left) != _track_tuple(right):
            raise OutputValidationError(f"Point {point_id} track changed")
        changed_errors += int(float(left.error) != float(right.error))
        if right.track.length() == 0:
            trackless += 1
            if float(right.error) != -1.0:
                raise OutputValidationError(f"Trackless point {point_id} error is not -1")
        else:
            tracked += 1
            expected = expected_errors[int(point_id)]
            if not math.isclose(float(right.error), expected, rel_tol=0.0, abs_tol=1e-12):
                raise OutputValidationError(f"Point {point_id} error is not its measured mean")

    return {
        "camera_ids_unchanged": True,
        "camera_models_dimensions_intrinsics_unchanged": True,
        "image_ids_names_camera_refs_unchanged": True,
        "image_qt_poses_unchanged": True,
        "image_point3d_references_unchanged": True,
        "point_ids_xyz_rgb_tracks_unchanged": True,
        "point2d_xy_matches_exact_formula": True,
        "point_errors_match_actual_post_residual_means": True,
        "trackless_point_errors_equal_minus_one": True,
        "only_point2d_xy_and_point_error_fields_permitted_to_differ": True,
        "changed_point2d_xy_count": changed_xy,
        "changed_point_error_count": changed_errors,
        "tracked_point_count": tracked,
        "trackless_point_count": trackless,
    }


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if completed.returncode:
        raise ProvenanceError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _find_lines(path: Path, needles: dict[str, str]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    found: dict[str, Any] = {}
    for label, needle in needles.items():
        matches = [index + 1 for index, line in enumerate(lines) if needle in line]
        if not matches:
            raise ProvenanceError(f"Expected pinned-source expression is absent: {needle!r}")
        found[label] = {"line_numbers": matches, "text": needle}
    return found


def official_source_evidence(layout: Layout) -> dict[str, Any]:
    root = layout.assert_member(layout.official_root, label="official source", must_exist=True)
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != OFFICIAL_COMMIT:
        raise ProvenanceError(f"Expected official commit {OFFICIAL_COMMIT}; found {revision}")
    tracked_status = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise ProvenanceError("Official checkout has tracked modifications")
    runner = root / "vggsfm" / "runners" / "runner.py"
    loader = root / "vggsfm" / "datasets" / "demo_loader.py"
    config = root / "cfgs" / "demo.yaml"
    evidence = {
        "repository_root": str(root),
        "commit": revision,
        "expected_commit": OFFICIAL_COMMIT,
        "tracked_checkout_clean": True,
        "runner": absolute_file_record(runner),
        "runner_lines": _find_lines(runner, {
            "resolved_config_passed_to_helper": "shift_point2d_to_original_res=self.cfg.shift_point2d_to_original_res,",
            "helper_default": "shift_point2d_to_original_res=False,",
            "conditional": "if shift_point2d_to_original_res:",
            "upstream_formula": "point2D.xy = (point2D.xy - top_left) * resize_ratio",
        }),
        "crop_parameter_source": absolute_file_record(loader),
        "crop_parameter_lines": _find_lines(loader, {
            "square_length": "crop_dim = max(h, w)",
            "top": "top = (h - crop_dim) // 2",
            "left": "left = (w - crop_dim) // 2",
            "bbox_rescale": "bbox_after = bbox / crop_dim * img_size",
        }),
        "default_config": absolute_file_record(config),
        "default_config_lines": _find_lines(config, {
            "shift_disabled": "shift_point2d_to_original_res: False",
        }),
    }
    return evidence


def _resolved_setting_evidence(log_path: Path, command: list[Any]) -> dict[str, Any]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise OutputValidationError("Successful attempt receipt command is not an argv list")
    lines = log_path.read_text(encoding="utf-8", errors="strict").splitlines()
    matches = [index + 1 for index, line in enumerate(lines)
               if re.fullmatch(r"\s*shift_point2d_to_original_res:\s*false\s*", line)]
    if not matches:
        raise OutputValidationError(
            "Official log does not explicitly resolve shift_point2d_to_original_res=false"
        )
    overrides = [str(item) for item in command if str(item).startswith("shift_point2d_to_original_res=")]
    if any(item.lower().endswith("=true") for item in overrides):
        raise OutputValidationError("Official command explicitly enabled the shift; no repair is valid")
    return {
        "shift_point2d_to_original_res": False,
        "official_log": absolute_file_record(log_path),
        "resolved_config_line_numbers": matches,
        "resolved_config_exact_value": "false",
        "command_overrides": overrides,
        "command_did_not_enable_shift": True,
    }


def _verify_manifest_files(output: Path, manifest: dict[str, Any]) -> list[Path]:
    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise OutputValidationError("Source manifest has no output file records")
    paths: list[Path] = []
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise OutputValidationError("Source manifest contains malformed file record")
        path = require_within(output / item["path"], output, label="published file", must_exist=True)
        current = absolute_file_record(path)
        if current["bytes"] != item.get("bytes") or current["sha256"] != item.get("sha256"):
            raise OutputValidationError(f"Source published checksum mismatch: {path}")
        paths.append(path)
    required = {str(output / "colmap" / "sparse" / "0" / name) for name in REQUIRED_COLMAP_FILES}
    required.add(str(output / "point_cloud.ply"))
    if not required <= {str(path) for path in paths}:
        raise OutputValidationError("Source manifest does not declare all three binaries and PLY")
    return paths


def _request_fingerprint_is_valid(request: dict[str, Any]) -> bool:
    descriptor = {key: value for key, value in request.items()
                  if key not in {"created_at", "request_fingerprint"}}
    return object_sha256(descriptor) == request.get("request_fingerprint")


def _input_rows(request: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows = request.get("input_manifest", {}).get("images")
    mapping = request.get("official_image_name_map")
    if not isinstance(rows, list) or not isinstance(mapping, list) or not rows:
        raise OutputValidationError("Request has no complete input image manifest/name map")
    source_rows = {item["path"]: item for item in rows}
    official = {item["official"]: item["source"] for item in mapping}
    if len(source_rows) != len(rows) or len(official) != len(mapping):
        raise OutputValidationError("Duplicate source or official image name in request")
    if set(official.values()) != set(source_rows):
        raise OutputValidationError("Official image map does not exactly cover input manifest")
    return source_rows, official


def _crop_corrections(
    layout: Layout, request: dict[str, Any], reconstruction: Any
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[Path]]:
    source_rows, official_map = _input_rows(request)
    dataset = validate_identifier(request["dataset"], label="dataset")
    dataset_root = layout.assert_member(layout.dataset_root(dataset), label="dataset", must_exist=True)
    img_size = int(request.get("profile", {}).get("img_size", 0))
    if img_size <= 0:
        raise OutputValidationError("Request profile has invalid img_size")
    if request.get("profile", {}).get("shared_camera") is not False:
        raise OutputValidationError("Review export requires independent per-image cameras")

    corrections: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    registered_names = {image.name for image in reconstruction.images.values()}
    if not registered_names or not registered_names <= set(official_map):
        raise OutputValidationError("Registered names are absent from the frozen official name map")
    for name in sorted(registered_names):
        source_relative = official_map[name]
        source_path = require_within(
            dataset_root / source_relative, dataset_root, label="input image", must_exist=True
        )
        image_record = absolute_file_record(source_path)
        requested = source_rows[source_relative]
        if (image_record["bytes"], image_record["sha256"]) != (
            requested.get("bytes"), requested.get("sha256")
        ):
            raise OutputValidationError(f"Input image differs from frozen request: {source_path}")
        with Image.open(source_path) as opened:
            width, height = opened.size
        correction = exact_crop_correction(width, height, img_size)
        correction.update({
            "official_image": name,
            "source_relative": source_relative,
            "source_image": image_record,
        })
        corrections[name] = correction
        rows.append(correction)
        image_paths.append(source_path)
    return corrections, rows, image_paths


def _reload_summary(reconstruction: Any, version: str) -> dict[str, Any]:
    return {
        "pycolmap_version": version,
        "camera_count": int(reconstruction.num_cameras()),
        "registered_image_count": int(reconstruction.num_reg_images()),
        "point_count": int(reconstruction.num_points3D()),
        "image_names": sorted(image.name for image in reconstruction.images.values()),
    }


def derive(run_id: str, *, expected_pycolmap: str = EXPECTED_PYCOLMAP) -> dict[str, Any]:
    validate_identifier(run_id, label="run ID")
    derived_run_id = validate_identifier(run_id + DERIVED_SUFFIX, label="derived run ID")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    source_output = layout.assert_member(
        layout.output_root(run_id), label="source published output", must_exist=True
    )
    destination = layout.assert_member(layout.output_root(derived_run_id), label="derived output")
    if destination.exists():
        raise OutputValidationError(f"Refusing to overwrite derived output: {destination}")
    source_manifest_path = source_output / "manifest.json"
    manifest = read_json(source_manifest_path)
    if manifest.get("status") != "complete" or manifest.get("run_id") != run_id:
        raise OutputValidationError("Source must be a successfully published complete run")
    declared_source_paths = _verify_manifest_files(source_output, manifest)

    experiment = layout.assert_member(
        layout.experiment_root(run_id), label="source experiment", must_exist=True
    )
    request_path = experiment / "request.json"
    request = read_json(request_path)
    if not _request_fingerprint_is_valid(request):
        raise OutputValidationError("Frozen request fingerprint is invalid")
    fingerprint = request["request_fingerprint"]
    if manifest.get("request_fingerprint") != fingerprint:
        raise OutputValidationError("Published manifest does not match request fingerprint")
    receipt_path = require_within(
        Path(manifest.get("experiment_receipt", "")), experiment,
        label="successful attempt receipt", must_exist=True,
    )
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "complete"
        or receipt.get("run_id") != run_id
        or receipt.get("request_fingerprint") != fingerprint
        or Path(receipt.get("output_manifest", "")).resolve(strict=False)
        != source_manifest_path.resolve(strict=True)
    ):
        raise OutputValidationError("Successful attempt receipt is not bound to published output")
    log_path = require_within(
        Path(receipt.get("log", "")), experiment, label="official log", must_exist=True
    )
    setting = _resolved_setting_evidence(log_path, receipt.get("command", []))
    source_code = official_source_evidence(layout)

    model = source_output / "colmap" / "sparse" / "0"
    source_binary_validation = colmap.validate_binary_model(model)
    source_ply = source_output / "point_cloud.ply"
    published_paths = [source_manifest_path, *declared_source_paths]
    source_before = records(published_paths)

    import pycolmap
    if pycolmap.__version__ != expected_pycolmap:
        raise ProvenanceError(
            f"Expected PyCOLMAP {expected_pycolmap}; found {pycolmap.__version__}"
        )
    source_reconstruction = pycolmap.Reconstruction(str(model))
    derived_reconstruction = pycolmap.Reconstruction(str(model))
    corrections, correction_rows, registered_input_paths = _crop_corrections(
        layout, request, source_reconstruction
    )
    frozen_paths = [
        *published_paths, request_path, receipt_path, log_path,
        Path(source_code["runner"]["path"]),
        Path(source_code["crop_parameter_source"]["path"]),
        Path(source_code["default_config"]["path"]),
        *registered_input_paths,
    ]
    frozen_before = records(frozen_paths)
    crop_digest = object_sha256(correction_rows)
    pre_residuals, _ = actual_reprojection_errors(source_reconstruction)
    correction_metrics = apply_coordinate_correction(derived_reconstruction, corrections)
    post_residuals, point_means = actual_reprojection_errors(derived_reconstruction)
    if post_residuals["count"] != pre_residuals["count"]:
        raise OutputValidationError("Track observation count changed during correction")
    if not post_residuals["mean_px"] < pre_residuals["mean_px"]:
        raise OutputValidationError("Correct coordinate conversion did not improve mean residual")
    error_metrics = set_reprojection_errors(derived_reconstruction, point_means)

    staging = layout.outputs / (
        f".{derived_run_id}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    )
    layout.assert_member(staging, label="derived staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        output_model = staging / "colmap" / "sparse" / "0"
        output_model.mkdir(parents=True)
        derived_reconstruction.write(str(output_model))
        for name in OPTIONAL_COLMAP_FILES:
            source_optional = model / name
            output_optional = output_model / name
            if source_optional.is_file() and not output_optional.exists():
                shutil.copyfile(source_optional, output_optional)
        output_ply = staging / "point_cloud.ply"
        shutil.copyfile(source_ply, output_ply)
        crop_path = staging / "crop_parameters.json"
        atomic_write_json(crop_path, {
            "schema_version": 1,
            "formula": FORMULA,
            "official_img_size": int(request["profile"]["img_size"]),
            "registered_image_count": len(correction_rows),
            "rows_sha256": crop_digest,
            "rows": correction_rows,
        })

        output_binary_validation = colmap.validate_binary_model(output_model)
        independent_reload = pycolmap.Reconstruction(str(output_model))
        reload_summary = _reload_summary(independent_reload, pycolmap.__version__)
        invariants = semantic_invariants(
            source_reconstruction, independent_reload, corrections, point_means
        )
        final_post_residuals, final_point_means = actual_reprojection_errors(independent_reload)
        if final_post_residuals != post_residuals:
            raise OutputValidationError("Post-write residual audit differs from pre-write audit")
        for point_id, expected in point_means.items():
            if not math.isclose(final_point_means[point_id], expected, rel_tol=0.0, abs_tol=1e-12):
                raise OutputValidationError("Post-write per-point residual means changed")
        if source_binary_validation["camera_count"] != output_binary_validation["camera_count"]:
            raise OutputValidationError("Binary validator camera count changed")
        if source_binary_validation["registered_image_count"] != output_binary_validation["registered_image_count"]:
            raise OutputValidationError("Binary validator image count changed")
        if source_binary_validation["point_count"] != output_binary_validation["point_count"]:
            raise OutputValidationError("Binary validator point count changed")
        # PyCOLMAP may serialize the same image-ID map in a different binary
        # record order. Semantic invariants above bind every unchanged ID to
        # its name/camera/pose; the validator-level check is therefore a set
        # comparison expressed as sorted unique names, not file order.
        if sorted(source_binary_validation["image_names"]) != sorted(
            output_binary_validation["image_names"]
        ):
            raise OutputValidationError("Binary validator image names changed")
        if reload_summary["image_names"] != sorted(output_binary_validation["image_names"]):
            raise OutputValidationError("Independent PyCOLMAP reload image names disagree")

        source_ply_record = absolute_file_record(source_ply)
        output_ply_record = absolute_file_record(
            output_ply, declared_path=destination / "point_cloud.ply"
        )
        if (source_ply_record["bytes"], source_ply_record["sha256"]) != (
            output_ply_record["bytes"], output_ply_record["sha256"]
        ):
            raise OutputValidationError("Derived PLY is not byte-identical to source PLY")

        model_files = [output_model / name for name in REQUIRED_COLMAP_FILES]
        model_files += [output_model / name for name in OPTIONAL_COLMAP_FILES
                        if (output_model / name).is_file()]
        artifact_paths = [*model_files, output_ply, crop_path]
        output_records = [
            absolute_file_record(path, declared_path=destination / path.relative_to(staging))
            for path in artifact_paths
        ]
        output_roles = {
            "cameras_bin": str(destination / "colmap" / "sparse" / "0" / "cameras.bin"),
            "images_bin": str(destination / "colmap" / "sparse" / "0" / "images.bin"),
            "points3D_bin": str(destination / "colmap" / "sparse" / "0" / "points3D.bin"),
            "point_cloud_ply": str(destination / "point_cloud.ply"),
            "crop_parameters": str(destination / "crop_parameters.json"),
            "review_export_receipt": str(destination / "review_export_receipt.json"),
        }

        # Everything used as provenance remains frozen through the whole export.
        frozen_after = records(frozen_paths)
        if _record_map(frozen_before) != _record_map(frozen_after):
            raise OutputValidationError("Source/evidence changed during export")
        source_after = records(published_paths)
        if _record_map(source_before) != _record_map(source_after):
            raise OutputValidationError("Original published output changed during export")

        audit = {
            "schema_version": 1,
            "status": "complete",
            "method": "derived_vggsfm_colmap_coordinate_review",
            "scope": (
                "CPU-only derived interoperability export from an already complete published "
                "VGGSfM result; no neural inference and no geometry reconstruction rerun."
            ),
            "source_run_id": run_id,
            "derived_run_id": derived_run_id,
            "run_id": derived_run_id,
            "created_at": utc_now(),
            "inference_rerun": False,
            "source_manifest_status": manifest["status"],
            "source_attempt_status": receipt["status"],
            "source_request_fingerprint": fingerprint,
            "source_manifest": absolute_file_record(source_manifest_path),
            "source_attempt_receipt": absolute_file_record(receipt_path),
            "source_official_log": absolute_file_record(log_path),
            "effective_setting": {
                **setting,
                "official_runner": source_code["runner"],
                "official_runner_lines": source_code["runner_lines"],
            },
            "official_source": source_code,
            "formula": {
                "expression": FORMULA,
                "implementation": "exact pinned float32 crop-parameter and resize-ratio semantics",
                "crop_parameter_source": source_code["crop_parameter_source"],
                "crop_parameter_source_lines": source_code["crop_parameter_lines"],
                "registered_image_correction_coverage": len(correction_rows),
                "registered_image_names": sorted(corrections),
                "crop_parameter_rows_sha256": crop_digest,
                "crop_parameters_file": output_roles["crop_parameters"],
            },
            "residuals": {
                "definition": "Euclidean pixels between saved linked point2D and camera projection of its tracked point3D",
                "pre": pre_residuals,
                "post": final_post_residuals,
                "tracked_observation_count": final_post_residuals["tracked_observation_count"],
                "tracked_point_count": final_post_residuals["tracked_point_count"],
                "trackless_point_count": final_post_residuals["trackless_point_count"],
                "post_mean_improved": final_post_residuals["mean_px"] < pre_residuals["mean_px"],
                "post_rmse_improved": final_post_residuals["rmse_px"] < pre_residuals["rmse_px"],
                "post_all_finite_nonnegative": final_post_residuals["all_finite_nonnegative"],
            },
            "coordinate_correction": correction_metrics,
            "stored_error_update": error_metrics,
            "invariants": invariants,
            "source_files_before": source_before,
            "source_files_after": source_after,
            "source_files_before_after_equal": True,
            "all_frozen_evidence_before_after_equal": True,
            "frozen_evidence": frozen_after,
            "validation": {
                "validate_binary_model_passed": True,
                "source_binary_model": source_binary_validation,
                "derived_binary_model": output_binary_validation,
                "binary_validator_reciprocal_tracks_passed": True,
                "all_linked_point2d_finite": correction_metrics["all_linked_finite"],
                "all_linked_point2d_within_camera_bounds": correction_metrics[
                    "all_linked_within_camera_bounds"
                ],
                "linked_point2d_outside_camera_bounds_count": correction_metrics[
                    "linked_outside_camera_bounds_count"
                ],
                "out_of_bounds_policy": "reported, not modified; exact formula and tracks preserved",
                "independent_pycolmap_reload_passed": True,
                "independent_pycolmap_reload": reload_summary,
                "counts_and_names_match_source": True,
            },
            "ply_geometry": {
                "byte_identical_to_source": True,
                "source": source_ply_record,
                "derived": output_ply_record,
                "meaning": "XYZ order/values and RGB are unchanged; only COLMAP 2D/error metadata changed.",
            },
            "output_roles": output_roles,
            "output_files": output_records,
            "execution": {
                "python": sys.executable,
                "pycolmap_version": pycolmap.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            },
        }
        manifest_out = {
            **audit,
            "manifest_kind": "derived_review_export",
            "files": output_records,
            "source_point_cloud": manifest.get("point_cloud"),
            "camera_count": output_binary_validation["camera_count"],
            "registered_image_count": output_binary_validation["registered_image_count"],
            "point_count": output_binary_validation["point_count"],
        }
        manifest_path = staging / "manifest.json"
        atomic_write_json(manifest_path, manifest_out)
        audit["output_manifest"] = absolute_file_record(
            manifest_path, declared_path=destination / "manifest.json"
        )
        receipt_out = staging / "review_export_receipt.json"
        atomic_write_json(receipt_out, audit)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    final_manifest = read_json(destination / "manifest.json")
    if final_manifest.get("status") != "complete":
        raise OutputValidationError("Derived manifest did not survive atomic publication")
    for item in final_manifest["files"]:
        current = absolute_file_record(Path(item["path"]))
        if current != item:
            raise OutputValidationError(f"Derived output checksum mismatch: {item['path']}")
    return read_json(destination / "review_export_receipt.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed original published run ID")
    parser.add_argument("--expected-pycolmap", default=EXPECTED_PYCOLMAP)
    args = parser.parse_args()
    result = derive(args.run_id, expected_pycolmap=args.expected_pycolmap)
    print(json.dumps({
        "status": result["status"],
        "source_run_id": result["source_run_id"],
        "derived_run_id": result["derived_run_id"],
        "inference_rerun": result["inference_rerun"],
        "output_manifest": result["output_manifest"],
        "residuals": result["residuals"],
        "invariants": result["invariants"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
