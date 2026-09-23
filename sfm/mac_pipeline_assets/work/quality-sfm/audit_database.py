#!/usr/bin/env python3
"""Audit a quality SfM database and its completed feature checkpoint read-only.

Usage: audit_database.py BASELINE CANDIDATE --output JSON
The CLI writes only the requested report. SQLite connections use mode=ro.
"""

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_IDENTITY_TABLES = ("images", "frames", "frame_data", "rigs", "rig_sensors", "pose_priors")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_stats(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _table_rows(database, table):
    # Table names are internal constants, never values from the recipe.
    column_count = len(database.execute(f"PRAGMA table_info({table})").fetchall())
    order = ",".join(str(number) for number in range(1, column_count + 1))
    return database.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()


def _all_checks_pass(checks):
    return all(_all_checks_pass(value) if isinstance(value, dict) else bool(value)
               for value in checks.values())


def _select_checkpoint(candidate, config, saved_inputs):
    """Ignore obsolete checkpoints; select the one fingerprinted by the result."""
    matching_hash = _sha256((candidate / config["matching_pairs"]).resolve())
    guided_hash = _sha256((candidate / config["guided_pairs"]).resolve())
    selected, ignored = [], []
    for metadata_path in sorted((candidate / "colmap/quality_feature_cache").glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
            signature = metadata["signature"]
            key = _json_hash(signature)
            recipe = {
                "features": key, "config": config,
                "matching_pairs_sha256": matching_hash,
                "guided_pairs_sha256": guided_hash,
                "triangulation": {
                    "fixed_poses": True, "refine_intrinsics": False,
                    "max_reprojection_error": 3.5, "min_angle": 1.5,
                },
            }
            if _json_hash(recipe) == saved_inputs.get("quality_refinement"):
                selected.append((metadata_path, metadata, key))
            else:
                ignored.append(metadata_path.name)
        except (OSError, ValueError, TypeError, KeyError):
            ignored.append(metadata_path.name)
    if len(selected) != 1:
        raise ValueError(
            f"Expected one feature checkpoint for the saved quality recipe; found {len(selected)}. "
            "The recipe, pair files, or saved result metadata may have changed."
        )
    return (*selected[0], ignored)


def _audit(baseline, candidate, report, watched):
    checks = report["checks"]
    config_path = candidate / "sfm_refine.json"
    inputs_path = candidate / "colmap/sparse/sfm_inputs.json"
    config = json.loads(config_path.read_text())
    saved_inputs = json.loads(inputs_path.read_text())
    cap = config.get("max_features", 18000)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("max_features must be a positive integer")
    checks["configured_baseline_matches_requested_baseline"] = (
        (candidate / config["baseline_dataset"]).resolve() == baseline
    )
    metadata_path, metadata, feature_key, ignored = _select_checkpoint(
        candidate, config, saved_inputs
    )
    checkpoint_path = metadata_path.with_suffix(".db")
    canonical_path = candidate / "colmap/database.db"
    baseline_path = baseline / "colmap/database.db"
    boxes_path = (candidate / config["boxes"]).resolve()
    image_dir = candidate / "images"
    image_paths = sorted(path for path in image_dir.rglob("*")
                         if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
    baseline_files = [baseline_path] + sorted(
        path for path in (baseline / "colmap/sparse").rglob("*") if path.is_file()
    )
    for path in [config_path, inputs_path, metadata_path, checkpoint_path, canonical_path,
                 boxes_path, (candidate / config["matching_pairs"]).resolve(),
                 (candidate / config["guided_pairs"]).resolve(), *image_paths, *baseline_files]:
        watched[path] = _file_stats(path)

    signature = metadata["signature"]
    checks["checkpoint_selected_by_saved_recipe"] = True
    checks["checkpoint_filename_matches_signature_sha256"] = checkpoint_path.stem == feature_key
    checks["checkpoint_file_size_matches_metadata"] = checkpoint_path.stat().st_size == metadata.get("database_size")
    checks["boxes_content_hash_matches_signature"] = _sha256(boxes_path) == signature.get("boxes_sha256")
    checks["signature_image_file_stats_match_current"] = signature.get("images") == [
        [path.relative_to(image_dir).as_posix(), *_file_stats(path)] for path in image_paths
    ]
    checks["signature_baseline_file_stats_match_current"] = signature.get("baseline") == [
        [str(path), *_file_stats(path)] for path in baseline_files
    ]
    checks["signature_feature_cap_matches_recipe"] = signature.get("max_features") == cap
    checks["saved_result_images_match_checkpoint_signature"] = saved_inputs.get("images") == signature.get("images")

    with ExitStack() as stack:
        databases = {
            name: stack.enter_context(closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)))
            for name, path in (("canonical", canonical_path), ("baseline", baseline_path),
                               ("checkpoint", checkpoint_path))
        }
        current, original, checkpoint = (databases[name] for name in ("canonical", "baseline", "checkpoint"))
        checks["sqlite_quick_check"] = {
            name: database.execute("PRAGMA quick_check").fetchall() == [("ok",)]
            for name, database in databases.items()
        }
        checks["baseline_identity_tables_unchanged"] = {
            table: _table_rows(current, table) == _table_rows(original, table)
            for table in _IDENTITY_TABLES
        }
        checks["checkpoint_identity_tables_equal_canonical"] = {
            table: _table_rows(checkpoint, table) == _table_rows(current, table)
            for table in _IDENTITY_TABLES
        }
        checks["baseline_camera_ids_and_models_preserved"] = (
            current.execute("SELECT camera_id,model FROM cameras ORDER BY camera_id").fetchall()
            == original.execute("SELECT camera_id,model FROM cameras ORDER BY camera_id").fetchall()
        )
        checks["checkpoint_calibration_equals_canonical"] = _table_rows(checkpoint, "cameras") == _table_rows(current, "cameras")
        report["identity_counts"] = {
            table: current.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("images", "cameras", "frames", "frame_data", "rigs", "rig_sensors")
        }
        cameras = {row[0]: row[1:] for row in current.execute(
            "SELECT camera_id,width,height,params,prior_focal_length FROM cameras"
        )}
        images = {row[0]: row[1:] for row in current.execute("SELECT image_id,name,camera_id FROM images")}
        native_mismatches = []
        for image_id, (name, camera_id) in images.items():
            with Image.open(image_dir / name) as image:
                size = image.size
            if size != cameras[camera_id][:2]:
                native_mismatches.append({"image_id": image_id, "name": name,
                                          "native": size, "database": cameras[camera_id][:2]})
        checks["all_camera_dimensions_match_native_images"] = not native_mismatches
        checks["all_camera_parameters_finite"] = all(
            camera[2] is not None and len(camera[2]) > 0
            and len(camera[2]) % 8 == 0 and np.isfinite(np.frombuffer(camera[2], dtype=np.float64)).all()
            for camera in cameras.values()
        )
        checks["all_cameras_have_prior_focal_length"] = all(camera[3] == 1 for camera in cameras.values())
        report["camera_dimensions"] = [{"camera_id": camera_id, "width": camera[0], "height": camera[1]}
                                       for camera_id, camera in sorted(cameras.items())]
        report["native_dimension_mismatches"] = native_mismatches
        keypoint_ids = {row[0] for row in current.execute("SELECT image_id FROM keypoints")}
        descriptor_ids = {row[0] for row in current.execute("SELECT image_id FROM descriptors")}
        checks["feature_tables_cover_each_image_exactly"] = keypoint_ids == descriptor_ids == set(images)
        checks["checkpoint_feature_tables_cover_each_image_exactly"] = (
            keypoint_ids == {row[0] for row in checkpoint.execute("SELECT image_id FROM keypoints")}
            == {row[0] for row in checkpoint.execute("SELECT image_id FROM descriptors")}
        )
        feature_counts = []
        totals = {"keypoint_payload_bytes": 0, "descriptor_payload_bytes": 0,
                  "invalid_keypoint_rows": 0, "coordinates_outside_image": 0,
                  "checkpoint_mismatched_feature_records": 0}
        all_dimensions = all_payloads = all_alignment = True
        for image_id, keypoint_rows, keypoint_cols, keypoint_blob in current.execute(
            "SELECT image_id,rows,cols,data FROM keypoints ORDER BY image_id"
        ):
            descriptor = current.execute("SELECT type,rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
            if descriptor is None or image_id not in images:
                report["violations"].append({"image_id": image_id, "error": "missing image or descriptor record"})
                all_alignment = False
                continue
            descriptor_type, descriptor_rows, descriptor_cols, descriptor_blob = descriptor
            keypoint_data, descriptor_data = keypoint_blob or b"", descriptor_blob or b""
            dimensions_ok = keypoint_rows >= 0 and descriptor_rows >= 0 and keypoint_cols == 6 and descriptor_cols == 128
            all_dimensions &= dimensions_ok
            payload_ok = (len(keypoint_data) == keypoint_rows * keypoint_cols * 4
                          and len(descriptor_data) == descriptor_rows * descriptor_cols)
            all_payloads &= payload_ok
            all_alignment &= keypoint_rows == descriptor_rows
            feature_counts.append(keypoint_rows)
            totals["keypoint_payload_bytes"] += len(keypoint_data)
            totals["descriptor_payload_bytes"] += len(descriptor_data)
            if not (dimensions_ok and payload_ok):
                report["violations"].append({"image_id": image_id, "error": "invalid feature dimensions or payload size"})
                continue
            matrix = np.frombuffer(keypoint_data, dtype=np.float32).reshape(keypoint_rows, 6)
            width, height, *_ = cameras[images[image_id][1]]
            finite = np.isfinite(matrix).all(axis=1)
            in_bounds = ((matrix[:, 0] >= 0) & (matrix[:, 0] < width)
                         & (matrix[:, 1] >= 0) & (matrix[:, 1] < height))
            totals["invalid_keypoint_rows"] += int((~finite).sum())
            totals["coordinates_outside_image"] += int((~in_bounds).sum())
            saved_keypoints = checkpoint.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
            saved_descriptors = checkpoint.execute("SELECT type,rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
            if (saved_keypoints != (keypoint_rows, keypoint_cols, keypoint_blob)
                    or saved_descriptors != (descriptor_type, descriptor_rows, descriptor_cols, descriptor_blob)):
                totals["checkpoint_mismatched_feature_records"] += 1
        checks.update({
            "keypoints_are_six_columns_descriptors_are_128_columns": bool(all_dimensions),
            "keypoint_descriptor_row_counts_aligned": bool(all_alignment),
            "feature_payload_sizes_correct": bool(all_payloads),
            "all_keypoints_finite": totals["invalid_keypoint_rows"] == 0,
            "all_keypoints_within_native_image_bounds": totals["coordinates_outside_image"] == 0,
            "feature_cap_respected": bool(feature_counts) and 0 <= min(feature_counts) <= max(feature_counts) <= cap,
            "checkpoint_feature_rows_byte_identical_to_canonical": totals["checkpoint_mismatched_feature_records"] == 0,
        })
        report["features"] = {
            "images": len(feature_counts), "total": sum(feature_counts),
            "minimum_per_image": min(feature_counts, default=0),
            "maximum_per_image": max(feature_counts, default=0),
            "median_per_image": float(np.median(feature_counts)) if feature_counts else None,
            "feature_cap": cap, "images_at_cap": sum(number == cap for number in feature_counts), **totals,
        }
        report["matching_tables"] = {
            name: {table: {
                "pair_records": database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                "correspondence_rows": database.execute(f"SELECT COALESCE(SUM(rows),0) FROM {table}").fetchone()[0],
            } for table in ("matches", "two_view_geometries")}
            for name, database in (("canonical", current), ("checkpoint", checkpoint))
        }
        checks["checkpoint_has_no_matches_or_geometries"] = all(
            value["pair_records"] == 0 for value in report["matching_tables"]["checkpoint"].values()
        )

    checkpoint_hash = _sha256(checkpoint_path)
    report["checkpoint"] = {
        "database": str(checkpoint_path), "metadata": str(metadata_path),
        "database_bytes": checkpoint_path.stat().st_size, "signature_sha256": feature_key,
        "database_sha256_computed_for_audit": checkpoint_hash,
        "metadata_contains_database_content_hash": "database_sha256" in metadata,
        "ignored_unreferenced_metadata": ignored,
    }
    if "database_sha256" in metadata:
        checks["checkpoint_database_content_hash_matches_metadata"] = checkpoint_hash == metadata["database_sha256"]
    else:
        report["concerns"].append(
            "Checkpoint metadata hashes the feature recipe and records database size, "
            "but does not store a database-content hash. Every feature row was directly "
            "compared with the canonical database; this report records its SHA256."
        )


def audit_database(baseline: Path, candidate: Path) -> dict:
    """Return checks, counts and findings without changing reconstruction files.

    Input/corruption failures are reported in ``violations`` with ``passed=False``.
    Camera rescaling against fitted model parameters belongs to the geometry audit;
    this audit checks camera identities, native dimensions and feature coordinates.
    """
    baseline, candidate = Path(baseline).resolve(), Path(candidate).resolve()
    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(candidate), "baseline": str(baseline), "sqlite_access": "mode=ro",
        "checks": {}, "concerns": [], "violations": [],
    }
    watched = {}
    try:
        _audit(baseline, candidate, report, watched)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        report["violations"].append({"error_type": type(error).__name__, "error": str(error)})
    changed = []
    for path, before in watched.items():
        try:
            if _file_stats(path) != before:
                changed.append(str(path))
        except OSError:
            changed.append(str(path))
    report["checks"]["source_file_stats_unchanged_during_audit"] = not changed
    if changed:
        report["violations"].append({"error": "Source files changed during audit", "paths": changed})
    report["passed"] = not report["violations"] and _all_checks_pass(report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_database(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve()),
                      "features": report.get("features", {}), "violations": report["violations"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
