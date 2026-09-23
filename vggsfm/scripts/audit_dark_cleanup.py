#!/usr/bin/env python3
"""Independently regenerate and audit the crutch-aware dark VGGSfM cleanup."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256, read_json
from vggsfm_engine.paths import production_layout
from vggsfm_engine.projection_cleanup import (
    corridor_hits,
    project_original_pixels,
    separated_view_confirmation,
)


SPEC = importlib.util.spec_from_file_location(
    "clean_dark_with_masks_and_crutches_audited",
    Path(__file__).with_name("clean_dark_with_masks_and_crutches.py"),
)
cleaner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cleaner)
AUDIT_ID = cleaner.DERIVED_RUN + "-validation-v1"


def absolute_record(path: Path) -> dict:
    return cleaner.absolute_record(path)


def verify_declared(root: Path, manifest: dict) -> list[dict]:
    verified = []
    for row in manifest.get("files", []):
        declared = Path(row["path"])
        path = declared if declared.is_absolute() else root / declared
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        current = file_record(resolved)
        if current["bytes"] != row.get("bytes") or current["sha256"] != row.get("sha256"):
            raise RuntimeError(f"manifest-bound file changed: {resolved}")
        verified.append(absolute_record(resolved))
    if not verified:
        raise RuntimeError(f"manifest has no declared files: {root}")
    return verified


def audit() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cleanup audit must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    raw_root = layout.output_root(cleaner.SOURCE_RUN)
    cleanup_root = layout.output_root(cleaner.DERIVED_RUN)
    input_root = layout.inputs / cleaner.INPUT_DATASET
    audit_root = layout.root / "evaluation" / AUDIT_ID
    if audit_root.exists():
        raise RuntimeError(f"refusing to overwrite {audit_root}")
    raw_manifest = read_json(raw_root / "manifest.json")
    cleanup_manifest = read_json(cleanup_root / "manifest.json")
    cleanup_receipt = read_json(cleanup_root / "cleanup_receipt.json")
    input_receipt_path = input_root / "receipt.json"
    input_receipt = read_json(input_receipt_path)
    pixel_identity_path = input_root / "pixel_identity_audit.json"
    pixel_identity = read_json(pixel_identity_path)
    corridor_path = input_root / "crutches/corridors.json"
    corridors = read_json(corridor_path)
    request_path = layout.experiment_root(cleaner.SOURCE_RUN) / "request.json"
    request = read_json(request_path)
    if (
        raw_manifest.get("status") != "complete"
        or cleanup_manifest.get("status") != "complete"
        or cleanup_receipt.get("status") != "complete"
        or pixel_identity.get("status") != "complete"
    ):
        raise RuntimeError("raw/cleanup/input identity state is not complete")

    raw_declared = verify_declared(raw_root, raw_manifest)
    cleanup_declared = verify_declared(cleanup_root, cleanup_manifest)
    source_code_paths = [
        Path(__file__).with_name("clean_dark_with_masks_and_crutches.py"),
        CODE_ROOT / "vggsfm_engine/projection_cleanup.py",
        Path(__file__).with_name("prepare_dark_cleanup_inputs.py"),
        Path(__file__).with_name("audit_dark_cleanup_pixel_identity.py"),
        CODE_ROOT / "tests/test_projection_cleanup.py",
    ]
    source_code = [absolute_record(path) for path in source_code_paths]

    mask_root = input_root / "masks"
    mask_rows = {row["image_relative"]: row for row in input_receipt["masks"]}
    mask_before = []
    for name, row in sorted(mask_rows.items()):
        path = mask_root / Path(row["mask_relative"]).relative_to("masks")
        current = file_record(path)
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            raise RuntimeError(f"mask differs from receipt: {name}")
        with Image.open(path) as opened:
            if opened.mode != "L" or opened.size != (row["width"], row["height"]):
                raise RuntimeError(f"mask metadata differs from receipt: {name}")
        mask_before.append(absolute_record(path))
    if len(mask_before) != cleaner.EXPECTED_IMAGES:
        raise RuntimeError("mask receipt does not cover 290 images")

    raw_model = raw_root / "colmap/sparse/0"
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(raw_model))
    official_to_source = {
        row["official"]: row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    point_ids = np.asarray(sorted(reconstruction.points3D), dtype=np.uint64)
    xyz = np.stack([
        np.asarray(reconstruction.points3D[int(point_id)].xyz, dtype=np.float64)
        for point_id in point_ids
    ])
    foreground = np.zeros(cleaner.EXPECTED_POINTS, dtype=np.uint16)
    usable = np.zeros(cleaner.EXPECTED_POINTS, dtype=np.uint16)
    annotations = corridors["annotated_image_corridors"]
    corridor_hits_by_view = []
    corridor_centers = []
    projection_rows = []
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        source_name = official_to_source[image.name]
        camera = reconstruction.cameras[image.camera_id]
        row = mask_rows[source_name]
        mask_path = mask_root / Path(row["mask_relative"]).relative_to("masks")
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened, dtype=np.uint8)
        if mask.shape != (int(camera.height), int(camera.width)):
            raise RuntimeError(f"mask/camera dimensions disagree: {source_name}")
        pixels, continuous = project_original_pixels(image, camera, xyz)
        indices = np.flatnonzero(continuous)
        sampled = np.rint(pixels[indices]).astype(np.int64)
        nearest = (
            (sampled[:, 0] >= 0) & (sampled[:, 0] < int(camera.width))
            & (sampled[:, 1] >= 0) & (sampled[:, 1] < int(camera.height))
        )
        indices = indices[nearest]
        sampled = sampled[nearest]
        usable[indices] += 1
        foreground[indices] += (
            mask[sampled[:, 1], sampled[:, 0]] >= cleaner.MASK_THRESHOLD
        ).astype(np.uint16)
        hit_count = 0
        if source_name in annotations:
            hits = corridor_hits(
                pixels, continuous, width=int(camera.width), height=int(camera.height),
                groups=annotations[source_name],
            )
            corridor_hits_by_view.append(hits)
            corridor_centers.append(np.asarray(image.projection_center(), dtype=np.float64))
            hit_count = int(hits.sum())
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
            "crutch_corridor_hit_count": hit_count,
        })

    body_rule = cleanup_receipt["body_rule"]
    crutch_rule = cleanup_receipt["crutch_rule"]
    body = cleaner.select_by_support(
        foreground, usable,
        minimum_usable=body_rule["minimum_usable_views"],
        agreement_threshold=body_rule["minimum_foreground_agreement"],
    )
    crutch = separated_view_confirmation(
        xyz, np.stack(corridor_centers), np.stack(corridor_hits_by_view),
        minimum_views=crutch_rule["minimum_distinct_annotated_views"],
        minimum_pairwise_angle_degrees=(
            crutch_rule["minimum_pairwise_camera_center_to_point_ray_separation_degrees"]
        ),
    )
    selected = body | crutch
    selected_ids = {int(value) for value in point_ids[selected]}
    selection = {
        "source_count": cleaner.EXPECTED_POINTS,
        "body_selected_count": int(body.sum()),
        "crutch_selected_count": int(crutch.sum()),
        "body_crutch_overlap_count": int((body & crutch).sum()),
        "crutch_only_protected_count": int((crutch & ~body).sum()),
        "retained_count": int(selected.sum()),
        "removed_count": int((~selected).sum()),
        "count_conservation": int(selected.sum()) + int((~selected).sum())
        == cleaner.EXPECTED_POINTS,
    }
    if selection != cleanup_receipt["selection"]:
        raise RuntimeError("independent body/crutch selection statistics differ")
    support = cleaner.support_summary(foreground, usable)
    if support != cleanup_receipt["support_statistics"]:
        raise RuntimeError("independent body support statistics differ")
    if [int(row.sum()) for row in corridor_hits_by_view] != cleanup_receipt[
        "corridor_per_view_hit_counts"
    ]:
        raise RuntimeError("independent corridor hit counts differ")

    published_rows = read_json(cleanup_root / "projection_rows.json")
    if (
        projection_rows != published_rows.get("rows")
        or object_sha256(projection_rows) != published_rows.get("rows_sha256")
    ):
        raise RuntimeError("independent per-image projection rows differ")

    staging = layout.root / "evaluation" / f".{AUDIT_ID}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        regenerated = staging / "regenerated_clean.ply"
        regenerated_metrics = cleaner.write_subset_ply(
            raw_model / "points3D.bin", raw_root / "point_cloud.ply",
            selected_ids, regenerated,
        )
        published_ply = cleanup_root / "point_cloud.ply"
        if regenerated.read_bytes() != published_ply.read_bytes():
            raise RuntimeError("independently regenerated PLY differs byte-for-byte")
        regenerated.unlink()

        mask_after = [absolute_record(Path(row["path"])) for row in mask_before]
        if mask_before != mask_after:
            raise RuntimeError("one or more masks changed during audit")
        if raw_declared != verify_declared(raw_root, raw_manifest):
            raise RuntimeError("raw manifest files changed during audit")
        if cleanup_declared != verify_declared(cleanup_root, cleanup_manifest):
            raise RuntimeError("cleanup manifest files changed during audit")
        if source_code != [absolute_record(path) for path in source_code_paths]:
            raise RuntimeError("cleanup source code changed during audit")

        receipt = {
            "schema_version": 1,
            "status": "complete",
            "audit_id": AUDIT_ID,
            "run_id": cleaner.DERIVED_RUN,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_source": {
                "run_id": cleaner.SOURCE_RUN,
                "manifest": absolute_record(raw_root / "manifest.json"),
                "declared_files_verified_before_after": raw_declared,
                "point_count": cleaner.EXPECTED_POINTS,
                "unchanged": True,
            },
            "cleanup": {
                "manifest": absolute_record(cleanup_root / "manifest.json"),
                "receipt": absolute_record(cleanup_root / "cleanup_receipt.json"),
                "declared_files_verified_before_after": cleanup_declared,
            },
            "source_code": source_code,
            "pixel_identity": {
                "audit": absolute_record(pixel_identity_path),
                "rows_sha256": pixel_identity["rows_sha256"],
                "exact_actual_image_hashes_290": True,
                "mask_native_dimensions_and_hashes_290": True,
            },
            "mask_input": {
                "receipt": absolute_record(input_receipt_path),
                "count": len(mask_before),
                "records_before": mask_before,
                "records_after": mask_after,
                "all_hashes_dimensions_modes_verified_before_after": True,
            },
            "camera_source": {
                "run_id": cleaner.SOURCE_RUN,
                "model_path": str(raw_model),
                "registered_images": int(reconstruction.num_reg_images()),
                "cameras": int(reconstruction.num_cameras()),
                "own_dark_vggsfm_poses_not_e10_or_mvs": True,
                "original_dimensions_and_simple_radial_projection": True,
            },
            "projection": {
                "body_rule": body_rule,
                "crutch_rule": crutch_rule,
                "union_rule": cleanup_receipt["union_rule"],
                "rows": absolute_record(cleanup_root / "projection_rows.json"),
                "rows_sha256": object_sha256(projection_rows),
                "image_count": len(projection_rows),
                "corridor_image_count": len(corridor_hits_by_view),
                "support_statistics": support,
                "corridor_per_view_hit_counts": [
                    int(row.sum()) for row in corridor_hits_by_view
                ],
                "occlusion_test": False,
            },
            "selection": {
                **selection,
                "regenerated_metrics": regenerated_metrics,
                "published_ply": absolute_record(published_ply),
                "published_matches_independent_regeneration_byte_for_byte": True,
                "finite_xyz_rgb": True,
            },
            "invariants": {
                "inference_rerun": False,
                "triangulation_rerun": False,
                "poses_recalculated": False,
                "xyz_rgb_recalculated": False,
                "raw_output_replaced": False,
                "uses_e10_or_mvs_geometry": False,
                "uses_world_capsules": False,
                "raw_files_unchanged": True,
                "mask_files_unchanged": True,
                "source_code_unchanged": True,
            },
            "limitations": cleanup_receipt["limitations"],
            "execution": {
                "python": sys.executable,
                "pycolmap_version": pycolmap.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            },
        }
        atomic_write_json(staging / "audit_receipt.json", receipt)
        os.replace(staging, audit_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return read_json(audit_root / "audit_receipt.json")


def main() -> int:
    result = audit()
    print(json.dumps({key: result[key] for key in (
        "status", "audit_id", "run_id", "pixel_identity", "projection",
        "selection", "invariants",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
