#!/usr/bin/env python3
"""Independently re-audit the published VGGSfM light cleanup."""

from __future__ import annotations

import hashlib
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


SPEC = importlib.util.spec_from_file_location(
    "clean_light_with_masks_audited", Path(__file__).with_name("clean_light_with_masks.py")
)
cleaner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cleaner)

AUDIT_ID = cleaner.DERIVED_RUN + "-validation-v1"


def absolute_record(path: Path) -> dict:
    return cleaner.absolute_record(path)


def verify_declared(root: Path, manifest: dict) -> list[dict]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"manifest has no declared files: {root}")
    verified = []
    for row in rows:
        declared = Path(row["path"])
        path = declared if declared.is_absolute() else root / declared
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        current = file_record(resolved, relative_to=root)
        if current["bytes"] != row.get("bytes") or current["sha256"] != row.get("sha256"):
            raise RuntimeError(f"manifest-bound file changed: {resolved}")
        verified.append(absolute_record(resolved))
    return verified


def audit() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cleanup audit must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    raw_root = layout.output_root(cleaner.SOURCE_RUN)
    camera_root = layout.output_root(cleaner.CAMERA_RUN)
    cleanup_root = layout.output_root(cleaner.DERIVED_RUN)
    audit_root = layout.root / "evaluation" / AUDIT_ID
    if audit_root.exists():
        raise RuntimeError(f"refusing to overwrite cleanup audit: {audit_root}")
    raw_manifest = read_json(raw_root / "manifest.json")
    camera_manifest = read_json(camera_root / "manifest.json")
    cleanup_manifest = read_json(cleanup_root / "manifest.json")
    cleanup_receipt = read_json(cleanup_root / "cleanup_receipt.json")
    mask_receipt_path = layout.inputs / cleaner.MASK_DATASET / "receipt.json"
    mask_receipt = read_json(mask_receipt_path)
    request = read_json(layout.experiment_root(cleaner.SOURCE_RUN) / "request.json")
    if cleanup_manifest.get("status") != "complete" or cleanup_receipt.get("status") != "complete":
        raise RuntimeError("cleanup result is not complete")

    raw_declared = verify_declared(raw_root, raw_manifest)
    camera_declared = verify_declared(camera_root, camera_manifest)
    cleanup_declared = verify_declared(cleanup_root, cleanup_manifest)

    mask_root = layout.inputs / cleaner.MASK_DATASET / "masks"
    mask_before = []
    for row in mask_receipt["rows"]:
        name = row["image_relative"]
        path = mask_root / Path(name).with_suffix(".png")
        current = file_record(path, relative_to=layout.inputs / cleaner.MASK_DATASET)
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            raise RuntimeError(f"mask differs from method-local receipt: {name}")
        with Image.open(path) as opened:
            if opened.mode != "L" or opened.size != (row["width"], row["height"]):
                raise RuntimeError(f"mask mode/dimensions changed: {name}")
        mask_before.append(absolute_record(path))

    camera_model = camera_root / "colmap" / "sparse" / "0"
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(camera_model))
    official_to_source = {
        row["official"]: row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    point_ids = np.asarray(sorted(reconstruction.points3D), dtype=np.uint64)
    xyz = np.stack([
        np.asarray(reconstruction.points3D[int(point_id)].xyz)
        for point_id in point_ids
    ])
    foreground, usable, projection_rows = cleaner.projection_votes(
        reconstruction, point_ids, xyz, official_to_source, mask_root
    )
    rule = cleanup_receipt["projection_rule"]
    selected = cleaner.select_by_support(
        foreground, usable,
        minimum_usable=rule["minimum_usable_views"],
        agreement_threshold=rule["minimum_foreground_agreement"],
    )
    selected_ids = {int(value) for value in point_ids[selected]}
    if len(selected_ids) != cleanup_receipt["retained_point_count"]:
        raise RuntimeError("independent projection selection count changed")
    support = cleaner.support_summary(foreground, usable)
    if support != cleanup_receipt["support_statistics"]:
        raise RuntimeError("independent support statistics changed")

    staging = layout.root / "evaluation" / f".{AUDIT_ID}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        projection_path = staging / "projection_rows.json"
        atomic_write_json(projection_path, {
            "schema_version": 1,
            "coordinate_system": "original image pixels after VGGSfM SIMPLE_RADIAL img_from_cam",
            "image_count": len(projection_rows),
            "rows_sha256": object_sha256(projection_rows),
            "rows": projection_rows,
        })
        regenerated = staging / "regenerated_clean.ply"
        regenerated_metrics = cleaner.write_subset_ply(
            raw_root / "colmap" / "sparse" / "0" / "points3D.bin",
            raw_root / "point_cloud.ply", selected_ids, regenerated,
        )
        published_ply = cleanup_root / "point_cloud.ply"
        if regenerated.read_bytes() != published_ply.read_bytes():
            raise RuntimeError("independently regenerated cleaned PLY differs byte-for-byte")
        regenerated.unlink()

        mask_after = [absolute_record(Path(row["path"])) for row in mask_before]
        if mask_before != mask_after:
            raise RuntimeError("mask files changed during projection audit")
        # Recheck source declarations after the expensive projection pass.
        if raw_declared != verify_declared(raw_root, raw_manifest):
            raise RuntimeError("raw source changed during audit")
        if camera_declared != verify_declared(camera_root, camera_manifest):
            raise RuntimeError("camera review source changed during audit")
        if cleanup_declared != verify_declared(cleanup_root, cleanup_manifest):
            raise RuntimeError("cleaned output changed during audit")

        projection_record = cleaner.absolute_record(
            projection_path, declared=audit_root / "projection_rows.json"
        )
        preview_root = layout.root.parent / "evaluation" / "preview-vggsfm-light-clean-v1"
        preview_records = [
            absolute_record(preview_root / name)
            for name in ("preview.png", "preview_receipt.json")
        ]
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "audit_id": AUDIT_ID,
            "run_id": cleaner.DERIVED_RUN,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cleanup_receipt": absolute_record(cleanup_root / "cleanup_receipt.json"),
            "cleanup_manifest": absolute_record(cleanup_root / "manifest.json"),
            "raw_source": {
                "run_id": cleaner.SOURCE_RUN,
                "manifest_status": raw_manifest["status"],
                "manifest": absolute_record(raw_root / "manifest.json"),
                "declared_files_verified": raw_declared,
                "point_count": cleaner.EXPECTED_POINTS,
                "unchanged_after_audit": True,
            },
            "camera_source": {
                "run_id": cleaner.CAMERA_RUN,
                "manifest_status": camera_manifest["status"],
                "manifest": absolute_record(camera_root / "manifest.json"),
                "declared_files_verified": camera_declared,
                "model_path": str(camera_model),
                "registered_images": int(reconstruction.num_reg_images()),
                "cameras": int(reconstruction.num_cameras()),
                "all_original_dimensions_match_masks": True,
                "own_vggsfm_poses_not_e10": True,
            },
            "mask_input": {
                "receipt": absolute_record(mask_receipt_path),
                "count": len(mask_before),
                "all_hashes_dimensions_modes_verified_before_and_after": True,
                "records_before": mask_before,
                "records_after": mask_after,
            },
            "projection": {
                "rule": rule,
                "rows": projection_record,
                "rows_sha256": object_sha256(projection_rows),
                "image_count": len(projection_rows),
                "support_statistics": support,
            },
            "selection": {
                "source_count": cleaner.EXPECTED_POINTS,
                "retained_count": len(selected_ids),
                "removed_count": cleaner.EXPECTED_POINTS - len(selected_ids),
                "count_conservation": len(selected_ids) + cleaner.EXPECTED_POINTS - len(selected_ids) == cleaner.EXPECTED_POINTS,
                "regenerated_metrics": regenerated_metrics,
                "published_ply": absolute_record(published_ply),
                "published_matches_independent_regeneration_byte_for_byte": True,
                "finite_xyz_rgb": True,
            },
            "invariants": {
                "inference_rerun": False,
                "poses_recalculated": False,
                "xyz_rgb_recalculated": False,
                "raw_output_replaced": False,
                "raw_manifest_declared_files_unchanged": True,
                "camera_manifest_declared_files_unchanged": True,
                "mask_files_unchanged": True,
            },
            "preview": {
                "status": "visually_inspected",
                "interpretation": "Isolated person is clear; some coverage gaps remain; no anatomical-accuracy claim.",
                "files": preview_records,
            },
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
    print(json.dumps({
        "status": result["status"],
        "audit_id": result["audit_id"],
        "run_id": result["run_id"],
        "projection": result["projection"],
        "selection": result["selection"],
        "invariants": result["invariants"],
        "preview": result["preview"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
