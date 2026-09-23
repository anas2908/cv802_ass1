#!/usr/bin/env python3
"""Stage E3 dark inputs and crutch provenance; never run MVS or alter SfM.

The first dense run is deliberately UNMASKED. Body masks omit crutch shafts;
they are copied for a future geometry-aware cleanup, not applied to raw MVS.
Historical protected point IDs and sparse track quality gates cannot be reused
as dense-point selections. Reprojection overlays require a new review.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv802_mvs.config import MVSConfig
from cv802_mvs.io_utils import atomic_json, sha256_file, utc_now
from cv802_mvs.paths import PathPolicy
from cv802_mvs.staging import _verified_copy, stage_inputs
from cv802_mvs.validation import validate_inputs


def main() -> int:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Run preparation inside the existing allocated compute node")
    import numpy as np
    import pycolmap

    policy = PathPolicy.production()
    code = Path(__file__).resolve().parents[1]
    historical = policy.data_root / "sfm/historical/transferred/sfm"
    source = historical / "reconstructions/black_shirt_crutches_quality"
    cleanup = historical / "reconstructions/experiments/black_person_crutches_cleanup"
    config_path = code / "configs/dark_e3_1024_raw_pycolmap.json"
    config = MVSConfig.load(config_path, policy)
    root = policy.experiment(config.experiment)
    if root.exists():
        raise RuntimeError(f"Refusing to alter existing preparation: {root}")
    # E3 supplies the richest verified unfiltered quality correspondence graph;
    # this does not claim that point count is an accuracy/ground-truth metric.
    comparison = []
    for label, relative in (
        ("E3", "black_shirt_crutches_quality"),
        ("E6", "experiments/E6_guided_off/black_shirt_crutches_quality"),
        ("E8", "experiments/E8_vocab_guided/black_shirt_crutches_quality"),
    ):
        report_path = historical / "reconstructions" / relative / "reconstruction_report.json"
        report = json.loads(report_path.read_text())
        comparison.append({"experiment": label, "report": str(report_path),
                           "report_sha256": sha256_file(report_path),
                           **{key: report[key] for key in ("registered_images", "points3D", "mean_track_length", "mean_reprojection_error_px", "mode")}})
    provenance = stage_inputs(
        policy=policy, experiment=config.experiment,
        images_source=source / "images", model_source=source / "colmap/sparse/0",
        masks_source=cleanup / "body_masks", mask_manifest_source=cleanup / "body_mask_manifest.json",
    )
    extras = []
    copied = root / "inputs/provenance/crutches"
    copied.mkdir()
    for original, relative in (
        (cleanup / "body_mask_manifest.json", "body_mask_manifest.original.json"),
        (cleanup / "crutch_protection/protection_approved.json", "protection_approved.original.json"),
        (cleanup / "crutch_protection/projection_qc.png", "historical_projection_qc.png"),
        (source / "reconstruction_report.json", "e3_reconstruction_report.original.json"),
    ):
        extras.append(_verified_copy(original, copied / relative))
    protection = json.loads((copied / "protection_approved.original.json").read_text())
    baseline = historical / "reconstructions/black_shirt_crutches/subject_preview/colmap/sparse/0"
    baseline_copy = copied / "reference_sparse"
    baseline_copy.mkdir()
    rebased_hashes = {}
    for original_mac_path, expected_hash in protection["source_model_sha256"].items():
        name = Path(original_mac_path).name
        original = baseline / name
        if sha256_file(original) != expected_hash:
            raise RuntimeError(f"Historical approved reference model hash changed: {original}")
        record = _verified_copy(original, baseline_copy / name)
        extras.append(record)
        rebased_hashes[str(baseline_copy / name)] = record["sha256"]
    rebased = copy.deepcopy(protection)
    rebased["source_model"] = str(baseline_copy)
    rebased["source_model_sha256"] = rebased_hashes
    rebased["review"]["reviewed_artifact"] = str(copied / "historical_projection_qc.png")
    if sha256_file(copied / "historical_projection_qc.png") != rebased["review"]["reviewed_artifact_sha256"]:
        raise RuntimeError("Historical crutch overlay hash disagrees with approval")
    rebased["relocation"] = {
        "original": str(copied / "protection_approved.original.json"),
        "dense_geometry_review_complete": False,
        "warning": "Historical sparse protection provenance only. Recompute on dense XYZ, validate camera/world frame, and review new overlays. Never transfer point IDs.",
    }
    atomic_json(copied / "protection_approved.rebased.json", rebased)

    model = pycolmap.Reconstruction(str(root / "inputs/sparse"))
    reference = pycolmap.Reconstruction(str(baseline_copy))
    reference_images = {image.name: image for image in reference.images.values()}
    names = {image.name for image in model.images.values()}
    if names != set(reference_images) or len(names) != 290:
        raise RuntimeError("Expected identical 290 dark images in E3 and approved reference")
    max_pose, max_calibration = 0.0, 0.0
    for image in model.images.values():
        before = reference_images[image.name]
        max_pose = max(max_pose, float(np.max(np.abs(image.cam_from_world().matrix() - before.cam_from_world().matrix()))))
        current_camera = model.cameras[image.camera_id]
        expected_camera = pycolmap.Camera(reference.cameras[before.camera_id].todict())
        expected_camera.rescale(current_camera.width, current_camera.height)
        if expected_camera.model != current_camera.model:
            raise RuntimeError("Calibrated camera models differ from approved reference")
        max_calibration = max(max_calibration, float(np.max(np.abs(expected_camera.params - current_camera.params))))
    if max_pose > 1e-10 or max_calibration > 1e-8:
        raise RuntimeError("E3 no longer preserves the crutch-approved world frame/calibration")
    helpers = []
    for relative in (
        "reconstructions/experiments/black_person_crutches_cleanup/filter_person_crutches.py",
        "reconstructions/experiments/black_person_crutches_cleanup/crutch_protection/build_protection.py",
        "work/quality-sfm/compare_models.py",
    ):
        original = code.parent / "sfm/legacy_tools" / relative
        destination = code / "legacy_tools" / relative
        if sha256_file(original) != sha256_file(destination):
            raise RuntimeError(f"Copied helper differs: {destination}")
        helpers.append({"original_source": str(original), "mvs_copy": str(destination), "sha256": sha256_file(destination),
                        "use": "Preserved sparse algorithm reference; not directly applied to dense points."})
    validation = validate_inputs(config, config.paths(policy))
    record = {
        "status": "prepared", "created_at": utc_now(), "experiment": config.experiment,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "source_experiment": "E3 dark quality, fixed E1 poses; not a new SfM estimate",
        "selection_reason": "Richest available unfiltered quality correspondence graph among E3/E6/E8 (101150 points); all preserve E1 poses. No dark E10 exists. Point count is not ground-truth accuracy.",
        "source_comparison": comparison, "input_validation": validation,
        "original_input_provenance": str(root / "manifests/input_provenance.json"),
        "staged_file_count": len(provenance["files"]), "extra_files": extras, "copied_helpers": helpers,
        "pycolmap_version": pycolmap.__version__, "max_reference_pose_difference": max_pose,
        "max_reference_scaled_calibration_difference": max_calibration,
        "registered_images": model.num_reg_images(), "cameras": model.num_cameras(), "source_points": model.num_points3D(),
        "mask_policy": "Raw unmasked MVS. Copied body masks are not applied because they omit crutch shafts.",
        "dense_cleanup_complete": False, "dense_crutch_overlay_review_complete": False,
        "dense_cleanup_requirements": ["Compute support on new XYZ rather than old point IDs", "Use the correct original/undistorted pixel coordinates", "Sparse feature-track quality gates cannot be applied unchanged to dense fusion points", "Create and review new crutch projection overlays; retain raw output"],
        "pilot_profile": "All 290 images; max1024,5 source views,3 PatchMatch iterations,4 CPU threads. Reduced-resolution/resource pilot, not the light1600 quality profile.",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu_execution_started": False,
    }
    atomic_json(root / "manifests/dark_preparation.json", record)
    print(json.dumps({key: record[key] for key in ("status", "experiment", "registered_images", "cameras", "source_points", "staged_file_count", "max_reference_pose_difference", "max_reference_scaled_calibration_difference", "mask_policy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
