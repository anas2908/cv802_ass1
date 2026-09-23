#!/usr/bin/env python3
"""Publish the independently audited and visually reviewed dark VGGSfM cleanup."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import atomic_write_json, file_record, read_json
from vggsfm_engine.paths import production_layout


SOURCE_RUN = "dark-shirt-vggsfm-all290-a10040-fit-v2"
RUN_ID = SOURCE_RUN + "-mask-crutch-clean-v1"
AUDIT_ID = RUN_ID + "-validation-v1"
PREVIEW_ROOT = Path("/l/users/anas.khan/cv_802_ass1/evaluation/preview-vggsfm-dark-clean-v1")


def absolute_record(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    record = file_record(resolved)
    record["path"] = str(resolved)
    return record


def require_record(path: Path, declared: dict) -> dict:
    actual = absolute_record(path)
    if actual["bytes"] != declared.get("bytes") or actual["sha256"] != declared.get("sha256"):
        raise RuntimeError(f"review artifact differs from receipt: {path}")
    return actual


def publish() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("publication must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    output = layout.output_root(RUN_ID)
    publication_path = output / "publication.json"
    visual_path = output / "visual_review.json"
    if publication_path.exists() or visual_path.exists():
        raise RuntimeError("refusing to overwrite an existing review/publication")

    cleanup_receipt_path = output / "cleanup_receipt.json"
    cleanup_manifest_path = output / "manifest.json"
    cleanup = read_json(cleanup_receipt_path)
    manifest = read_json(cleanup_manifest_path)
    audit_path = layout.root / "evaluation" / AUDIT_ID / "audit_receipt.json"
    audit = read_json(audit_path)
    overlays_root = output / "overlays"
    overlays_path = overlays_root / "overlay_receipt.json"
    overlays = read_json(overlays_path)
    preview_path = PREVIEW_ROOT / "preview_receipt.json"
    preview = read_json(preview_path)
    if (
        cleanup.get("status") != "complete"
        or manifest.get("status") != "complete"
        or audit.get("status") != "complete"
        or overlays.get("status") != "rendered_pending_visual_review"
        or cleanup.get("run_id") != RUN_ID
        or audit.get("run_id") != RUN_ID
        or overlays.get("run_id") != RUN_ID
    ):
        raise RuntimeError("cleanup/audit/overlay state is not publication-ready")

    selection = cleanup["selection"]
    expected_selection = {
        "source_count": 187308,
        "body_selected_count": 57061,
        "crutch_selected_count": 6140,
        "body_crutch_overlap_count": 312,
        "crutch_only_protected_count": 5828,
        "retained_count": 62889,
        "removed_count": 124419,
        "count_conservation": True,
    }
    if selection != expected_selection or any(
        audit["selection"].get(key) != value for key, value in expected_selection.items()
    ):
        raise RuntimeError("cleanup and independent audit selection counts are not canonical")
    expected_invariants = {
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
    }
    if audit["invariants"] != expected_invariants:
        raise RuntimeError("independent audit has a failed invariant")

    point_cloud = output / "point_cloud.ply"
    point_record = absolute_record(point_cloud)
    audit_point = audit["selection"]["published_ply"]
    if (
        point_record["bytes"] != 943607
        or point_record["sha256"]
        != "0ec0e6983502cd567b1616a1cc315e468b889feabcc77c961c645920448b67c8"
        or point_record["bytes"] != audit_point["bytes"]
        or point_record["sha256"] != audit_point["sha256"]
        or audit["selection"]["published_matches_independent_regeneration_byte_for_byte"]
        is not True
    ):
        raise RuntimeError("cleaned PLY is not the independently regenerated canonical file")

    overlay_artifacts = []
    for row in overlays["rows"]:
        overlay_artifacts.append(
            require_record(overlays_root / row["overlay"]["path"], row["overlay"])
        )
    contact = require_record(
        overlays_root / overlays["contact_sheet"]["path"], overlays["contact_sheet"]
    )
    preview_png = require_record(Path(preview["output"]["path"]), preview["output"])
    if (
        preview.get("displayed_points") != 62889
        or preview.get("original_points") != 62889
        or preview.get("coordinate_transform") != "none"
        or preview.get("clipping", "").split(";")[0] != "none"
        or preview.get("source", {}).get("sha256") != point_record["sha256"]
    ):
        raise RuntimeError("3D preview does not bind the full cleaned point cloud")

    visual_review = {
        "schema_version": 1,
        "status": "accepted_with_limitations",
        "accepted_for_separate_saved_view": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "reviewed_point_count": 62889,
        "reviewed_artifacts": {
            "three_dimensional_preview": preview_png,
            "six_view_contact_sheet": contact,
            "six_individual_overlays": overlay_artifacts,
        },
        "observations": [
            "The full 62,889-point preview is a recognizably isolated person with head, torso, arms and legs and greatly reduced radial background scatter.",
            "Bright, slender bilateral crutch-candidate bands are visible in the 3D preview.",
            "Across six original-image overlays, three-view/15-degree-confirmed green XYZ broadly follow both reviewed crutch shaft/fork/handle corridors.",
            "The occluded far crutch is absent where it was deliberately not annotated.",
        ],
        "limitations": [
            "Crutch bands can overlap hand, leg, body, armpit/shoulder and floor-contact regions at corridor boundaries.",
            "No occlusion test is performed; a projection vote is not visibility proof.",
            "The cleanup filters existing raw points and cannot repair gaps, nonuniform density or absent surfaces.",
            "Visual acceptance is not a metric-accuracy, completeness or anatomical-accuracy certification.",
        ],
        "review_basis": (
            "Independent 3D preview plus six original-pixel corridor/mask/confirmed-XYZ overlays"
        ),
        "reviewers": [
            "Codex cleanup implementation agent (direct contact-sheet inspection)",
            "Root orchestration agent (independent preview and contact-sheet inspection)",
        ],
    }
    atomic_write_json(visual_path, visual_review)

    source_paths = [
        Path(__file__),
        Path(__file__).with_name("clean_dark_with_masks_and_crutches.py"),
        Path(__file__).with_name("audit_dark_cleanup.py"),
        Path(__file__).with_name("render_dark_crutch_overlays.py"),
        Path(__file__).with_name("audit_dark_cleanup_pixel_identity.py"),
        Path(__file__).with_name("prepare_dark_cleanup_inputs.py"),
        Path(__file__).resolve().parents[1] / "vggsfm_engine/projection_cleanup.py",
        Path(__file__).resolve().parents[1] / "tests/test_projection_cleanup.py",
    ]
    artifacts = [
        point_record,
        absolute_record(cleanup_receipt_path),
        absolute_record(cleanup_manifest_path),
        absolute_record(output / "projection_rows.json"),
        absolute_record(audit_path),
        absolute_record(overlays_path),
        absolute_record(preview_path),
        absolute_record(visual_path),
        contact,
        preview_png,
        *overlay_artifacts,
    ]
    publication = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN,
        "point_count": 62889,
        "point_cloud": {
            "path": str(point_cloud.resolve(strict=True)),
            "bytes": point_record["bytes"],
            "sha256": point_record["sha256"],
            "vertex_count": 62889,
            "all_xyz_finite": True,
            "has_rgb": True,
            "format": "binary_little_endian",
        },
        "selection": expected_selection,
        "profile": {
            "body": "uint8>=128; >=6 usable projections; >=90% body agreement",
            "crutches": "reviewed original-pixel corridors in >=3 views with every confirming ray pair >=15 degrees",
            "union": "body OR crutch",
            "uses_world_capsules": False,
            "occlusion_test": False,
        },
        "own_dark_vggsfm_cameras": True,
        "uses_e10_or_mvs_geometry": False,
        "exact_actual_image_pixel_identity_290": True,
        "body_masks_verified_before_after_290": True,
        "independent_exact_selection_and_byte_regeneration": True,
        "raw_sources_unchanged": True,
        "raw_output_preserved": True,
        "inference_rerun": False,
        "accepted_for_separate_saved_view": True,
        "quality_label": (
            "derived body-mask projection consensus plus separated-view crutch-corridor protection; accepted with limitations"
        ),
        "visual_review": {
            "status": visual_review["status"],
            **absolute_record(visual_path),
        },
        "independent_audit": absolute_record(audit_path),
        "artifacts": artifacts,
        "source_code": [absolute_record(path) for path in source_paths],
        "limitations": visual_review["limitations"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }
    atomic_write_json(publication_path, publication)
    return publication


def main() -> int:
    result = publish()
    print(json.dumps({key: result[key] for key in (
        "status", "run_id", "point_count", "point_cloud", "selection",
        "accepted_for_separate_saved_view", "visual_review", "independent_audit",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
