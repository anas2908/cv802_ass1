#!/usr/bin/env python3
"""Prove dark masks/corridors name the exact pixels used by dark VGGSfM."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256
from vggsfm_engine.paths import production_layout


RUN_ID = "dark-shirt-vggsfm-all290-a10040-fit-v2"
INPUT_DATASET = "dark_shirt_cleanup_inputs"
MVS_ROOT = Path(
    "/l/users/anas.khan/cv_802_ass1/mvs/experiments/"
    "dark_e3_colmap_mvs_1024_raw_v1"
)


def _map(rows: list[dict], *, path_key: str, prefix: str = "") -> dict[str, dict]:
    result = {}
    for row in rows:
        value = row[path_key]
        if prefix:
            if not value.startswith(prefix):
                continue
            value = value.removeprefix(prefix)
        if value in result:
            raise RuntimeError(f"duplicate provenance name: {value}")
        result[value] = row
    return result


def audit() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("pixel-identity audit must run in the active allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    vgg_input_root = layout.inputs / "dark_shirt"
    cleanup_root = layout.inputs / INPUT_DATASET
    destination = cleanup_root / "pixel_identity_audit.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("status") != "complete" or existing.get("image_count") != 290:
            raise RuntimeError("existing pixel-identity audit is not complete")
        return existing

    request_path = layout.experiment_root(RUN_ID) / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_receipt_path = vgg_input_root / "input_receipt.json"
    input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
    source_audit_path = vgg_input_root / "source_provenance_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    mvs_provenance_path = MVS_ROOT / "manifests/input_provenance.json"
    mvs_provenance = json.loads(mvs_provenance_path.read_text(encoding="utf-8"))
    mask_manifest_path = MVS_ROOT / "inputs/mask_manifest.json"
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    cleanup_receipt_path = cleanup_root / "receipt.json"
    cleanup_receipt = json.loads(cleanup_receipt_path.read_text(encoding="utf-8"))

    request_images = _map(request["input_manifest"]["images"], path_key="path", prefix="images/")
    receipt_images = _map(input_receipt["files"], path_key="path", prefix="images/")
    audit_images = _map(source_audit["rows"], path_key="relative_path")
    mvs_images = _map(
        [row for row in mvs_provenance["files"] if row.get("role") == "registered_image"],
        path_key="destination_relative", prefix="inputs/images/",
    )
    local_masks = {row["image_relative"]: row for row in cleanup_receipt["masks"]}
    source_masks = mask_manifest["images"]
    official_sources = {
        row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    sets = [
        set(request_images), set(receipt_images), set(audit_images), set(mvs_images),
        set(local_masks), set(source_masks), official_sources,
    ]
    if any(names != sets[0] for names in sets[1:]) or len(sets[0]) != 290:
        raise RuntimeError("image/mask provenance name sets are not identical 290-way")
    if not (
        source_audit.get("status") == "complete"
        and source_audit.get("source_and_destination_sha256_verified") is True
        and source_audit.get("record_sets_equal") is True
        and source_audit.get("source_root") == str(MVS_ROOT / "inputs/images")
        and mask_manifest.get("complete") is True
        and cleanup_receipt.get("status") == "complete"
    ):
        raise RuntimeError("upstream copy/mask provenance is not complete")

    rows = []
    for name in sorted(sets[0]):
        expected_hashes = {
            request_images[name]["sha256"], receipt_images[name]["sha256"],
            audit_images[name]["sha256"], mvs_images[name]["sha256"],
        }
        expected_sizes = {
            request_images[name]["bytes"], receipt_images[name]["bytes"],
            audit_images[name]["bytes"], mvs_images[name]["size_bytes"],
        }
        if len(expected_hashes) != 1 or len(expected_sizes) != 1:
            raise RuntimeError(f"declared image identity mismatch: {name}")
        vgg_path = vgg_input_root / "images" / name
        mvs_path = MVS_ROOT / "inputs/images" / name
        if vgg_path.is_symlink() or mvs_path.is_symlink():
            raise RuntimeError(f"image identity input cannot be a symlink: {name}")
        vgg_record = file_record(vgg_path)
        mvs_record = file_record(mvs_path)
        expected_hash = next(iter(expected_hashes))
        expected_size = next(iter(expected_sizes))
        if (
            vgg_record["sha256"] != expected_hash
            or mvs_record["sha256"] != expected_hash
            or vgg_record["bytes"] != expected_size
            or mvs_record["bytes"] != expected_size
        ):
            raise RuntimeError(f"actual MVS/VGGSfM image pixels differ: {name}")
        with Image.open(vgg_path) as image:
            dimensions = list(image.size)
        source_mask = source_masks[name]
        local_mask = local_masks[name]
        if dimensions != [source_mask["width"], source_mask["height"]] or dimensions != [
            local_mask["width"], local_mask["height"]
        ]:
            raise RuntimeError(f"image/mask native dimensions differ: {name}")
        mask_path = cleanup_root / local_mask["mask_relative"]
        mask_record = file_record(mask_path)
        if (
            mask_record["sha256"] != source_mask["sha256"]
            or mask_record["sha256"] != local_mask["sha256"]
            or mask_record["bytes"] != local_mask["bytes"]
        ):
            raise RuntimeError(f"actual mask identity differs: {name}")
        rows.append({
            "image_relative": name,
            "width": dimensions[0],
            "height": dimensions[1],
            "image_bytes": expected_size,
            "image_sha256": expected_hash,
            "vggsfm_image_path": str(vgg_path),
            "mvs_provenance_image_path": str(mvs_path),
            "mask_bytes": mask_record["bytes"],
            "mask_sha256": mask_record["sha256"],
            "same_actual_image_bytes_sha256": True,
            "mask_native_dimensions_match_image": True,
        })

    receipt = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "image_count": len(rows),
        "mask_count": len(rows),
        "all_name_sets_equal": True,
        "all_declared_and_actual_image_hashes_equal": True,
        "all_mask_hashes_and_native_dimensions_match": True,
        "identity_statement": (
            "Every method-local body mask and 2D corridor name addresses the exact original "
            "image bytes hashed into the dark VGGSfM request."
        ),
        "sources": {
            "vggsfm_request": file_record(request_path),
            "vggsfm_input_receipt": file_record(input_receipt_path),
            "vggsfm_source_copy_audit": file_record(source_audit_path),
            "mvs_input_provenance": file_record(mvs_provenance_path),
            "source_mask_manifest": file_record(mask_manifest_path),
            "method_local_cleanup_receipt": file_record(cleanup_receipt_path),
        },
        "rows_sha256": object_sha256(rows),
        "rows": rows,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }
    atomic_write_json(destination, receipt)
    return receipt


def main() -> int:
    receipt = audit()
    print(json.dumps({key: receipt[key] for key in (
        "status", "image_count", "mask_count", "rows_sha256",
        "all_declared_and_actual_image_hashes_equal",
        "all_mask_hashes_and_native_dimensions_match",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
