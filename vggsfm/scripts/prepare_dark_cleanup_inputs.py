#!/usr/bin/env python3
"""Copy and verify dark body masks and portable 2D crutch corridors.

Only image-space evidence is copied.  Historical/MVS point IDs, world-space
capsules and camera poses are deliberately excluded: a later cleanup must
project dark VGGSfM's own XYZ with dark VGGSfM's own cameras.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import (
    atomic_write_json,
    file_record,
    file_sha256,
    object_sha256,
)
from vggsfm_engine.paths import production_layout, require_within


SOURCE_INPUT = Path(
    "/l/users/anas.khan/cv_802_ass1/mvs/experiments/"
    "dark_e3_colmap_mvs_1024_raw_v1/inputs"
)
SOURCE_MASK_MANIFEST = SOURCE_INPUT / "mask_manifest.json"
SOURCE_CORRIDORS = (
    SOURCE_INPUT / "provenance/crutches/protection_approved.rebased.json"
)
DARK_INPUT_RECEIPT = Path(
    "/l/users/anas.khan/cv_802_ass1/vggsfm/inputs/dark_shirt/input_receipt.json"
)
DARK_REQUEST = Path(
    "/l/users/anas.khan/cv_802_ass1/vggsfm/experiments/"
    "dark-shirt-vggsfm-all290-a10040-fit-v2/request.json"
)
DESTINATION_NAME = "dark_shirt_cleanup_inputs"
EXPECTED_IMAGES = 290
EXPECTED_CORRIDOR_IMAGES = 6


def _source_names() -> set[str]:
    input_receipt = json.loads(DARK_INPUT_RECEIPT.read_text(encoding="utf-8"))
    request = json.loads(DARK_REQUEST.read_text(encoding="utf-8"))
    receipt_names = {
        row["path"].removeprefix("images/")
        for row in input_receipt.get("files", [])
        if row.get("path", "").startswith("images/")
    }
    request_names = {
        row["source"].removeprefix("images/")
        for row in request.get("official_image_name_map", [])
    }
    if (
        input_receipt.get("image_count") != EXPECTED_IMAGES
        or len(receipt_names) != EXPECTED_IMAGES
        or len(request_names) != EXPECTED_IMAGES
        or receipt_names != request_names
    ):
        raise RuntimeError("dark VGGSfM input/request image sets are not the exact same 290 names")
    return request_names


def _portable_corridors(source_names: set[str]) -> dict[str, Any]:
    source = json.loads(SOURCE_CORRIDORS.read_text(encoding="utf-8"))
    annotations = source.get("annotated_image_corridors")
    if not isinstance(annotations, dict) or len(annotations) != EXPECTED_CORRIDOR_IMAGES:
        raise RuntimeError("expected exactly six reviewed crutch-corridor images")
    if not set(annotations).issubset(source_names):
        raise RuntimeError("a crutch corridor does not name a dark VGGSfM input image")

    for image_name, groups in annotations.items():
        if not isinstance(groups, dict) or not groups:
            raise RuntimeError(f"empty corridor group set: {image_name}")
        for group_name, group in groups.items():
            width = group.get("half_width_fraction_of_image_width")
            polylines = group.get("polylines")
            if not isinstance(width, (int, float)) or not 0.0 < width <= 0.05:
                raise RuntimeError(f"unsafe corridor width: {image_name}/{group_name}")
            if not isinstance(polylines, list) or not polylines:
                raise RuntimeError(f"empty polylines: {image_name}/{group_name}")
            for polyline in polylines:
                if not isinstance(polyline, list) or len(polyline) < 2:
                    raise RuntimeError(f"invalid polyline: {image_name}/{group_name}")
                for point in polyline:
                    if (
                        not isinstance(point, list)
                        or len(point) != 2
                        or not all(isinstance(value, (int, float)) for value in point)
                        or not all(0.0 <= float(value) <= 1.0 for value in point)
                    ):
                        raise RuntimeError(f"non-normalized corridor point: {image_name}")

    return {
        "schema_version": 1,
        "status": "reviewed_image_space_evidence",
        "annotated_image_corridors": annotations,
        "coordinate_frame": (
            "normalized [x/original_width, y/original_height], top-left origin; "
            "corridor half-width is a fraction of original image width"
        ),
        "portable_selection_policy": {
            "per_view": "union of every labelled corridor group/polyline in that image",
            "minimum_distinct_annotated_views": 3,
            "minimum_pairwise_camera_center_to_point_ray_separation_degrees": 15.0,
            "visibility": (
                "only explicitly annotated exposed image corridors vote; no occlusion test"
            ),
        },
        "scope": (
            "2D reviewed evidence only; later projection must use VGGSfM's own dark cameras/XYZ"
        ),
        "explicit_exclusions": [
            "no MVS/E1 point IDs",
            "no MVS/E1 world-space capsules or coordinates",
            "no MVS/E1 camera poses",
            "no broad candidate seed regions",
        ],
        "limitations": [
            "A/B or near/far corridor groups are unioned within each image.",
            "A confirming projection does not prove visibility because no occlusion test exists.",
            "Smooth crutch surfaces absent from raw geometry cannot be restored by filtering.",
            "Hand, body and floor contact regions remain ambiguous.",
        ],
        "source_record": file_record(SOURCE_CORRIDORS),
        "source_review_status": source.get("review", {}).get("status"),
    }


def _validate_existing(destination: Path) -> dict[str, Any]:
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("mask_count") != EXPECTED_IMAGES
        or receipt.get("crutch_corridor_image_count") != EXPECTED_CORRIDOR_IMAGES
    ):
        raise RuntimeError("existing dark cleanup input has no complete receipt")
    for row in receipt.get("masks", []):
        path = require_within(
            destination / row["mask_relative"], destination / "masks",
            label="existing dark body mask", must_exist=True,
        )
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"existing dark body mask changed: {row['image_relative']}")
        with Image.open(path) as image:
            if image.mode != "L" or list(image.size) != [row["width"], row["height"]]:
                raise RuntimeError(f"existing dark body mask metadata changed: {path}")
    corridors = destination / "crutches" / "corridors.json"
    declared = receipt["crutch_corridors"]
    if corridors.stat().st_size != declared["bytes"] or file_sha256(corridors) != declared["sha256"]:
        raise RuntimeError("existing crutch corridor evidence changed")
    return receipt


def prepare() -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("dark cleanup preparation must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    destination = layout.inputs / DESTINATION_NAME
    if destination.exists():
        return _validate_existing(destination)

    source_names = _source_names()
    source_root = SOURCE_INPUT.resolve(strict=True)
    source_masks = (source_root / "masks").resolve(strict=True)
    manifest = json.loads(SOURCE_MASK_MANIFEST.read_text(encoding="utf-8"))
    rows_by_name = manifest.get("images", {})
    if (
        manifest.get("complete") is not True
        or len(rows_by_name) != EXPECTED_IMAGES
        or set(rows_by_name) != source_names
        or not all(row.get("usable") is True for row in rows_by_name.values())
    ):
        raise RuntimeError("body-mask manifest does not exactly cover all 290 dark inputs")
    portable_corridors = _portable_corridors(source_names)

    staging = layout.inputs / f".{DESTINATION_NAME}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    layout.assert_member(staging, label="dark cleanup input staging")
    (staging / "masks").mkdir(parents=True)
    (staging / "crutches").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        for image_name in sorted(source_names):
            source_row = rows_by_name[image_name]
            relative = Path(source_row["mask_relative_path"])
            if relative.parts[:1] != ("masks",) or ".." in relative.parts:
                raise RuntimeError(f"unsafe body-mask path: {relative}")
            source = require_within(
                source_root / relative, source_masks,
                label="source dark body mask", must_exist=True,
            )
            if file_sha256(source) != source_row["sha256"]:
                raise RuntimeError(f"source body-mask checksum mismatch: {image_name}")
            target_relative = Path(image_name).with_suffix(".png")
            target = staging / "masks" / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.copying")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            with Image.open(target) as image:
                extrema = image.getextrema()
                if (
                    image.mode != "L"
                    or image.size != (source_row["width"], source_row["height"])
                    or extrema != (0, 255)
                ):
                    raise RuntimeError(f"unexpected body-mask metadata: {image_name}")
            digest = file_sha256(target)
            if digest != source_row["sha256"]:
                raise RuntimeError(f"copied body-mask checksum mismatch: {image_name}")
            rows.append({
                "image_relative": image_name,
                "mask_relative": (Path("masks") / target_relative).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": digest,
                "width": int(source_row["width"]),
                "height": int(source_row["height"]),
                "mode": "L",
                "minimum": 0,
                "maximum": 255,
            })

        corridor_path = staging / "crutches" / "corridors.json"
        atomic_write_json(corridor_path, portable_corridors)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": DESTINATION_NAME,
            "mask_count": len(rows),
            "image_count": len(source_names),
            "crutch_corridor_image_count": len(
                portable_corridors["annotated_image_corridors"]
            ),
            "exact_dark_vggsfm_input_name_coverage": True,
            "body_masks_include_crutches": False,
            "body_mask_threshold_policy": "foreground iff uint8 value >= 128",
            "body_mask_semantics": manifest["value_semantics"],
            "body_mask_pixel_coordinates": manifest["pixel_coordinates"],
            "source_mask_manifest": file_record(SOURCE_MASK_MANIFEST),
            "dark_vggsfm_input_receipt": file_record(DARK_INPUT_RECEIPT),
            "dark_vggsfm_request": file_record(DARK_REQUEST),
            "masks_sha256": object_sha256(rows),
            "masks": rows,
            "crutch_corridors": file_record(corridor_path, relative_to=staging),
            "runtime_independence": (
                "All later cleanup inputs are method-local; source MVS files are provenance only."
            ),
            "no_cross_method_geometry": True,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        }
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _validate_existing(destination)


def main() -> int:
    receipt = prepare()
    print(json.dumps({key: receipt[key] for key in (
        "status", "dataset", "image_count", "mask_count",
        "crutch_corridor_image_count", "masks_sha256", "crutch_corridors",
        "no_cross_method_geometry",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
