#!/usr/bin/env python3
"""Derive a person-only VGGSfM PLY from verified historical Mac masks.

This follows the reviewed saved-result policy: project *this run's* VGGSfM
points through *this run's* VGGSfM cameras into native-size Mac person masks;
keep points with >=6 usable views and >=90% foreground agreement. For the dark
shirt, also retain points confirmed by three angularly separated, reviewed
image-space crutch corridors. No historical SfM/MVS 3D points or poses enter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tempfile

import numpy as np
from PIL import Image

from clean_point_cloud import read_coloured_ply
from vggsfm_engine import colmap
from vggsfm_engine.projection_cleanup import (
    corridor_hits, project_original_pixels, separated_view_confirmation,
)


SOURCE_NAMES = {
    "light_shirt": "light_shirt_cleanup_masks",
    "dark_shirt": "dark_shirt_cleanup_inputs",
}
MINIMUM_USABLE = 6
FOREGROUND_AGREEMENT = 0.9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or any(part in (".", "..") for part in value.parts):
        raise ValueError(f"Unsafe relative path in mask receipt: {name!r}")
    return Path(*value.parts)


def source_rows(receipt: dict) -> list[dict]:
    rows = receipt.get("rows", receipt.get("masks"))
    if receipt.get("status") != "complete" or not isinstance(rows, list) or not rows:
        raise ValueError("Historical mask receipt is incomplete")
    names = [row["image_relative"] for row in rows]
    if len(set(names)) != len(names) or receipt.get("mask_count") != len(rows):
        raise ValueError("Historical mask receipt has duplicate or missing rows")
    return rows


def verify_staged(folder: Path) -> dict:
    receipt = json.loads((folder / "receipt.json").read_text())
    rows = source_rows(receipt)
    for row in rows:
        relative = safe_relative(row["mask_relative"])
        if relative.parts[0] != "masks":
            raise ValueError("Mask must reside below masks/")
        path = folder / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Staged mask failed checksum: {relative}")
    if receipt["dataset"] == "dark_shirt":
        corridor = folder / "crutches" / "corridors.json"
        if corridor.is_symlink() or not corridor.is_file() or sha256(corridor) != receipt["corridors_sha256"]:
            raise ValueError("Staged crutch corridors failed checksum")
    return receipt


def stage_masks(dataset: str, method_root: Path, reference_root: Path) -> tuple[Path, dict]:
    """Copy only image-space evidence into the current VGGSfM data root."""
    if dataset not in SOURCE_NAMES:
        raise ValueError("Mac masks are available only for light_shirt and dark_shirt")
    inputs = method_root / "inputs"
    destination = inputs / f"{dataset}_mac_masks"
    if destination.exists():
        return destination, verify_staged(destination)
    source = reference_root / "vggsfm" / "inputs" / SOURCE_NAMES[dataset]
    source_receipt = source / "receipt.json"
    if not source_receipt.is_file():
        raise FileNotFoundError(
            f"Mac masks not found at {source_receipt}. Set CV802_REFERENCE_DATA_ROOT "
            "to the prior assignment data root containing vggsfm/inputs."
        )
    original = json.loads(source_receipt.read_text())
    rows = source_rows(original)
    inputs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-mac-masks-", dir=inputs) as temporary:
        staging = Path(temporary)
        for row in rows:
            relative = safe_relative(row["mask_relative"])
            if relative.parts[0] != "masks":
                raise ValueError("Historical mask path must reside below masks/")
            source_path = source / relative
            if source_path.is_symlink() or not source_path.is_file() or source_path.stat().st_size != row["bytes"] or sha256(source_path) != row["sha256"]:
                raise ValueError(f"Historical mask failed checksum: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            if sha256(target) != row["sha256"]:
                raise ValueError(f"Copied mask failed checksum: {relative}")
        corridors_hash = None
        if dataset == "dark_shirt":
            description = original.get("crutch_corridors", {})
            corridor_source = source / safe_relative(description["path"])
            corridors_hash = description["sha256"]
            if corridor_source.is_symlink() or sha256(corridor_source) != corridors_hash:
                raise ValueError("Historical crutch corridor checksum mismatch")
            target = staging / "crutches" / "corridors.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(corridor_source, target)
            if sha256(target) != corridors_hash:
                raise ValueError("Copied crutch corridors failed checksum")
        local = {
            "schema_version": 1, "status": "complete", "dataset": dataset,
            "mask_count": len(rows), "source_receipt_sha256": sha256(source_receipt),
            "source_receipt_path": str(source_receipt.resolve()),
            "corridors_sha256": corridors_hash, "rows": rows,
            "provenance": "Historical Mac segmentation masks and reviewed 2D crutch corridors only; no historical 3D geometry or poses.",
        }
        (staging / "receipt.json").write_text(json.dumps(local, indent=2, sort_keys=True) + "\n")
        if destination.exists():
            return destination, verify_staged(destination)
        os.replace(staging, destination)
    return destination, verify_staged(destination)


def select_by_support(foreground: np.ndarray, usable: np.ndarray) -> np.ndarray:
    if foreground.shape != usable.shape:
        raise ValueError("Projection vote arrays differ in shape")
    return (usable >= MINIMUM_USABLE) & (
        foreground >= np.ceil(FOREGROUND_AGREEMENT * usable - 1e-12).astype(np.uint16)
    )


def raw_input_records(raw_root: Path, request_path: Path, mask_receipt_path: Path) -> dict[str, str]:
    paths = [
        raw_root / "manifest.json", raw_root / "point_cloud.ply",
        *(raw_root / "colmap" / "sparse" / "0" / name
          for name in ("cameras.bin", "images.bin", "points3D.bin")),
        request_path, mask_receipt_path,
    ]
    return {str(path): sha256(path) for path in paths}


def derive(dataset: str, data_root: Path, reference_root: Path) -> dict:
    """Create a separate mask-cleaned point cloud; never replace the raw one."""
    method_root = data_root / "vggsfm"
    run_id = f"ciai-{dataset}-vggsfm-v1"
    raw_root = method_root / "outputs" / run_id
    destination = method_root / "outputs" / f"{run_id}-mac-mask-clean-v1"
    raw_ply = raw_root / "point_cloud.ply"
    raw_manifest = json.loads((raw_root / "manifest.json").read_text())
    if raw_manifest.get("status") != "complete" or raw_manifest.get("run_id") != run_id:
        raise ValueError("A completed raw VGGSfM run is required before mask cleanup")
    request_path = method_root / "experiments" / run_id / "request.json"
    request = json.loads(request_path.read_text())
    masks_root, mask_receipt = stage_masks(dataset, method_root, reference_root)
    mask_rows = {row["image_relative"]: row for row in mask_receipt["rows"]}
    official_to_source = {
        row["official"]: row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    if (len(official_to_source) != len(mask_rows)
            or set(official_to_source.values()) != set(mask_rows)):
        raise ValueError("Mac masks do not exactly cover this VGGSfM image set")
    model = raw_root / "colmap" / "sparse" / "0"
    colmap.validate_binary_model(model)
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(model))
    if {image.name for image in reconstruction.images.values()} != set(official_to_source):
        raise ValueError("VGGSfM registered images differ from the mask image set")
    point_ids = np.asarray(sorted(reconstruction.points3D), dtype=np.uint64)
    xyz = np.stack([np.asarray(reconstruction.points3D[int(i)].xyz, dtype=np.float64) for i in point_ids])
    if raw_manifest.get("point_cloud", {}).get("point_count") != len(point_ids):
        raise ValueError("Raw PLY/COLMAP point count differs from manifest")
    raw_hash = sha256(raw_ply)
    receipt_path = destination / "cleanup_receipt.json"
    if destination.exists():
        if not receipt_path.is_file():
            raise FileExistsError(f"Incomplete prior cleanup preserved at {destination}")
        receipt = json.loads(receipt_path.read_text())
        output = destination / "point_cloud.ply"
        if (receipt.get("raw_sha256") != raw_hash
                or receipt.get("mask_receipt_sha256") != sha256(masks_root / "receipt.json")
                or not output.is_file() or receipt.get("cleaned_sha256") != sha256(output)):
            raise FileExistsError(f"Existing cleanup differs from current inputs: {destination}")
        return receipt

    before = raw_input_records(raw_root, request_path, masks_root / "receipt.json")
    foreground = np.zeros(len(point_ids), dtype=np.uint16)
    usable = np.zeros(len(point_ids), dtype=np.uint16)
    annotations = {}
    if dataset == "dark_shirt":
        annotations = json.loads((masks_root / "crutches" / "corridors.json").read_text())["annotated_image_corridors"]
        if len(annotations) != 6 or not set(annotations).issubset(mask_rows):
            raise ValueError("Reviewed dark crutch corridor set is incomplete")
    corridor_hits_by_view = []
    corridor_centers = []
    ordered_images = sorted(reconstruction.images.values(), key=lambda item: item.name)
    for index, image in enumerate(ordered_images, 1):
        name = official_to_source[image.name]
        row = mask_rows[name]
        camera = reconstruction.cameras[image.camera_id]
        mask_path = masks_root / safe_relative(row["mask_relative"])
        if sha256(mask_path) != row["sha256"]:
            raise ValueError(f"Mask changed before projection: {name}")
        with Image.open(mask_path) as opened:
            if opened.mode != "L" or opened.size != (int(camera.width), int(camera.height)):
                raise ValueError(f"Mask/camera dimensions or mode disagree: {name}")
            mask = np.asarray(opened, dtype=np.uint8)
        pixels, valid = project_original_pixels(image, camera, xyz)
        indices = np.flatnonzero(valid)
        rounded = np.rint(pixels[indices]).astype(np.int64)
        nearest = ((rounded[:, 0] >= 0) & (rounded[:, 0] < int(camera.width))
                   & (rounded[:, 1] >= 0) & (rounded[:, 1] < int(camera.height)))
        indices, rounded = indices[nearest], rounded[nearest]
        usable[indices] += 1
        foreground[indices] += (mask[rounded[:, 1], rounded[:, 0]] >= 128).astype(np.uint16)
        if name in annotations:
            corridor_hits_by_view.append(corridor_hits(
                pixels, valid, width=int(camera.width), height=int(camera.height),
                groups=annotations[name],
            ))
            corridor_centers.append(np.asarray(image.projection_center(), dtype=np.float64))
        if index % 20 == 0 or index == len(ordered_images):
            print(f"Mask projection {index}/{len(ordered_images)} views", flush=True)
    body = select_by_support(foreground, usable)
    crutches = np.zeros(len(point_ids), dtype=bool)
    if dataset == "dark_shirt":
        if len(corridor_hits_by_view) != 6:
            raise ValueError("Not all six reviewed crutch-corridor views are registered")
        crutches = separated_view_confirmation(
            xyz, np.stack(corridor_centers), np.stack(corridor_hits_by_view),
            minimum_views=3, minimum_pairwise_angle_degrees=15.0,
        )
    selected_ids = {int(value) for value in point_ids[body | crutches]}
    if not selected_ids or len(selected_ids) == len(point_ids):
        raise ValueError("Mask cleanup selected an implausible empty/full result")
    header, records, _ = read_coloured_ply(raw_ply)
    count, points = colmap.iter_points3d(model / "points3D.bin")
    if count != len(records):
        raise ValueError("Raw PLY does not match VGGSfM COLMAP points")
    selected = np.zeros(count, dtype=bool)
    for index, point in enumerate(points):
        record = struct.pack("<fffBBB", point.x, point.y, point.z,
                             point.red, point.green, point.blue)
        if records[index].tobytes() != record:
            raise ValueError("Raw PLY point order or XYZ/RGB differs from COLMAP")
        selected[index] = point.point_id in selected_ids
    if int(selected.sum()) != len(selected_ids):
        raise ValueError("Selected COLMAP point IDs not found in the raw PLY")
    verify_staged(masks_root)
    new_header, changes = re.subn(
        rb"(?m)^element vertex [0-9]+(?=\r?$)",
        b"element vertex " + str(len(selected_ids)).encode(), header,
    )
    if changes != 1:
        raise ValueError("Could not update PLY vertex count")
    if raw_input_records(raw_root, request_path, masks_root / "receipt.json") != before:
        raise ValueError("A VGGSfM source file changed during cleanup")
    outputs = method_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-mask-clean-", dir=outputs) as temporary:
        staged = Path(temporary)
        cloud = staged / "point_cloud.ply"
        cloud.write_bytes(new_header + records[selected].tobytes())
        receipt = {
            "schema_version": 1, "status": "complete", "dataset": dataset,
            "run_id": destination.name, "source_run_id": run_id,
            "method": "VGGSfM native-Mac-mask projection consensus with dark crutch-corridor protection",
            "raw_sha256": raw_hash, "cleaned_sha256": sha256(cloud),
            "mask_receipt_sha256": sha256(masks_root / "receipt.json"),
            "mask_source": str(masks_root), "mask_count": len(mask_rows),
            "registered_images": len(ordered_images), "raw_points": len(point_ids),
            "cleaned_points": len(selected_ids), "removed_points": len(point_ids) - len(selected_ids),
            "body_selected_points": int(body.sum()),
            "crutch_selected_points": int(crutches.sum()),
            "minimum_usable_views": MINIMUM_USABLE,
            "minimum_foreground_agreement": FOREGROUND_AGREEMENT,
            "crutch_minimum_views": 3 if dataset == "dark_shirt" else None,
            "crutch_minimum_pairwise_angle_degrees": 15.0 if dataset == "dark_shirt" else None,
            "raw_poses_reused": True, "historical_3d_geometry_or_poses_reused": False,
            "raw_output_replaced": False,
            "limitations": "Projection is not an occlusion test; masks can omit thin foreground and corridors cannot restore missing crutch points.",
        }
        (staged / "cleanup_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if destination.exists():
            raise FileExistsError(f"Cleanup appeared concurrently: {destination}")
        os.replace(staged, destination)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCE_NAMES), required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.dataset, args.data_root, args.reference_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
