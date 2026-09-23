#!/usr/bin/env python3
"""CPU-only silhouette cleanup plus explicit new-XYZ crutch preservation.

This retains original PLY vertex bytes, not resampled geometry. Body silhouettes
are not depth/occlusion tests. Capsule/corridor crutch candidates deliberately
bypass body masks, which omit shafts; candidates are not certified metal points.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv802_mvs.config import MVSConfig
from cv802_mvs.io_utils import atomic_json, sha256_file, utc_now
from cv802_mvs.paths import PathPolicy, safe_experiment_name
from cv802_mvs.validation import validate_colored_ply


def select_union(foreground, usable, protected, minimum_views=6, agreement=.90):
    import numpy as np
    foreground, usable, protected = map(np.asarray, (foreground, usable, protected))
    if foreground.shape != usable.shape or usable.shape != protected.shape:
        raise ValueError("Vote arrays must have matching shapes")
    if minimum_views < 1 or not 0 < agreement <= 1 or np.any(foreground > usable):
        raise ValueError("Invalid silhouette voting parameters")
    body = (usable >= minimum_views) & (foreground >= np.ceil(agreement * usable - 1e-12))
    return body, body | protected.astype(bool)


def read_vertices(path):
    """Read a binary scalar vertex-only PLY and retain its precise byte schema."""
    import numpy as np
    scalar = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
              "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4"}
    with Path(path).open("rb") as stream:
        lines, fields, count, active = [], [], None, None
        if stream.readline() != b"ply\n":
            raise ValueError("Expected PLY")
        lines.append(b"ply\n")
        for _ in range(256):
            line = stream.readline()
            lines.append(line)
            words = line.decode("ascii").split()
            if words == ["end_header"]:
                break
            if words[:1] == ["format"] and words[1] != "binary_little_endian":
                raise ValueError("Only binary little-endian supported")
            if words[:1] == ["element"]:
                active = words[1]
                if active != "vertex":
                    raise ValueError("Only vertex-only PLY supported")
                count = int(words[2])
            if words[:1] == ["property"] and active == "vertex":
                if len(words) != 3 or words[1] not in scalar:
                    raise ValueError("Unsupported scalar schema")
                fields.append((words[2], "<" + scalar[words[1]]))
        else:
            raise ValueError("Missing PLY terminator")
        dtype = np.dtype(fields)
        if not count or not {"x", "y", "z", "red", "green", "blue"} <= set(dict(fields)):
            raise ValueError("Missing XYZRGB")
        if Path(path).stat().st_size - stream.tell() != count * dtype.itemsize:
            raise ValueError("PLY payload length mismatch")
        rows = np.memmap(path, mode="r", dtype=dtype, offset=stream.tell(), shape=(count,))
    return lines, rows


def write_exact_subset(source, selected, destination):
    import numpy as np
    lines, rows = read_vertices(source)
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(rows),) or not selected.any():
        raise ValueError("Invalid or empty selection")
    with Path(destination).open("xb") as output:
        for line in lines:
            output.write(f"element vertex {selected.sum()}\n".encode() if line.startswith(b"element vertex ") else line)
        output.write(rows[selected].tobytes())
    _, actual = read_vertices(destination)
    if actual.tobytes() != rows[selected].tobytes():
        raise ValueError("Subset bytes did not survive serialization")
    return int(selected.sum())


def records(paths):
    result = []
    for path in sorted(set(map(Path, paths))):
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changing during hash: {path}")
        result.append({"path": str(path), "bytes": after.st_size, "sha256": digest})
    return result


def run(config_path, derived_id):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside the existing authorized allocation")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pycolmap
    from PIL import Image, ImageDraw
    start = time.monotonic()
    policy = PathPolicy.production()
    config = MVSConfig.load(config_path, policy)
    paths = config.paths(policy)
    if config.experiment != "dark_e3_colmap_mvs_1024_raw_v1":
        raise ValueError("This profile is scoped to the validated dark E3 run")
    final_path = paths.manifests / "final_validation.json"
    final = json.loads(final_path.read_text())
    if json.loads((paths.root / "run_state.json").read_text())["status"] != "complete":
        raise ValueError("Raw experiment must be complete")
    if final["config_digest"] != config.digest or validate_colored_ply(paths.fused) != final["colored_dense_point_cloud"]:
        raise ValueError("Raw PLY/config differs from authoritative receipt")
    destination = policy.require_method(policy.method_root / "outputs" / safe_experiment_name(derived_id), "derived output")
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "state.json", {"status": "running", "created_at": utc_now(), "raw_source": str(paths.fused)})
    mask_manifest_path = paths.root / "inputs/mask_manifest.json"
    mask_manifest = json.loads(mask_manifest_path.read_text())
    protection_path = paths.root / "inputs/provenance/crutches/protection_approved.rebased.json"
    protection = json.loads(protection_path.read_text())
    model = pycolmap.Reconstruction(str(paths.model))
    dense_model = pycolmap.Reconstruction(str(paths.workspace / "sparse"))
    images = {im.name: im for im in model.images.values()}
    dense_images = {im.name: im for im in dense_model.images.values()}
    if images.keys() != dense_images.keys() or set(mask_manifest["images"]) != set(images):
        raise ValueError("Exact image/model/mask coverage disagreement")
    pose_delta = max(float(np.max(np.abs(im.cam_from_world().matrix() - dense_images[name].cam_from_world().matrix()))) for name, im in images.items())
    if pose_delta > 1e-10:
        raise ValueError("Dense world frame changed")
    mask_paths = {}
    for name, row in mask_manifest["images"].items():
        path = policy.require_method(paths.root / "inputs" / row["mask_relative_path"], "body mask", must_exist=True)
        if not row["usable"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Mask checksum/validity changed: {name}")
        mask_paths[name] = path
    helper_path = Path(__file__).resolve().parents[1] / "legacy_tools/reconstructions/experiments/black_person_crutches_cleanup/crutch_protection/build_protection.py"
    frozen = [paths.fused, final_path, paths.root / "run_state.json", protection_path, mask_manifest_path, helper_path,
              *paths.model.glob("*.bin"), *mask_paths.values(), *[Path(p) for p in protection["source_model_sha256"]]]
    for path, expected in protection["source_model_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("Approved reference changed")
    before = records(frozen)
    _, vertices = read_vertices(paths.fused)
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    foreground = np.zeros(len(xyz), dtype=np.uint16)
    usable = np.zeros(len(xyz), dtype=np.uint16)
    projection_rows = []
    for index, (name, im) in enumerate(sorted(images.items())):
        camera, pose = model.cameras[im.camera_id], im.cam_from_world()
        with Image.open(mask_paths[name]) as source:
            if source.mode != "L":
                raise ValueError("Masks must be native grayscale")
            mask = np.asarray(source)
        height, width = mask.shape
        declared = mask_manifest["images"][name]
        if (width, height) != (declared["width"], declared["height"]):
            raise ValueError("Mask dimensions changed")
        # Historical SfM calibration can be downscaled. The distortion is
        # applied in calibrated pixels FIRST, then scaled into native masks.
        sx, sy = width / camera.width, height / camera.height
        if abs(sx / sy - 1) > .002:
            raise ValueError("Mask and calibrated image aspect ratios disagree")
        cam = xyz @ pose.rotation.matrix().T + pose.translation
        uv = np.asarray(camera.img_from_cam(np.ascontiguousarray(cam))) * [sx, sy]
        finite = np.isfinite(uv).all(axis=1) & (cam[:, 2] > 0)
        finite &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        ids = np.flatnonzero(finite)
        xy = np.floor(uv[ids]).astype(np.int64)
        hits = mask[xy[:, 1], xy[:, 0]] >= 128
        usable[ids] += 1
        foreground[ids[hits]] += 1
        projection_rows.append({"image": name, "camera_id": int(im.camera_id), "camera_model": camera.model.name,
                                "camera_width": camera.width, "camera_height": camera.height, "mask_width": width,
                                "mask_height": height, "scale_x": sx, "scale_y": sy, "usable": len(ids), "foreground": int(hits.sum())})
        if index % 25 == 0:
            atomic_json(destination / "state.json", {"status": "running", "completed_mask_views": index + 1, "total_mask_views": len(images)})
            print(f"mask projection {index + 1}/{len(images)}", flush=True)
    spec = importlib.util.spec_from_file_location("mvs_protected_geometry", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    capsules = {name: helper.capsule_gate(xyz, region["capsules"]) for name, region in protection["regions"].items()}
    candidate_ids = np.flatnonzero(np.logical_or.reduce(list(capsules.values())))
    support, separated, per_view = helper.confirmation_gate(model, xyz[candidate_ids], protection["annotated_image_corridors"], 15.)
    protected = np.zeros(len(xyz), dtype=bool)
    protected[candidate_ids[separated]] = True
    body, selected = select_union(foreground, usable, protected)
    count = write_exact_subset(paths.fused, selected, destination / "point_cloud.ply")
    np.savez_compressed(destination / "selection_audit.npz", source_vertex_indices=np.flatnonzero(selected),
                        foreground=foreground, usable=usable, body=body, protected=protected)
    atomic_json(destination / "projection_rows.json", {"rows": projection_rows})
    overlays = []
    for index, (name, corridors) in enumerate(protection["annotated_image_corridors"].items()):
        im = images[name]
        camera, pose = model.cameras[im.camera_id], im.cam_from_world()
        with Image.open(paths.images / name) as photo:
            photo = photo.convert("RGB")
        scale = min(1., 1400 / max(photo.size))
        size = tuple(round(value * scale) for value in photo.size)
        pixels = np.asarray(photo.resize(size, Image.Resampling.LANCZOS)).copy()
        cam = xyz @ pose.rotation.matrix().T + pose.translation
        uv = np.asarray(camera.img_from_cam(np.ascontiguousarray(cam))) * [size[0] / camera.width, size[1] / camera.height]
        finite = np.isfinite(uv).all(axis=1) & (cam[:, 2] > 0)
        finite &= (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
        for flag, color in ((body, (50, 210, 80)), (protected, (255, 60, 40))):
            xy = np.floor(uv[flag & finite]).astype(int)
            pixels[xy[:, 1], xy[:, 0]] = color
        overlay = Image.fromarray(pixels)
        draw = ImageDraw.Draw(overlay)
        for corridor in corridors.values():
            for line in corridor["polylines"]:
                draw.line([(round(x * size[0]), round(y * size[1])) for x, y in line], fill=(255, 220, 30), width=2)
        draw.rectangle((0, 0, size[0], 44), fill=(0, 0, 0))
        draw.text((5, 4), "Cleaned: green=body; red=protected crutch candidates; yellow=annotation", fill="white")
        draw.text((5, 23), "Projection only: no occlusion certification. " + name, fill="white")
        target = destination / f"overlay-{index:02d}.png"
        overlay.save(target)
        overlays.append({"path": str(target), "sha256": sha256_file(target), "image": name})
    after = records(frozen)
    if before != after:
        raise ValueError("Frozen source hashes changed")
    cloud = validate_colored_ply(destination / "point_cloud.ply")
    receipt = {"status": "validated_pending_visual_review", "created_at": utc_now(), "derived_id": derived_id,
               "raw_experiment": config.experiment, "raw_count": len(xyz), "retained_count": count, "removed_count": len(xyz) - count,
               "body_count": int(body.sum()), "protected_count": int(protected.sum()), "protected_not_body_count": int((protected & ~body).sum()),
               "body_rule": {"minimum_usable_views": 6, "foreground_agreement": .90, "mask_threshold": 128, "pixel_lookup": "floor after native mask scaling"},
               "crutch_rule": "bounded approved A/B capsules AND three annotated corridor views with pairwise rays >=15 degrees; union with body",
               "crutch_regions": {name: int((mask & protected).sum()) for name, mask in capsules.items()},
               "own_mvs_geometry": True, "old_point_ids_used": False, "max_pose_delta": pose_delta,
               "exact_original_vertex_bytes_preserved": True, "source_records_before": before, "source_records_after": after,
               "source_unchanged": True, "point_cloud": cloud, "overlays": overlays,
               "audit_files": records([destination / "selection_audit.npz", destination / "projection_rows.json"]),
               "source_code": records([Path(__file__), helper_path]), "elapsed_seconds": time.monotonic() - start,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cpu_only": True,
               "limitations": ["Silhouette votes are not depth-visibility tests.", "Protected corridor/capsule points can contain background and occluded surfaces.", "No dense 90% geometric accuracy or historical sparse-track quality claim.", "Raw reconstruction and historical outputs unchanged."]}
    atomic_json(destination / "cleanup_receipt.json", receipt)
    atomic_json(destination / "state.json", {"status": receipt["status"], "retained_count": count})
    print(json.dumps({key: receipt[key] for key in ("status", "derived_id", "retained_count", "body_count", "protected_count", "protected_not_body_count", "elapsed_seconds")}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--derived-id", default="dark-e3-mvs-body90-crutches-v1")
    options = parser.parse_args()
    run(options.config, options.derived_id)
