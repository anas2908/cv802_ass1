#!/usr/bin/env python3
"""Read-only dense-geometry crutch plausibility review, NOT a cleanup algorithm.

Historical sparse point IDs/feature tracks are never used for new dense points.
We recompute bounded-capsule membership and three separated annotated-corridor
projections on the new XYZ. These are geometric candidate checks, not proof of
pixel visibility, true crutch surface accuracy, or the old sparse quality gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv802_mvs.config import MVSConfig
from cv802_mvs.io_utils import atomic_json, sha256_file, utc_now
from cv802_mvs.paths import PathPolicy, safe_experiment_name
from cv802_mvs.validation import validate_colored_ply


def read_dense_xyz(path: Path):
    """Read every XYZ row without interpreting vertex indices as SfM point IDs."""
    import numpy as np
    scalar = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
              "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4"}
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Expected PLY")
        fields, count, binary, active = [], None, False, None
        for _ in range(256):
            words = stream.readline().decode("ascii").split()
            if words == ["end_header"]:
                break
            if words[:1] == ["format"]:
                binary = words[1] == "binary_little_endian"
            elif words[:1] == ["element"]:
                active = words[1]
                if active == "vertex":
                    count = int(words[2])
            elif words[:1] == ["property"] and active == "vertex":
                if len(words) != 3 or words[1] not in scalar:
                    raise ValueError("Unsupported dense scalar PLY schema")
                fields.append((words[2], "<" + scalar[words[1]]))
        else:
            raise ValueError("Missing PLY header terminator")
        if not binary or not count or not {"x", "y", "z"} <= set(dict(fields)):
            raise ValueError("Expected nonempty binary little-endian XYZ PLY")
        dtype = np.dtype(fields)
        if path.stat().st_size - stream.tell() < count * dtype.itemsize:
            raise ValueError("Truncated dense vertex array")
        rows = np.memmap(path, mode="r", dtype=dtype, offset=stream.tell(), shape=(count,))
        xyz = np.column_stack([rows[name] for name in ("x", "y", "z")]).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite dense points")
    return xyz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review-id", default="dark-e3-1024-crutch-projections-v1")
    args = parser.parse_args()
    import numpy as np
    import pycolmap
    from PIL import Image, ImageDraw

    policy = PathPolicy.production()
    config = MVSConfig.load(args.config, policy)
    paths = config.paths(policy)
    if config.masking_mode != "none":
        raise ValueError("This review is scoped to the raw unmasked dark result")
    state = json.loads((paths.root / "run_state.json").read_text())
    final_path = paths.manifests / "final_validation.json"
    final = json.loads(final_path.read_text())
    if state.get("status") != "complete" or final["config_digest"] != config.digest:
        raise ValueError("Wait for the controller to publish the completed configured run")
    before = {str(path): sha256_file(path) for path in (final_path, paths.fused)}
    cloud = validate_colored_ply(paths.fused)
    if cloud != final["colored_dense_point_cloud"]:
        raise ValueError("Dense PLY no longer agrees with the completed receipt")
    output = policy.require_method(
        policy.method_root / "reviews" / safe_experiment_name(args.review_id),
        "new review", must_exist=False,
    )
    crutch_root = paths.root / "inputs/provenance/crutches"
    protection_path = crutch_root / "protection_approved.rebased.json"
    protection = json.loads(protection_path.read_text())
    for path, expected in protection["source_model_sha256"].items():
        source = policy.require_method(path, "approved reference", must_exist=True)
        if sha256_file(source) != expected:
            raise ValueError("Copied approved reference hash changed")
    original = pycolmap.Reconstruction(str(paths.model))
    dense_model = pycolmap.Reconstruction(str(paths.workspace / "sparse"))
    source_images = {image.name: image for image in original.images.values()}
    dense_images = {image.name: image for image in dense_model.images.values()}
    if source_images.keys() != dense_images.keys():
        raise ValueError("Dense model registered names differ from original calibration")
    pose_delta = max(float(np.max(np.abs(image.cam_from_world().matrix() - dense_images[name].cam_from_world().matrix())))
                     for name, image in source_images.items())
    if pose_delta > 1e-10:
        raise ValueError("Dense output world frame changed; approved capsules need transformation")
    helper_path = Path(__file__).resolve().parents[1] / "legacy_tools/reconstructions/experiments/black_person_crutches_cleanup/crutch_protection/build_protection.py"
    spec = importlib.util.spec_from_file_location("mvs_crutch_geometry", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    xyz = read_dense_xyz(paths.fused)
    capsule_masks = {name: helper.capsule_gate(xyz, region["capsules"])
                     for name, region in protection["regions"].items()}
    # Restrict projection work to the geometrically bounded candidate union.
    candidate_rows = np.flatnonzero(np.logical_or.reduce(list(capsule_masks.values())))
    candidate_xyz = xyz[candidate_rows]
    support, separated, per_view = helper.confirmation_gate(
        original, candidate_xyz, protection["annotated_image_corridors"],
        protection["selection"]["minimum_pairwise_camera_ray_separation_degrees_for_confirming_triple"],
    )
    labels = {name: mask[candidate_rows] & separated for name, mask in capsule_masks.items()}
    counts = {name: {"inside_capsules": int(mask.sum()),
                     "three_separated_corridor_candidates": int(labels[name].sum())}
              for name, mask in capsule_masks.items()}
    output.mkdir(parents=True, exist_ok=False)
    overlays = []
    colors = {"A": (255, 80, 40), "B": (30, 210, 255)}
    for index, (name, corridors) in enumerate(protection["annotated_image_corridors"].items()):
        image = source_images[name]
        camera, pose = original.cameras[image.camera_id], image.cam_from_world()
        with Image.open(paths.images / name) as source:
            photo = source.convert("RGB")
        scale = min(1.0, 1400 / max(photo.size))
        size = tuple(round(value * scale) for value in photo.size)
        photo = photo.resize(size, Image.Resampling.LANCZOS)
        pixels = np.asarray(photo).copy()
        cam = candidate_xyz @ pose.rotation.matrix().T + pose.translation
        uv = camera.img_from_cam(np.ascontiguousarray(cam))
        finite = np.isfinite(uv).all(axis=1) & (cam[:, 2] > 0)
        finite &= (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
        visible_projection_counts = {}
        for label, selected in labels.items():
            usable = selected & finite
            visible_projection_counts[label] = int(usable.sum())
            xy = np.floor(uv[usable] * scale).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, size[0] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, size[1] - 1)
            pixels[xy[:, 1], xy[:, 0]] = colors[label]
        overlay = Image.fromarray(pixels)
        draw = ImageDraw.Draw(overlay)
        for corridor in corridors.values():
            for line in corridor["polylines"]:
                draw.line([(round(x * size[0]), round(y * size[1])) for x, y in line], fill=(255, 220, 30), width=2)
        draw.rectangle((0, 0, size[0], 55), fill=(0, 0, 0))
        draw.text((6, 4), "NEW dense candidates: red=A, cyan=B; yellow=historical corridor axis", fill="white")
        draw.text((6, 21), "Projection plausibility only; occlusion and dense quality gates NOT certified", fill="white")
        draw.text((6, 38), name, fill="white")
        destination = output / f"crutch-overlay-{index:02d}.png"
        overlay.save(destination)
        overlays.append({"source_image": name, "path": str(destination), "sha256": sha256_file(destination),
                         "projected_candidates": visible_projection_counts})
    for path, digest in before.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("Completed reconstruction changed during read-only review")
    report = {
        "status": "projection_evidence_created", "created_at": utc_now(),
        "experiment": config.experiment, "full_dense_points_examined": len(xyz),
        "source_ply": cloud, "source_final_validation_sha256": before[str(final_path)],
        "source_protection": str(protection_path), "source_protection_sha256": sha256_file(protection_path),
        "helper_source_sha256": sha256_file(helper_path), "new_xyz_used": True,
        "historical_point_ids_used": False, "max_undistorted_pose_difference": pose_delta,
        "crutches": counts, "capsule_union_candidates": len(candidate_rows),
        "three_separated_view_candidates": int(separated.sum()), "per_view": per_view,
        "overlays": overlays, "review_status": "awaiting_visual_review",
        "cleaned_dense_cloud_created": False,
        "limitations": ["Candidate support is recomputed on new dense XYZ, never historical point IDs.",
                        "Three corridor projections do not prove actual visibility or crutch-surface accuracy.",
                        "No dense depth/occlusion support gate or historical sparse-track error/angle filter was applied.",
                        "Raw fused cloud and completed experiment receipts were not modified."],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(output / "review.json", report)
    print(json.dumps({"status": report["status"], "points": len(xyz), "crutches": counts,
                      "review": str(output / "review.json"), "overlays": len(overlays)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
