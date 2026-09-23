#!/usr/bin/env python3
"""Compare two already subject-filtered COLMAP models without modifying them.

Example:
    .venv/bin/python work/quality-sfm/compare_models.py BASELINE_PREVIEW \
        CANDIDATE_PREVIEW --output comparison.json

Both inputs must use the same subject-selection policy and world coordinates.
The baseline preview's analysis.json supplies one fixed upright frame and body
volume for both models. This is a consistency/coverage comparison, not a measure
of anatomical accuracy. Neither a model alignment nor subject detection is run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pycolmap


def model_at(source):
    source = source.resolve()
    search = [source, source / "colmap" / "sparse", source / "sparse"]
    candidates = set()
    for folder in search:
        if not folder.is_dir():
            continue
        for path in [folder, *sorted(p for p in folder.iterdir() if p.is_dir())]:
            if any(all((path / (stem + ext)).is_file()
                       for stem in ("cameras", "images", "points3D"))
                   for ext in (".bin", ".txt")):
                candidates.add(path)
    if not candidates:
        raise ValueError(f"No complete COLMAP model found under {source}")
    loaded = [(path, pycolmap.Reconstruction(path)) for path in sorted(candidates)]
    return max(loaded, key=lambda pair: (pair[1].num_reg_images(), pair[1].num_points3D()))


def image_states(model):
    result = {}
    for image_id in model.reg_image_ids():
        image = model.images[image_id]
        pose = image.cam_from_world()
        rotation = np.asarray(pose.rotation.matrix())
        translation = np.asarray(pose.translation)
        camera = model.cameras[image.camera_id]
        result[image.name] = {
            "image_id": image_id, "rotation": rotation,
            "translation": translation, "center": -rotation.T @ translation,
            "width": camera.width, "height": camera.height,
        }
    return result


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    qs = [0, .5, .9, .95, .99, 1]
    return {"count": len(values), "mean": float(values.mean()),
            **{str(q): float(v) for q, v in zip(qs, np.quantile(values, qs))}}


def point_measurements(model, baseline_states):
    """Reproject observations; normalize each residual to its baseline image size."""
    ids = sorted(model.points3D.keys())
    points = [model.points3D[i] for i in ids]
    xyz = np.asarray([point.xyz for point in points], dtype=float).reshape(-1, 3)
    states = image_states(model)
    state_by_id = {state["image_id"]: state for state in states.values()}
    observations = defaultdict(list)
    tracks = np.asarray([point.track.length() for point in points], dtype=int)
    distinct_views = np.asarray([
        len({element.image_id for element in point.track.elements}) for point in points
    ], dtype=int)
    angles = np.zeros(len(points))
    for index, point in enumerate(points):
        centers = []
        for element in point.track.elements:
            image = model.images[element.image_id]
            observations[element.image_id].append((index, element.point2D_idx))
            if element.image_id in state_by_id:
                centers.append(state_by_id[element.image_id]["center"])
        if len(centers) >= 2 and np.isfinite(xyz[index]).all():
            rays = np.asarray(centers) - xyz[index]
            lengths = np.linalg.norm(rays, axis=1)
            rays = rays[np.isfinite(lengths) & (lengths > 1e-12)]
            if len(rays) >= 2:
                rays /= np.linalg.norm(rays, axis=1)[:, None]
                # The acute angle rejects both coincident and antiparallel rays.
                cosine = float(np.min(np.abs(rays @ rays.T)))
                angles[index] = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
    native_sums = np.zeros(len(points))
    normalized_sums = np.zeros(len(points))
    good_counts = np.zeros(len(points), dtype=int)
    invalid_observations = missing_reference = 0
    for image_id, rows in observations.items():
        image = model.images[image_id]
        if image.name not in baseline_states or image_id not in state_by_id:
            missing_reference += len(rows)
            continue
        state = state_by_id[image_id]
        reference = baseline_states[image.name]
        indices = np.asarray([row[0] for row in rows], dtype=int)
        observed = np.asarray([image.point2D(row[1]).xy for row in rows])
        camera_xyz = xyz[indices] @ state["rotation"].T + state["translation"]
        valid = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 1e-8)
        valid &= np.isfinite(observed).all(axis=1)
        projected = np.full((len(rows), 2), np.nan)
        if valid.any():
            projected[valid] = model.cameras[image.camera_id].img_from_cam(
                np.ascontiguousarray(camera_xyz[valid]))
        valid &= np.isfinite(projected).all(axis=1)
        invalid_observations += int((~valid).sum())
        delta = projected[valid] - observed[valid]
        scale = np.array([reference["width"] / state["width"],
                          reference["height"] / state["height"]])
        np.add.at(native_sums, indices[valid], np.linalg.norm(delta, axis=1))
        np.add.at(normalized_sums, indices[valid], np.linalg.norm(delta * scale, axis=1))
        np.add.at(good_counts, indices[valid], 1)
    native = np.divide(native_sums, good_counts, out=np.full(len(points), np.inf), where=good_counts > 0)
    normalized = np.divide(normalized_sums, good_counts, out=np.full(len(points), np.inf), where=good_counts > 0)
    return {"xyz": xyz, "tracks": tracks, "distinct_views": distinct_views, "angles": angles, "native": native,
            "normalized": normalized, "all_observations_valid": good_counts == tracks,
            "stored_native": np.asarray([point.error for point in points]),
            "invalid_observations": invalid_observations,
            "observations_without_baseline_image": missing_reference}


def voxel_set(display, low, high, side):
    if not len(display):
        return set()
    indices = np.floor((display - low) / side).astype(np.int64)
    maximum = np.maximum(0, np.ceil((high - low) / side).astype(np.int64) - 1)
    indices = np.clip(indices, 0, maximum)
    return {tuple(int(v) for v in row) for row in indices}


def summarize(model, reference, basis, origin, low, high, args):
    data = point_measurements(model, reference)
    display = (data["xyz"] - origin) @ basis
    finite = np.isfinite(display).all(axis=1)
    long_tracks = data["distinct_views"] >= args.min_track
    quality = finite & long_tracks & data["all_observations_valid"]
    quality &= np.isfinite(data["normalized"]) & (data["normalized"] <= args.max_error)
    quality &= data["angles"] >= args.min_angle
    inside = np.all((display >= low - 1e-10) & (display <= high + 1e-10), axis=1)
    keep = quality & inside
    retained = display[keep]
    height = float(high[2] - low[2])
    normalized_height = (retained[:, 2] - low[2]) / height
    quartiles = np.minimum((np.clip(normalized_height, 0, 1) * 4).astype(int), 3)
    grids = {f"height/{resolution}": voxel_set(retained, low, high, height / resolution)
             for resolution in (100, 50)}
    metrics = {
        "registered_images": model.num_reg_images(), "input_subject_points": len(display),
        "points_with_track_at_least_minimum": int((data["tracks"] >= args.min_track).sum()),
        "points_with_distinct_views_at_least_minimum": int(long_tracks.sum()),
        "points_passing_quality_before_fixed_bounds": int(quality.sum()),
        "quality_points_outside_fixed_baseline_bounds": int((quality & ~inside).sum()),
        "retained_comparable_points": int(keep.sum()),
        "invalid_or_behind_camera_observations": data["invalid_observations"],
        "observations_without_baseline_image": data["observations_without_baseline_image"],
        "stored_native_point_error_all": quantiles(data["stored_native"]),
        "recomputed_native_mean_point_error_retained": quantiles(data["native"][keep]),
        "baseline_pixel_mean_point_error_retained": quantiles(data["normalized"][keep]),
        "track_length_retained": quantiles(data["tracks"][keep]),
        "distinct_image_views_retained": quantiles(data["distinct_views"][keep]),
        "maximum_acute_triangulation_angle_degrees_retained": quantiles(data["angles"][keep]),
        "occupied_voxels": {key: len(value) for key, value in grids.items()},
        "vertical_quartiles_bottom_to_top": [],
    }
    for number in range(4):
        subset = retained[quartiles == number]
        metrics["vertical_quartiles_bottom_to_top"].append({
            "normalized_height_interval": [number / 4, (number + 1) / 4],
            "points": len(subset),
            "occupied_voxels": {f"height/{resolution}": len(voxel_set(subset, low, high, height / resolution))
                                for resolution in (100, 50)},
        })
    return metrics, grids


def compare(args):
    baseline_path, baseline = model_at(args.baseline_preview)
    candidate_path, candidate = model_at(args.candidate_preview)
    analysis_path = args.baseline_analysis
    if analysis_path is None:
        analysis_path = next((p / "analysis.json" for p in [args.baseline_preview.resolve(),
                              baseline_path, *baseline_path.parents]
                              if (p / "analysis.json").is_file()), None)
    if analysis_path is None:
        raise ValueError("Baseline analysis.json is required; pass --baseline-analysis.")
    analysis = json.loads(analysis_path.read_text())
    basis = np.asarray(analysis["display_orientation"]["world_to_display_row_matrix"], dtype=float)
    origin = np.asarray(analysis["plot"]["display_origin_world"], dtype=float)
    bounds = analysis["plot"]["upright_bounds"]
    low, high = np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
    if basis.shape != (3, 3) or not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8):
        raise ValueError("Baseline upright basis is not finite and orthonormal")
    if not np.isfinite([origin, low, high]).all() or not np.all(high > low):
        raise ValueError("Baseline body bounds/origin are invalid")
    first, second = image_states(baseline), image_states(candidate)
    common = sorted(first.keys() & second.keys())
    if len(common) < 2:
        raise ValueError("At least two registered image names must match between models")
    pose_deltas = [max(float(np.max(np.abs(first[n]["rotation"] - second[n]["rotation"]))),
                       float(np.max(np.abs(first[n]["translation"] - second[n]["translation"])))) for n in common]
    center_deltas = [float(np.linalg.norm(first[n]["center"] - second[n]["center"])) for n in common]
    exact = all(np.array_equal(first[n][k], second[n][k]) for n in common for k in ("rotation", "translation"))
    unchanged = max(pose_deltas) <= 1e-10
    baseline_metrics, baseline_grids = summarize(baseline, first, basis, origin, low, high, args)
    candidate_metrics, candidate_grids = summarize(candidate, first, basis, origin, low, high, args)
    return {
        "baseline_model": str(baseline_path), "candidate_model": str(candidate_path),
        "baseline_analysis": str(analysis_path.resolve()),
        "method": {
            "min_track_length": args.min_track, "max_mean_reprojection_error_baseline_pixels": args.max_error,
            "min_distinct_image_views": args.min_track,
            "track_gate": "Require this many distinct image IDs; repeated features in one image do not count as independent views. Raw track statistics remain available separately.",
            "min_maximum_acute_triangulation_angle_degrees": args.min_angle,
            "pixel_normalization": "Each observation residual is scaled independently in x/y to the same named baseline image's camera width/height, then norms are averaged per 3D point. Baseline photo pixels are typically at a 3200-pixel long edge; video resolution stays at its own baseline size.",
            "bounds": "Fixed baseline subject-preview min/max in its saved upright display frame; candidate points outside are counted separately.",
            "origin_world": origin.tolist(), "world_to_upright_row_matrix": basis.tolist(),
            "upright_min": low.tolist(), "upright_max": high.tolist(), "body_height_arbitrary_units": float(high[2] - low[2]),
            "voxel_sizes": {f"height/{n}": float((high[2] - low[2]) / n) for n in (100, 50)},
        },
        "camera_pose_invariance": {"common_images": len(common), "baseline_only_images": sorted(first.keys() - second.keys()),
            "candidate_only_images": sorted(second.keys() - first.keys()), "exact_float_array_equality": exact,
            "unchanged_within_1e_minus_10": unchanged, "max_pose_element_absolute_difference": max(pose_deltas),
            "max_camera_center_distance": max(center_deltas)},
        "baseline": baseline_metrics, "candidate": candidate_metrics,
        "voxel_comparison": {key: {"baseline": len(baseline_grids[key]), "candidate": len(candidate_grids[key]),
            "shared": len(baseline_grids[key] & candidate_grids[key]),
            "candidate_only": len(candidate_grids[key] - baseline_grids[key]),
            "baseline_only": len(baseline_grids[key] - candidate_grids[key])} for key in baseline_grids},
        "warnings": ([] if unchanged else ["Camera poses changed; same-frame assumption requires independent review. No alignment was applied."]),
        "limitations": ["Inputs must already use the same subject/volume filtering. This script does not rerun detection or masking.",
            "More occupied voxels can also reflect noise; inspect matching views for duplicate surfaces and anatomical shape.",
            "Camera registration is inherited when poses are fixed and is not an improvement metric.",
            "Baseline bounds may omit newly recovered body extremities; outside counts are reported, not scored as coverage.",
            "No ground-truth geometry is available; these measures do not certify body-shape accuracy."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_preview", type=Path)
    parser.add_argument("candidate_preview", type=Path)
    parser.add_argument("--baseline-analysis", type=Path)
    parser.add_argument("--output", type=Path, help="JSON file; defaults to standard output")
    parser.add_argument("--min-track", type=int, default=3, help="Minimum distinct image IDs observing each point")
    parser.add_argument("--max-error", type=float, default=3.)
    parser.add_argument("--min-angle", type=float, default=1.5)
    args = parser.parse_args()
    if args.min_track < 2 or not np.isfinite(args.max_error) or args.max_error <= 0 or not 0 < args.min_angle <= 90:
        parser.error("Invalid track, error or triangulation-angle threshold")
    report = compare(args)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
