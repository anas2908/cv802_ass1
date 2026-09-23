#!/usr/bin/env python3
"""Read-only geometry, feature-database and checkpoint audit of a quality model."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from compare_models import image_states, model_at, point_measurements, quantiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-database", action="store_true", help="Audit geometry only")
    args = parser.parse_args()
    baseline_path, baseline = model_at(args.baseline)
    model_path, model = model_at(args.candidate)
    reference, states = image_states(baseline), image_states(model)
    if set(reference) != set(states):
        raise ValueError("Fixed-pose model does not contain the baseline image set")
    measure = point_measurements(model, reference)
    pose_delta, nonfinite, orthogonal, determinants, id_changes = [], [], [], [], []
    groups = defaultdict(list)
    for name, state in states.items():
        first = reference[name]
        pose_delta.append(max(float(np.max(np.abs(state[k] - first[k])))
                              for k in ("rotation", "translation")))
        if not all(np.isfinite(state[k]).all() for k in ("rotation", "translation")):
            nonfinite.append(name)
        orthogonal.append(float(np.max(np.abs(state["rotation"].T @ state["rotation"] - np.eye(3)))))
        determinants.append(float(np.linalg.det(state["rotation"])))
        if state["image_id"] != first["image_id"]:
            id_changes.append(name)
        groups[Path(name).parts[0]].append(model.images[state["image_id"]].num_points3D)
    cameras = []
    for camera_id, camera in sorted(model.cameras.items()):
        original = baseline.cameras[camera_id]
        expected = pycolmap.Camera(original.todict())
        expected.rescale(camera.width, camera.height)
        cameras.append({"camera_id": camera_id, "model": str(camera.model),
            "width": camera.width, "height": camera.height, "params": camera.params.tolist(),
            "finite": bool(np.isfinite(camera.params).all()),
            "focal_length_positive": bool(camera.focal_length > 0),
            "size_scaling": [camera.width / original.width, camera.height / original.height],
            "max_parameter_difference_from_rescaled_baseline": float(np.max(np.abs(camera.params - expected.params))),
            "model_matches_baseline": camera.model == original.model})
    by_image = defaultdict(list)
    point_ids_by_image = defaultdict(set)
    duplicate_track_images = invalid_links = 0
    for point_id, point in model.points3D.items():
        seen = set()
        for obs in point.track.elements:
            duplicate_track_images += int(obs.image_id in seen)
            seen.add(obs.image_id)
            image = model.images[obs.image_id]
            if obs.point2D_idx >= len(image.points2D) or image.point2D(obs.point2D_idx).point3D_id != point_id:
                invalid_links += 1
            by_image[obs.image_id].append((point.xyz, obs.point2D_idx))
            point_ids_by_image[obs.image_id].add(int(point_id))
    negative_depth = nonfinite_depth = 0
    depths, observation_errors, baseline_errors = [], [], []
    for image_id, rows in by_image.items():
        image = model.images[image_id]
        camera, pose = model.cameras[image.camera_id], image.cam_from_world()
        xyz = np.asarray([r[0] for r in rows])
        camxyz = xyz @ pose.rotation.matrix().T + pose.translation
        z = camxyz[:, 2]
        negative_depth += int((z <= 0).sum())
        nonfinite_depth += int((~np.isfinite(z)).sum())
        depths.extend(z)
        observed = np.asarray([image.point2D(r[1]).xy for r in rows])
        projected = camera.img_from_cam(np.ascontiguousarray(camxyz))
        delta, base = projected - observed, reference[image.name]
        observation_errors.extend(np.linalg.norm(delta, axis=1))
        baseline_errors.extend(np.linalg.norm(delta * np.array(
            [base["width"] / camera.width, base["height"] / camera.height]), axis=1))
    image_support = [{"image_id": int(image_id), "name": model.images[image_id].name,
        "camera_id": int(model.images[image_id].camera_id),
        "distinct_3D_points": len(point_ids_by_image[image_id]),
        "track_observations": len(by_image[image_id])}
        for image_id in sorted(model.reg_image_ids(), key=lambda i: model.images[i].name)]
    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": str(baseline_path), "quality_model": str(model_path),
        "scope": "Numerical and data-integrity checks only. More points and inherited camera coverage do not establish improved body geometry. Human motion remains a limitation.",
        "registered_images": model.num_reg_images(), "points3D": model.num_points3D(),
        "all_point_coordinates_finite": bool(np.isfinite(measure["xyz"]).all()),
        "all_stored_point_errors_finite_nonnegative": bool((np.isfinite(measure["stored_native"]) & (measure["stored_native"] >= 0)).all()),
        "camera_invariance": {"same_registered_names": True, "changed_image_ids": id_changes,
            "max_pose_element_difference": max(pose_delta), "unchanged_within_1e_minus_10": max(pose_delta) <= 1e-10,
            "nonfinite_pose_images": nonfinite, "max_rotation_orthogonality_error": max(orthogonal),
            "rotation_determinant_range": [min(determinants), max(determinants)], "cameras": cameras},
        "tracks": {"observations": int(measure["tracks"].sum()), "lengths": quantiles(measure["tracks"]),
            "distinct_image_views": quantiles(measure["distinct_views"]),
            "two_observation_points": int((measure["tracks"] == 2).sum()),
            "two_view_points": int((measure["distinct_views"] == 2).sum()),
            "points_with_at_least_3_observations": int((measure["tracks"] >= 3).sum()),
            "points_with_at_least_3_views": int((measure["distinct_views"] >= 3).sum()),
            "points_with_at_least_5_views": int((measure["distinct_views"] >= 5).sum()),
            "points_with_track3_but_fewer_than_3_views": int(((measure["tracks"] >= 3) & (measure["distinct_views"] < 3)).sum()),
            "duplicate_image_observations": duplicate_track_images, "invalid_reciprocal_point_links": invalid_links,
            "observations_behind_camera": negative_depth, "nonfinite_observation_depths": nonfinite_depth,
            "depth_quantiles_arbitrary_scale": quantiles(depths),
            "invalid_reprojection_observations": measure["invalid_observations"],
            "max_acute_triangulation_angle_degrees": quantiles(measure["angles"]),
            "points_below_1_5_degree_max_angle": int((measure["angles"] < 1.5).sum())},
        "reprojection": {"per_point_native_pixels": quantiles(measure["native"]),
            "per_point_baseline_equivalent_pixels": quantiles(measure["normalized"]),
            "per_observation_native_pixels": quantiles(observation_errors),
            "per_observation_baseline_equivalent_pixels": quantiles(baseline_errors),
            "max_stored_vs_recomputed_mean_error_difference": float(np.max(np.abs(measure["stored_native"] - measure["native"]))),
            "points_above_3px_baseline_equivalent": int((measure["normalized"] > 3).sum()),
            "points_passing_distinct_views3_error3_angle1_5_all_observations_valid": int(((measure["distinct_views"] >= 3) & (measure["normalized"] <= 3) & (measure["angles"] >= 1.5) & measure["all_observations_valid"]).sum())},
        "image_point_support": {"contributing_images": sum(row["distinct_3D_points"] > 0 for row in image_support),
            "zero_support_images": [row["name"] for row in image_support if row["distinct_3D_points"] == 0],
            "per_image": image_support},
        "group_coverage": {name: {"registered": len(values), "observations_per_image": quantiles(values)}
                           for name, values in sorted(groups.items())}}
    for group, summary in report["group_coverage"].items():
        rows = [row for row in image_support if Path(row["name"]).parts[0] == group]
        summary["contributing_images"] = sum(row["distinct_3D_points"] > 0 for row in rows)
        summary["zero_support_images"] = [row["name"] for row in rows if row["distinct_3D_points"] == 0]
        summary["distinct_3D_points_per_image"] = quantiles([row["distinct_3D_points"] for row in rows])
        summary["distinct_3D_points_across_group"] = len(set().union(
            *(point_ids_by_image[row["image_id"]] for row in rows)))
    report["geometric_integrity_passed"] = all((
        report["all_point_coordinates_finite"], report["all_stored_point_errors_finite_nonnegative"],
        not nonfinite, not id_changes, max(pose_delta) <= 1e-10,
        max(orthogonal) <= 1e-10, max(abs(value - 1) for value in determinants) <= 1e-10,
        all(c["finite"] and c["focal_length_positive"] and c["model_matches_baseline"]
            and c["max_parameter_difference_from_rescaled_baseline"] <= 1e-10 for c in cameras),
        invalid_links == 0, negative_depth == 0, nonfinite_depth == 0,
        measure["invalid_observations"] == 0,
        report["reprojection"]["max_stored_vs_recomputed_mean_error_difference"] <= 1e-6,
    ))
    if not args.skip_database:
        from audit_database import audit_database
        report["database_and_feature_checkpoint"] = audit_database(args.baseline.resolve(), args.candidate.resolve())
    report["passed"] = report["geometric_integrity_passed"] and report.get("database_and_feature_checkpoint", {}).get("passed", True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("registered_images", "points3D", "passed", "geometric_integrity_passed", "tracks", "reprojection")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
