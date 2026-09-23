#!/usr/bin/env python3
"""Prepare image-grounded crutch protection; never edit a reconstruction.

The output remains provisional until its photo overlays are reviewed. Capsule
endpoints are selection parameters, not reconstructed or newly inserted points.
"""
from pathlib import Path
import hashlib
from itertools import combinations
import json
import sys

import numpy as np
from PIL import Image
import pycolmap


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "reconstructions/black_shirt_crutches/subject_preview"
sys.path.insert(0, str(ROOT / "work/quality-sfm"))
from compare_models import image_states, point_measurements


def projected_roi_hits(uv, camera, regions):
    result = np.zeros(len(uv), dtype=bool)
    for record in regions.values():
        minimum = np.full(len(uv), np.inf)
        for normalized in record["polylines"]:
            line = np.asarray(normalized) * [camera.width, camera.height]
            for start, end in zip(line, line[1:]):
                vector = end - start
                if vector @ vector == 0:
                    continue
                t = np.clip((uv - start) @ vector / (vector @ vector), 0, 1)
                minimum = np.minimum(minimum, np.linalg.norm(uv - (start + t[:, None] * vector), axis=1))
        result |= minimum <= record["half_width_fraction_of_image_width"] * camera.width
    return result


def capsule_gate(xyz, capsules):
    result = np.zeros(len(xyz), dtype=bool)
    for capsule in capsules:
        start, end = np.asarray(capsule["start_world"]), np.asarray(capsule["end_world"])
        vector = end - start
        t = np.clip((xyz - start) @ vector / (vector @ vector), 0, 1)
        result |= np.linalg.norm(xyz - (start + t[:, None] * vector), axis=1) <= capsule["radius"]
    return result


def confirmation_gate(model, xyz, definitions, minimum_separation_degrees=15.):
    """Require three exposed annotated views, separated around each 3D point."""
    images = {image.name: image for image in model.images.values()}
    votes, rays, per_view = [], [], []
    for name, rois in definitions.items():
        image = images[name]
        camera, pose = model.cameras[image.camera_id], image.cam_from_world()
        cam = xyz @ pose.rotation.matrix().T + pose.translation
        uv = camera.img_from_cam(np.ascontiguousarray(cam))
        valid = ((cam[:, 2] > 0) & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
        vote = valid & projected_roi_hits(uv, camera, rois)
        votes.append(vote)
        ray = image.projection_center() - xyz
        ray /= np.linalg.norm(ray, axis=1)[:, None]
        rays.append(ray)
        per_view.append({"image": name, "points_inside_visible_crutch_corridors": int(vote.sum())})
    votes, rays = np.asarray(votes), np.asarray(rays)
    separated = np.zeros(len(xyz), dtype=bool)
    cosine = np.cos(np.radians(minimum_separation_degrees))
    for first, second, third in combinations(range(len(definitions)), 3):
        selected = votes[first] & votes[second] & votes[third]
        for left, right in ((first, second), (first, third), (second, third)):
            selected &= np.einsum('ij,ij->i', rays[left], rays[right]) <= cosine
        separated |= selected
    return votes.sum(axis=0), separated, per_view


def fit_axis(cloud, random):
    lower = cloud[(cloud[:, 2] < -2.1) & (cloud[:, 2] > -3.7)]
    best = None
    for _ in range(800):
        first, second = lower[random.choice(len(lower), 2, replace=False)]
        vector = second - first
        length = np.linalg.norm(vector)
        if length < .7:
            continue
        direction = vector / length
        if abs(direction[2]) < .8:
            continue
        distances = np.linalg.norm(np.cross(lower - first, direction), axis=1)
        keep = distances < .065
        score = (len(np.unique(np.floor(lower[keep, 2] / .15))), int(keep.sum()))
        if best is None or score > best[0]:
            best = score, keep
    if best is None or best[0][0] < 5:
        raise ValueError("Insufficient independent shaft-height coverage for a stable axis")
    good = lower[best[1]]
    bins = np.floor(good[:, 2] / .15)
    medians = np.asarray([np.median(good[bins == value], axis=0) for value in np.unique(bins)])
    center = medians.mean(axis=0)
    _, _, vectors = np.linalg.svd(medians - center)
    direction = vectors[0] * np.sign(vectors[0, 2])
    return center, direction, {"lower_seed_points": len(lower), "lower_shaft_inlier_points": len(good),
        "occupied_height_bins": best[0][0], "height_bin_size": .15, "ransac_distance": .065,
        "fit_weighting": "One median per occupied height bin, to avoid a dense foot cluster dominating the axis",
        "axis_point_upright": center.tolist(), "axis_direction_upright": direction.tolist()}


def draw_overlays(path, model, xyz, selected, definitions, regions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(definitions)
    figure, axes = plt.subplots((len(names) + 2) // 3, 3, figsize=(15, 6 * ((len(names) + 2) // 3)), layout="constrained")
    images = {image.name: image for image in model.images.values()}
    for ax, name in zip(axes.flat, names):
        image = images[name]
        camera, pose = model.cameras[image.camera_id], image.cam_from_world()
        photo = Image.open(SOURCE / "images" / name)
        photo.thumbnail((700, 950))
        ax.imshow(photo, extent=(0, 1, 1, 0))
        for record in definitions[name].values():
            for line in record["polylines"]:
                line = np.asarray(line)
                ax.plot(line[:, 0], line[:, 1], color="cyan", linewidth=2, alpha=.7)
        for label, color in (("A", "#ff3b30"), ("B", "#348aff")):
            cam = xyz[selected[label]] @ pose.rotation.matrix().T + pose.translation
            projection = camera.img_from_cam(np.ascontiguousarray(cam)) / [camera.width, camera.height]
            valid = (cam[:, 2] > 0) & np.isfinite(projection).all(axis=1)
            ax.scatter(projection[valid, 0], projection[valid, 1], s=5, c=color, linewidths=0,
                       alpha=.7, label=f"{label}: existing points")
            for capsule in regions[label]["capsules"]:
                endpoints = np.asarray([capsule["start_world"], capsule["end_world"]])
                vector = endpoints[1] - endpoints[0]
                direction = vector / np.linalg.norm(vector)
                trial = np.array([1., 0, 0]) if abs(direction[0]) < .9 else np.array([0., 1, 0])
                normal = np.cross(direction, trial)
                normal /= np.linalg.norm(normal)
                second = np.cross(direction, normal)
                for angle in np.arange(0, 2 * np.pi, np.pi / 2):
                    offset = capsule["radius"] * (np.cos(angle) * normal + np.sin(angle) * second)
                    edge = endpoints + offset
                    projected = camera.img_from_cam(edge @ pose.rotation.matrix().T + pose.translation)
                    projected /= [camera.width, camera.height]
                    ax.plot(projected[:, 0], projected[:, 1], color=color, linewidth=.8, alpha=.65)
        ax.set(xlim=(0, 1), ylim=(1, 0), title=Path(name).name)
        ax.set_aspect(4 / 3)
        ax.legend(fontsize=7, loc="lower left")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    figure.suptitle("Crutch-protection prototype — review before filtering\nCyan: image-grounded shaft guides. Red/blue: existing points and bounded selection regions.", fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    model_path = SOURCE / "colmap/sparse/0"
    model = pycolmap.Reconstruction(model_path)
    source_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in model_path.iterdir() if p.is_file()}
    analysis = json.loads((SOURCE / "analysis.json").read_text())
    basis = np.asarray(analysis["display_orientation"]["world_to_display_row_matrix"])
    origin = np.asarray(analysis["plot"]["display_origin_world"])
    ids = np.asarray(sorted(model.points3D), dtype=np.uint64)
    xyz = np.asarray([model.points3D[int(i)].xyz for i in ids])
    upright = (xyz - origin) @ basis
    source = json.loads((HERE / "seed_image_rois.json").read_text())
    definitions = dict(json.loads((HERE / "confirmation_image_rois.json").read_text())["views"])
    additional_path = HERE / "additional_rois_normalized.json"
    if additional_path.is_file():
        definitions.update(json.loads(additional_path.read_text())["views"])
    images = {image.name: image for image in model.images.values()}
    seed = {label: np.ones(len(ids), dtype=bool) for label in ("A", "B")}
    support, separated_support, per_view = confirmation_gate(model, xyz, definitions)
    for name, rois in source["views"].items():
        image = images[name]
        camera, pose = model.cameras[image.camera_id], image.cam_from_world()
        cam = xyz @ pose.rotation.matrix().T + pose.translation
        uv = camera.img_from_cam(np.ascontiguousarray(cam))
        valid = (cam[:, 2] > 0) & np.isfinite(uv).all(axis=1)
        for label in seed:
            seed[label] &= valid & projected_roi_hits(uv, camera, {label: rois[label]})
    regions = {}
    random = np.random.default_rng(12)
    for label in ("A", "B"):
        cloud = upright[seed[label]]
        center, direction, fit = fit_axis(cloud, random)
        at_height = lambda value: center + (value - center[2]) * direction / direction[2]
        grip = np.median(cloud[(cloud[:, 2] > -1.55) & (cloud[:, 2] < -.65)], axis=0)
        top = np.median(cloud[cloud[:, 2] >= np.quantile(cloud[:, 2], .95)], axis=0)
        capsules = []
        for part, start, end, radius in (
            ("lower_shaft_and_tip", at_height(float(cloud[:, 2].min()) - .015), at_height(-2.12), .105),
            ("lower_fork_to_grip", at_height(-2.12), grip, .18),
            ("upper_support", grip, top, .17)):
            capsules.append({"part": part, "start_world": (start @ basis.T + origin).tolist(),
                "end_world": (end @ basis.T + origin).tolist(), "radius": radius})
        regions[label] = {"identity": f"Image-{'left' if label == 'A' else 'right'} crutch in front reference0076",
            "seed_points_in_both_front_corridors": int(seed[label].sum()), "axis_fit": fit, "capsules": capsules}
    measurements = point_measurements(model, image_states(model))
    quality = ((measurements["distinct_views"] >= 3) & (measurements["normalized"] <= 3)
        & (measurements["angles"] >= 1.5) & measurements["all_observations_valid"])
    selected = {label: capsule_gate(xyz, region["capsules"]) & separated_support & quality
                for label, region in regions.items()}
    for label, keep in selected.items():
        regions[label]["candidate_source_point_ids"] = ids[keep].tolist()
        regions[label]["candidate_points"] = int(keep.sum())
    report = {"status": "prototype_requires_root_photo_overlay_review", "ready_for_filter": False,
        "source_model": str(model_path), "source_model_sha256": source_files,
        "coordinate_frame": "Original baseline world coordinates; no alignment or geometry creation",
        "regions": regions, "annotated_image_corridors": definitions, "view_support": per_view,
        "selection": {"minimum_annotated_views": 3, "minimum_distinct_track_views": 3,
            "minimum_pairwise_camera_ray_separation_degrees_for_confirming_triple": 15.,
            "visibility": "Only explicitly drawn exposed corridors can vote; occluded hand/shirt/leg gaps are omitted. Broad fit seed regions are not used as positive confirmation.",
            "max_mean_baseline_pixel_error": 3, "minimum_triangulation_angle_degrees": 1.5,
            "required": "Existing point lies in a bounded capsule and at least three image-grounded crutch corridors"},
        "limitations": ["Capsules are protection regions, not estimated solid crutch surfaces or added model points.",
            "Smooth metal shaft sections have sparse or missing SfM points; this cannot restore absent geometry.",
            "Grip/underarm regions can include adjacent hand/body points, and contact points near the floor remain ambiguous.",
            "Final cleanup must combine verified crutch protection with body-mask agreement; body masks alone omit shafts."],
        "candidate_union_points": int(np.logical_or.reduce(list(selected.values())).sum())}
    (HERE / "protection.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(HERE / "candidate_existing_points.npz", source_point_ids=ids, xyz=xyz,
        crutch_A=selected["A"], crutch_B=selected["B"], annotated_view_support=support,
        three_separated_confirming_views=separated_support)
    draw_overlays(HERE / "projection_qc.png", model, xyz, selected, definitions, regions)
    for name, digest in source_files.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source model changed during the read-only prototype")
    print(json.dumps({"status": report["status"], "views": len(definitions),
        "A": regions["A"]["candidate_points"], "B": regions["B"]["candidate_points"],
        "union": report["candidate_union_points"]}))


if __name__ == "__main__":
    main()
