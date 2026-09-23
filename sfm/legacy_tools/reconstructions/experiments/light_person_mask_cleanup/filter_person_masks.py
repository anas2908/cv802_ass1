#!/usr/bin/env python3
"""Remove existing SfM points using calibrated foreground-mask consensus.

Each variant is a new, independent GUI-readable dataset. Source models, images,
and masks are read only. Run only after the masks have passed visual review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
from PIL import Image
import pycolmap


HERE = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_foreground(foreground, uv, tolerance):
    """Sample a binary raster with a small disk tolerance, without changing it."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    if not np.isfinite(uv).all() or tolerance < 0:
        raise ValueError("Mask samples and tolerance must be finite and valid")
    height, width = foreground.shape
    x, y = np.floor(uv).astype(np.int64).T
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValueError("Mask samples must already be inside the image")
    found = np.zeros(len(uv), dtype=bool)
    radius = math.ceil(tolerance)
    offsets = sorted((dx * dx + dy * dy, dy, dx)
        for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= tolerance * tolerance + 1e-12)
    for _, dy, dx in offsets:
        remaining = np.flatnonzero(~found)
        if not len(remaining):
            break
        xx, yy = x[remaining] + dx, y[remaining] + dy
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        indices = remaining[valid]
        found[indices] = foreground[yy[valid], xx[valid]]
    return found


def choose_points(foreground_counts, in_frame_counts, distinct_views, reliable_count,
                  consensus, min_positive_view_fraction=.2, min_track_views=3):
    foreground_counts = np.asarray(foreground_counts)
    in_frame_counts = np.asarray(in_frame_counts)
    distinct_views = np.asarray(distinct_views)
    minimum = math.ceil(min_positive_view_fraction * reliable_count)
    fraction = np.divide(foreground_counts, in_frame_counts,
        out=np.zeros_like(foreground_counts, dtype=float), where=in_frame_counts > 0)
    return ((in_frame_counts > 0) & (foreground_counts >= minimum)
            & (fraction + 1e-12 >= consensus) & (distinct_views >= min_track_views))


def load_model(source):
    sparse = source / "colmap/sparse"
    candidates = [sparse] + sorted(p for p in sparse.iterdir() if p.is_dir())
    models = [(p, pycolmap.Reconstruction(p)) for p in candidates
        if any(all((p / (stem + ext)).is_file() for stem in ("cameras", "images", "points3D"))
               for ext in (".bin", ".txt"))]
    if not models:
        raise ValueError(f"No source sparse model: {source}")
    return max(models, key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()))


def load_display_frame(analysis):
    basis = np.asarray(analysis["display_orientation"]["world_to_display_row_matrix"], dtype=float)
    plot = analysis["plot"]
    origin = np.asarray(plot["display_origin_world"], dtype=float)
    center = np.asarray(plot["display_center"], dtype=float)
    radius = float(plot["equal_axis_radius"])
    if (basis.shape != (3, 3) or origin.shape != (3,) or center.shape != (3,)
            or not np.isfinite([*basis.ravel(), *origin, *center, radius]).all()
            or radius <= 0 or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)):
        raise ValueError("Source preview does not provide a valid fixed display frame")
    return basis, origin, center, radius


def verify_subset(source, filtered):
    original_ids, output_ids = set(source.points3D), set(filtered.points3D)
    if not output_ids < original_ids:
        raise AssertionError("Saved model is not a proper subset of source point IDs")
    if (set(source.reg_image_ids()) != set(filtered.reg_image_ids())
            or set(source.cameras) != set(filtered.cameras)
            or set(source.frames) != set(filtered.frames) or set(source.rigs) != set(filtered.rigs)):
        raise AssertionError("Saved model changed image, camera, frame or rig identities")
    max_pose_delta = 0.
    for image_id in source.reg_image_ids():
        before, after = source.images[image_id], filtered.images[image_id]
        if before.name != after.name or before.camera_id != after.camera_id:
            raise AssertionError("Saved model changed camera-image association")
        max_pose_delta = max(max_pose_delta, float(np.max(np.abs(
            before.cam_from_world().matrix() - after.cam_from_world().matrix()))))
    if max_pose_delta > 1e-10:
        raise AssertionError("Saved model changed camera poses")
    for camera_id, before in source.cameras.items():
        after = filtered.cameras[camera_id]
        if (before.model != after.model or before.width != after.width or before.height != after.height
                or not np.array_equal(before.params, after.params)):
            raise AssertionError("Saved model changed camera calibration")
    for point_id in output_ids:
        before, after = source.points3D[point_id], filtered.points3D[point_id]
        if not np.array_equal(before.xyz, after.xyz) or not np.array_equal(before.color, after.color):
            raise AssertionError("Saved model changed retained point coordinates or colors")
        track = lambda point: sorted((e.image_id, e.point2D_idx) for e in point.track.elements)
        if track(before) != track(after):
            raise AssertionError("Saved model changed retained point observations")
    return {"proper_subset": True, "retained_points": len(output_ids),
        "all_coordinates_colors_tracks_exactly_preserved": True,
        "all_camera_calibrations_exactly_preserved": True,
        "all_image_frame_rig_ids_preserved": True, "max_pose_element_difference": max_pose_delta}


def plot_comparison(path, xyz, colors, keep, frame, label, consensus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    basis, origin, center, radius = frame
    display = (xyz - origin) @ basis
    fig, axes = plt.subplots(3, 2, figsize=(10, 14), layout="constrained")
    for row, (a, b, view) in enumerate(((0, 2, "Front"), (1, 2, "Side"), (0, 1, "Top"))):
        for col, selected in enumerate((np.ones(len(xyz), dtype=bool), keep)):
            ax = axes[row, col]
            ax.set_facecolor("#f2f3f5")
            ax.scatter(display[selected, a], display[selected, b], c=colors[selected] / 255.,
                       s=1.8, linewidths=0, rasterized=True)
            ax.set(xlim=(center[a] - radius, center[a] + radius),
                   ylim=(center[b] - radius, center[b] + radius),
                   xlabel=("Horizontal", "Depth", "Up")[a],
                   ylabel=("Horizontal", "Depth", "Up")[b],
                   title=f"{view} — {'Source' if col == 0 else 'Mask cleanup'}: {selected.sum():,} points")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.15)
    fig.suptitle(f"Person mask cleanup — {label}\n{consensus:.0%} foreground agreement; identical view and scale", fontsize=15)
    fig.supxlabel("Only existing sparse points are removed. Original camera poses, coordinates and colors are preserved.", fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"source_points": len(xyz), "retained_points": int(keep.sum()),
        "world_to_display_row_matrix": basis.tolist(), "display_origin_world": origin.tolist(),
        "display_center": center.tolist(), "equal_axis_radius": radius,
        "point_marker_area": 1.8, "subsampling": False,
        "source_points_outside_display_cube": int(np.any(np.abs(display - center) > radius, axis=1).sum()),
        "retained_points_outside_display_cube": int((keep & np.any(np.abs(display - center) > radius, axis=1)).sum())}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parents[1] / "light_shirt_quality/subject_preview")
    parser.add_argument("--manifest", type=Path, default=HERE / "mask_manifest.json")
    parser.add_argument("--output-parent", type=Path, default=HERE)
    parser.add_argument("--variants", nargs="+", default=["mask_consensus_90:0.90", "mask_consensus_97:0.97"],
                        help="Separate output folder names and foreground fractions, NAME:FRACTION")
    parser.add_argument("--foreground-threshold", type=float, default=.5)
    parser.add_argument("--tolerance-baseline-pixels", type=float, default=3.)
    parser.add_argument("--reference-long-edge", type=float, default=3200.)
    parser.add_argument("--min-positive-view-fraction", type=float, default=.2)
    parser.add_argument("--min-track-views", type=int, default=3)
    args = parser.parse_args()
    if not (0 < args.foreground_threshold < 1 and np.isfinite(args.tolerance_baseline_pixels)
            and args.tolerance_baseline_pixels >= 0 and args.reference_long_edge > 0
            and 0 < args.min_positive_view_fraction <= 1 and args.min_track_views >= 3):
        parser.error("Invalid mask, tolerance, support or distinct-view threshold")
    variants = []
    for item in args.variants:
        try:
            name, value = item.rsplit(":", 1)
            fraction = float(value)
        except ValueError:
            parser.error(f"Expected NAME:FRACTION, got {item!r}")
        if not name or Path(name).name != name or name in (".", "..") or not 0 < fraction <= 1:
            parser.error(f"Invalid variant: {item!r}")
        variants.append((name, fraction))
    if len({name for name, _ in variants}) != len(variants):
        parser.error("Each variant needs its own unique folder name")
    args.variants = variants
    return args


def main():
    args = arguments()
    source, manifest_path, output_parent = args.source.resolve(), args.manifest.resolve(), args.output_parent.resolve()
    for name, _ in args.variants:
        if (output_parent / name).exists():
            raise FileExistsError(f"Preserving existing experiment: {output_parent / name}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or not manifest.get("ready_for_filter"):
        raise ValueError("Mask manifest must be complete and ready_for_filter after review")
    records = manifest.get("images")
    if not isinstance(records, dict):
        raise ValueError("Mask manifest images must map source image names to records")
    model_path, model = load_model(source)
    if model.num_points3D() < 3 or model.num_reg_images() < 3:
        raise ValueError("Source needs at least three cameras and three existing points")
    analysis_path = source / "analysis.json"
    source_analysis = json.loads(analysis_path.read_text())
    frame = load_display_frame(source_analysis)
    image_root = (source / "images").resolve()
    source_database = source / "colmap/database.db"
    metadata_path = source / "colmap/sparse/sfm_inputs.json"
    metadata = json.loads(metadata_path.read_text())
    if "quality_refinement" in metadata:
        raise ValueError("Use the source preview cache signature, without a refinement recipe")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    current_images = [[p.relative_to(image_root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns] for p in images]
    camera_groups = image_root.parent / "camera_groups.json"
    if (metadata.get("images") != current_images or metadata.get("camera_mode", "AUTO")
            != ("PER_FOLDER" if camera_groups.is_file() else "AUTO")):
        raise ValueError("Source preview cache signature no longer matches linked images")
    with sqlite3.connect(source_database.as_uri() + "?mode=ro", uri=True) as database:
        if database.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("Source preview database failed its integrity check")
    watched = {p: file_hash(p) for p in [manifest_path, analysis_path, metadata_path, source_database,
        *(p for p in model_path.iterdir() if p.is_file())]}
    ids = np.asarray(sorted(model.points3D), dtype=np.uint64)
    xyz = np.asarray([model.points3D[int(i)].xyz for i in ids], dtype=float)
    colors = np.asarray([model.points3D[int(i)].color for i in ids], dtype=np.uint8)
    distinct_views = np.asarray([len({e.image_id for e in model.points3D[int(i)].track.elements}) for i in ids])
    if not np.isfinite(xyz).all():
        raise ValueError("Source contains nonfinite 3D points")
    foreground_counts = np.zeros(len(ids), dtype=np.int32)
    in_frame_counts = np.zeros(len(ids), dtype=np.int32)
    view_stats, excluded = [], []
    for image_id in sorted(model.reg_image_ids(), key=lambda i: model.images[i].name):
        image = model.images[image_id]
        record = records.get(image.name)
        if not isinstance(record, dict) or record.get("status") != "valid" or record.get("usable") is not True:
            excluded.append({"image": image.name, "reason": "missing or unapproved manifest record",
                             "record_status": record.get("status") if isinstance(record, dict) else None})
            continue
        camera = model.cameras[image.camera_id]
        try:
            mask_path = Path(record.get("mask_path") or manifest_path.parent / record["mask_relative_path"]).resolve()
            if Path(record["source_path"]).resolve() != (image_root / image.name).resolve():
                raise ValueError("Mask record refers to a different source image")
            source_stat = (image_root / image.name).stat()
            if (record.get("source_size_bytes") != source_stat.st_size
                    or record.get("source_mtime_ns") != source_stat.st_mtime_ns):
                raise ValueError("Source image changed since its mask was generated")
            mask_hash = file_hash(mask_path)
            if record.get("mask_sha256") and record["mask_sha256"] != mask_hash:
                raise ValueError("Mask SHA256 does not match its manifest")
            with Image.open(mask_path) as raster:
                if raster.mode != "L" or raster.size != (camera.width, camera.height):
                    raise ValueError("Mask must be native-size grayscale L uint8")
                values = np.asarray(raster, dtype=np.uint8).copy()
            if (record.get("width"), record.get("height")) != (camera.width, camera.height):
                raise ValueError("Manifest dimensions do not match camera dimensions")
            foreground = values >= args.foreground_threshold * 255.
            foreground_pixels = int(foreground.sum())
            if foreground_pixels == 0 or foreground_pixels == foreground.size:
                raise ValueError("Empty or all-foreground mask provides no reliable person boundary")
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append({"image": image.name, "reason": str(error)})
            continue
        watched[mask_path] = mask_hash
        pose = image.cam_from_world()
        camera_xyz = xyz @ pose.rotation.matrix().T + pose.translation
        finite = np.isfinite(camera_xyz).all(axis=1)
        front = finite & (camera_xyz[:, 2] > 1e-8)
        indices = np.flatnonzero(front)
        projected = camera.img_from_cam(np.ascontiguousarray(camera_xyz[indices]))
        project_finite = np.isfinite(projected).all(axis=1)
        in_image = (project_finite & (projected[:, 0] >= 0) & (projected[:, 0] < camera.width)
                    & (projected[:, 1] >= 0) & (projected[:, 1] < camera.height))
        visible = indices[in_image]
        tolerance = args.tolerance_baseline_pixels * max(1., max(camera.width, camera.height) / args.reference_long_edge)
        votes = sample_foreground(foreground, projected[in_image], tolerance)
        in_frame_counts[visible] += 1
        foreground_counts[visible[votes]] += 1
        view_stats.append({"image": image.name, "image_id": int(image_id), "mask": str(mask_path),
            "mask_sha256": mask_hash, "width": camera.width, "height": camera.height,
            "foreground_pixel_fraction": foreground_pixels / foreground.size,
            "tolerance_native_pixels": tolerance, "in_frame_points": len(visible),
            "foreground_points": int(votes.sum()), "background_points": int((~votes).sum()),
            "ignored_nonfinite_camera_points": int((~finite).sum()),
            "ignored_nonpositive_depth_points": int((finite & ~front).sum()),
            "ignored_nonfinite_projections": int((~project_finite).sum()),
            "ignored_out_of_frame_points": int((project_finite & ~in_image).sum())})
        print(f"Mask votes {len(view_stats)}/{model.num_reg_images()}: {image.name}", flush=True)
    reliable = len(view_stats)
    if reliable < 3:
        raise ValueError(f"Only {reliable} reliable masks; excluded records: {excluded}")
    for path, digest in watched.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Source or mask changed during filtering: {path}")
    masks = {name: choose_points(foreground_counts, in_frame_counts, distinct_views, reliable,
        fraction, args.min_positive_view_fraction, args.min_track_views) for name, fraction in args.variants}
    for name, selected in masks.items():
        if not 3 <= int(selected.sum()) < len(ids):
            raise ValueError(f"{name}: expected a nonempty proper subset; retained {selected.sum()}/{len(ids)}")
    ordered = sorted(args.variants, key=lambda item: item[1])
    for (first, _), (second, _) in zip(ordered, ordered[1:]):
        if np.any(masks[second] & ~masks[first]):
            raise AssertionError("Higher consensus must be a subset of lower consensus")
    output_parent.mkdir(parents=True, exist_ok=True)
    source_names = {model.images[i].name for i in model.reg_image_ids()}
    for name, fraction in args.variants:
        selected = masks[name]
        accepted = set(map(int, ids[selected]))
        candidate = pycolmap.Reconstruction(model_path)
        for point_id in list(candidate.points3D):
            if int(point_id) not in accepted:
                candidate.delete_point3D(point_id)
        if set(candidate.points3D) != accepted or not accepted < set(map(int, ids)):
            raise AssertionError("Output must be a proper subset of original point IDs")
        for point_id in accepted:
            if not np.array_equal(candidate.points3D[point_id].xyz, model.points3D[point_id].xyz):
                raise AssertionError("Filtering changed a source point coordinate")
        if ({candidate.images[i].name for i in candidate.reg_image_ids()} != source_names
                or candidate.num_reg_images() != model.num_reg_images()):
            raise AssertionError("Filtering changed the registered camera set")
        with tempfile.TemporaryDirectory(prefix=f".{name}_", dir=output_parent) as temporary:
            staging = Path(temporary) / "dataset"
            sparse = staging / "colmap/sparse"
            (sparse / "0").mkdir(parents=True)
            # Compute relative to the final folder, since staging has an extra level.
            (staging / "images").symlink_to(os.path.relpath(image_root, output_parent / name), target_is_directory=True)
            if camera_groups.is_file():
                shutil.copy2(camera_groups, staging / "camera_groups.json")
            shutil.copy2(source_database, staging / "colmap/database.db")
            shutil.copy2(metadata_path, sparse / "sfm_inputs.json")
            candidate.write_binary(sparse / "0")
            serialized_validation = verify_subset(model, pycolmap.Reconstruction(sparse / "0"))
            candidate.export_PLY(staging / "subject.ply")
            (staging / "mask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            np.savez_compressed(staging / "mask_votes.npz", source_point_ids=ids,
                foreground_views=foreground_counts, in_frame_views=in_frame_counts,
                distinct_track_views=distinct_views, retained=selected)
            stats = {"experiment": name, "source_dataset": str(source), "source_model": str(model_path),
                "source_points": len(ids), "retained_points": int(selected.sum()),
                "removed_points": int((~selected).sum()), "registered_images": model.num_reg_images(),
                "reliable_mask_views": reliable, "excluded_masks": excluded,
                "manifest_records_without_registered_source_image": sorted(set(records) - source_names),
                "reliable_groups": dict(Counter(Path(v["image"]).parts[0] for v in view_stats)),
                "filter": {"foreground_agreement": fraction,
                    "normalized_matte_threshold": args.foreground_threshold,
                    "mask_value_meaning": "Confidence-like grayscale matte, not calibrated probability",
                    "minimum_positive_views": math.ceil(args.min_positive_view_fraction * reliable),
                    "minimum_positive_fraction_of_reliable_views": args.min_positive_view_fraction,
                    "minimum_distinct_track_views": args.min_track_views,
                    "tolerance_baseline_pixels": args.tolerance_baseline_pixels,
                    "reference_long_edge": args.reference_long_edge,
                    "native_tolerance": "baseline tolerance * max(1, native long edge / reference long edge)",
                    "mask_sampling": "floor projected pixel coordinates, with a disk of integer-pixel offsets within tolerance",
                    "vote_denominator": "Only reliable masks with a finite, positive-depth projection inside the image"},
                "ignored_projection_totals": {key: sum(v[key] for v in view_stats) for key in view_stats[0] if key.startswith("ignored_")},
                "views": view_stats, "proper_subset_of_source": True,
                "camera_poses_and_world_coordinates": "unchanged", "display_orientation": source_analysis["display_orientation"],
                "source_plot_frame": source_analysis["plot"],
                "source_sfm_inputs": metadata,
                "serialized_subset_validation": serialized_validation,
                "limitations": ["Masks can remove valid body details or retain occluded background. Subject motion and segmentation mistakes can affect votes.",
                    "The filter removes existing points only. It does not create geometry, run dense reconstruction, or certify anatomical accuracy."]}
            stats["plot"] = plot_comparison(staging / "comparison.png", xyz, colors, selected, frame, name, fraction)
            stats["plot"]["upright_bounds"] = source_analysis["plot"]["upright_bounds"]
            (staging / "analysis.json").write_text(json.dumps(stats, indent=2) + "\n")
            provenance = {"source_files_sha256": {str(p): digest for p, digest in watched.items()},
                "filter_script_sha256": file_hash(Path(__file__)), "pycolmap": pycolmap.__version__,
                "source_point_ids_preserved": True, "source_coordinates_preserved": True,
                "new_geometry_created": False, "cache_database_is_independent_copy": True}
            (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            (staging / "README.md").write_text(
                f"# Person mask cleanup: {name}\n\n"
                f"Retained {selected.sum():,} of {len(ids):,} existing sparse points using {fraction:.0%} foreground agreement, "
                f"at least {math.ceil(args.min_positive_view_fraction * reliable)} positive views, and at least {args.min_track_views} distinct track images.\n\n"
                f"Source: `{source}`. All {model.num_reg_images()} cameras and the original coordinate system are preserved. "
                "This folder owns its filtered model and database; images link to the original input photos.\n\n"
                "Open this folder in the starter GUI to view the saved result. Fit Colmap recomputes a reconstruction in this experiment folder.\n\n"
                "`comparison.png` uses the source preview's exact display frame and scale. `mask_votes.npz`, `analysis.json`, "
                "the mask manifest and `provenance.json` record the selection and its inputs.\n\n"
                "No artificial points or dense geometry were added. Segmentation errors or subject motion can remove genuine body points; "
                "foreground agreement does not guarantee that every retained point belongs to the person.\n")
            staging.rename(output_parent / name)
        print(f"SAVED {output_parent / name}: {selected.sum():,}/{len(ids):,} original points", flush=True)
    for path, digest in watched.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Source or mask changed while writing variants: {path}")


if __name__ == "__main__":
    main()
