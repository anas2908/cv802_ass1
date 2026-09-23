#!/usr/bin/env python3
"""Read-only E3/E4, E6/E7 and E8/E9 comparison in the original E1 frame.

Run with the SfM virtual environment. --self-test repeats E3/E4 in all three
columns and checks independently measured arrays, metrics and voxel sets.
Only this script's comparison output folder is written; no model is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SFM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SFM_ROOT / "work" / "quality-sfm"))
from compare_models import image_states, model_at, summarize
from plot_comparison import comparable_points, frame_from_baseline, pose_check

SUBJECTS = ("light_shirt", "black_shirt_crutches")
LABELS = {"light_shirt": "Light shirt", "black_shirt_crutches": "Dark shirt and crutches"}
EXPERIMENTS = (("E3", "E6", "E8"), ("E4", "E7", "E9"))
COLUMN_LABELS = ("Original quality / guided subset", "Same pairs / guided OFF", "Vocabulary tree / guided ON")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-root", type=Path, default=SFM_ROOT)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--min-track", type=int, default=3, help="Minimum distinct image views")
    parser.add_argument("--max-error", type=float, default=3., help="Mean residual in E1 baseline pixels")
    parser.add_argument("--min-angle", type=float, default=1.5, help="Minimum maximum acute triangulation angle")
    parser.add_argument("--point-size", type=float, default=1.8, help="Identical marker area in every panel")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    args.sfm_root = args.sfm_root.resolve()
    if args.output_dir is None:
        args.output_dir = args.sfm_root / "work" / "matching-ablation" / "comparison"
        if args.self_test:
            args.output_dir /= "self_test"
    args.output_dir = args.output_dir.resolve()
    if args.min_track < 2 or args.max_error <= 0 or not 0 < args.min_angle <= 90 or args.point_size <= 0 or args.dpi <= 0:
        parser.error("Invalid quality or plotting settings")
    return args


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_hashes(path):
    # Hash every COLMAP file present, including rigs/frames when provided.
    return {str(p.resolve()): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(path.iterdir()) if p.is_file() and p.suffix in (".bin", ".txt")}


def datasets(root, subject, self_test):
    recon = root / "reconstructions"
    experiments = recon / "experiments"
    quality = recon / f"{subject}_quality" / "subject_preview"
    clean = (experiments / "light_person_mask_cleanup" / "mask_consensus_90" if subject == "light_shirt"
             else experiments / "black_person_crutches_cleanup" / "body_mask_consensus_90_crutches")
    if self_test:
        return ((quality, quality, quality), (clean, clean, clean))
    return (
        (quality, experiments / "E6_guided_off" / f"{subject}_quality" / "subject_preview",
         experiments / "E8_vocab_guided" / f"{subject}_quality" / "subject_preview"),
        (clean, experiments / "E7_guided_off_consensus90" / subject,
         experiments / "E9_vocab_guided_consensus90" / subject),
    )


def check_cameras(baseline, candidate, reference):
    states = image_states(candidate)
    poses = pose_check(reference, states)
    if set(reference) != set(states):
        raise ValueError("The registered image sets differ; fixed-camera comparison cannot proceed")
    changed_ids = [name for name in reference if reference[name]["image_id"] != states[name]["image_id"]]
    if changed_ids:
        raise ValueError(f"Image IDs changed for {len(changed_ids)} images")
    if set(baseline.cameras) != set(candidate.cameras):
        raise ValueError("Camera ID sets differ from E1")
    camera_rows = []
    for camera_id, camera in sorted(candidate.cameras.items()):
        if camera_id not in baseline.cameras:
            raise ValueError(f"Unexpected camera {camera_id}")
        original = baseline.cameras[camera_id]
        expected = pycolmap.Camera(original.todict())
        expected.rescale(camera.width, camera.height)
        delta = float(np.max(np.abs(camera.params - expected.params)))
        ok = camera.model == original.model and np.isfinite(camera.params).all() and delta <= 1e-10
        if not ok:
            raise ValueError(f"Camera {camera_id} differs from the resolution-scaled E1 calibration")
        camera_rows.append({"camera_id": int(camera_id), "model": str(camera.model),
                            "width": camera.width, "height": camera.height,
                            "max_difference_from_resolution_scaled_E1_params": delta})
    changed_assignments = [name for name in reference
                           if baseline.images[reference[name]["image_id"]].camera_id !=
                           candidate.images[states[name]["image_id"]].camera_id]
    if changed_assignments:
        raise ValueError("Camera assignments differ from E1")
    return {"passed": True, "same_registered_image_names": True, "same_image_ids": True,
            "same_image_camera_assignments": True, "poses": poses, "calibrations": camera_rows}


def load_subject(subject, args):
    baseline_preview = args.sfm_root / "reconstructions" / subject / "subject_preview"
    baseline_path, baseline = model_at(baseline_preview)
    basis, origin, low, high, analysis_path = frame_from_baseline(baseline_preview)
    reference = image_states(baseline)
    frozen = model_hashes(baseline_path)
    frozen[str(analysis_path.resolve())] = {"bytes": analysis_path.stat().st_size, "sha256": sha256(analysis_path)}
    all_rows, audit_rows, all_grids = [], [], []
    for row_index, paths in enumerate(datasets(args.sfm_root, subject, args.self_test)):
        plot_row, metric_row, grid_row = [], [], []
        for column, dataset in enumerate(paths):
            model_path, model = model_at(dataset)
            hashes = model_hashes(model_path)
            for file, signature in hashes.items():
                if file in frozen and frozen[file] != signature:
                    raise RuntimeError(f"An input changed between repeated reads: {file}")
                frozen[file] = signature
            cameras = check_cameras(baseline, model, reference)
            # Use both existing helpers and assert their selections are identical.
            metrics, grids = summarize(model, reference, basis, origin, low, high, args)
            plot = comparable_points(model, reference, basis, origin, low, high, args)
            for name, value in plot["summary"].items():
                if metrics[name] != value:
                    raise AssertionError(f"Metric/plot disagreement: {subject} {name}")
            assert len(plot["xyz"]) == metrics["retained_comparable_points"]
            actual = EXPERIMENTS[row_index][0] if args.self_test else EXPERIMENTS[row_index][column]
            metric_row.append({"experiment": EXPERIMENTS[row_index][column], "actual_input_experiment": actual,
                               "dataset": str(dataset.resolve()), "model": str(model_path),
                               "input_model_hashes": hashes, "camera_unchanged_check": cameras, "metrics": metrics})
            plot_row.append(plot)
            grid_row.append(grids)
        all_rows.append(plot_row)
        audit_rows.append(metric_row)
        all_grids.append(grid_row)
    self_test_checks = []
    if args.self_test:
        for row in range(2):
            for column in (1, 2):
                checks = {"row": row, "column": column,
                          "same_display_coordinates": np.array_equal(all_rows[row][0]["xyz"], all_rows[row][column]["xyz"]),
                          "same_rgb_colors": np.array_equal(all_rows[row][0]["rgb"], all_rows[row][column]["rgb"]),
                          "same_metrics": audit_rows[row][0]["metrics"] == audit_rows[row][column]["metrics"],
                          "same_voxel_sets": all_grids[row][0] == all_grids[row][column]}
                if not all(value for name, value in checks.items() if name not in ("row", "column")):
                    raise AssertionError(f"Repeated-input self-test failed: {checks}")
                self_test_checks.append(checks)
    voxel_comparisons = []
    for row in range(2):
        for column in (1, 2):
            first, second = all_grids[row][0], all_grids[row][column]
            voxel_comparisons.append({"reference_experiment": EXPERIMENTS[row][0],
                                      "candidate_experiment": EXPERIMENTS[row][column],
                                      "scales": {key: {"reference": len(first[key]), "candidate": len(second[key]),
                                                        "shared": len(first[key] & second[key]),
                                                        "candidate_only": len(second[key] - first[key]),
                                                        "reference_only": len(first[key] - second[key])} for key in first}})
    audit = {"subject": subject, "baseline_reference_model": str(baseline_path),
             "baseline_frame_analysis": str(analysis_path), "rows": audit_rows,
             "voxel_comparisons": voxel_comparisons, "self_test_checks": self_test_checks,
             "fixed_frame": {"origin_world": origin.tolist(), "world_to_upright_row_matrix": basis.tolist(),
                             "upright_min": low.tolist(), "upright_max": high.tolist(),
                             "display_center": ((low + high) / 2).tolist(),
                             "equal_axis_radius": float(np.max(high - low)) * .535}}
    return all_rows, audit, frozen


def plot_subject(subject, plots, audit, args):
    center = np.asarray(audit["fixed_frame"]["display_center"])
    radius = audit["fixed_frame"]["equal_axis_radius"]
    output = {}
    for horizontal, view in ((0, "front"), (1, "side")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10.7), layout="constrained")
        title = f"{LABELS[subject]} · {view} reference view"
        if args.self_test:
            title += " · SELF-TEST (repeated original inputs)"
        fig.suptitle(title, fontsize=19)
        for row in range(2):
            for column in range(3):
                axis = axes[row, column]
                data = plots[row][column]
                xyz = data["xyz"]
                axis.scatter(xyz[:, horizontal], xyz[:, 2], c=data["rgb"], s=args.point_size,
                             linewidths=0, alpha=1, rasterized=True)
                axis.set_xlim(center[horizontal] - radius, center[horizontal] + radius)
                axis.set_ylim(center[2] - radius, center[2] + radius)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel(("Horizontal" if horizontal == 0 else "Depth") + " (arbitrary units)", fontsize=9)
                axis.set_ylabel("Up (arbitrary units)", fontsize=9)
                axis.grid(alpha=.13)
                axis.tick_params(labelsize=8)
                exp = EXPERIMENTS[row][column]
                method = "Quality + rectangle preview" if row == 0 else "Segmentation consensus 90%"
                m = audit["rows"][row][column]["metrics"]
                column_label = COLUMN_LABELS[column] if not args.self_test else "Original input repeated"
                axis.set_title(f"{exp} · {column_label}\n{method}\n"
                               f"{m['input_subject_points']:,} input points → {len(xyz):,} comparable", fontsize=10.5, pad=8)
        fig.supxlabel(
            f"Every accepted point shown; same marker size, E1 frame and bounds. Distinct views ≥ {args.min_track}; "
            f"mean error ≤ {args.max_error:g} E1 pixels; acute angle ≥ {args.min_angle:g}°.\n"
            "Top row already uses the common rectangle preview; bottom row adds 90% segmentation (crutches protected for dark shirt).\n"
            "Outside-bound points are excluded and counted in the report. Point count and coverage do not measure anatomical accuracy.",
            fontsize=9,
        )
        path = args.output_dir / f"{subject}_{view}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        output[view] = str(path)
    return output


def markdown_report(report):
    lines = ["# Sparse SfM matching comparison", ""]
    if report["self_test"]:
        lines += ["**Self-test only:** all three columns use the original E3/E4 inputs. These are not new experiment results.", ""]
    method = report["method"]
    lines += ["All measurements use the original E1 cameras for pixel normalization and the saved E1 upright frame and bounds. "
              f"Comparable points have at least {method['min_distinct_views']} distinct image views, "
              f"mean reprojection error at most {method['max_mean_error_E1_pixels']:g} E1 pixels, "
              f"and maximum acute triangulation angle at least {method['min_maximum_acute_angle_degrees']:g}°. "
              "Points outside the fixed bounds are reported separately.", "",
              "The first row contains rectangle-filtered quality previews, not complete full-scene clouds. "
              "The second row adds the same 90% segmentation policy, with crutches protected for the dark-shirt subject. "
              "The 90% threshold is mask agreement across valid projected views, not accuracy or percentage of photographs retained.", ""]
    for subject in report["subjects"]:
        lines += [f"## {LABELS[subject['subject']]}", "",
                  "| Experiment | Input preview points | Comparable points | Mean error (E1 px) | Occupied voxels (height/100) | Quality points outside bounds |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in subject["rows"]:
            for item in row:
                m = item["metrics"]
                label = item["experiment"]
                if report["self_test"]:
                    label += f" (input {item['actual_input_experiment']})"
                lines.append(f"| {label} | {m['input_subject_points']:,} | {m['retained_comparable_points']:,} | "
                             f"{m['baseline_pixel_mean_point_error_retained']['mean']:.4f} | "
                             f"{m['occupied_voxels']['height/100']:,} | {m['quality_points_outside_fixed_baseline_bounds']:,} |")
        lines += ["", "All compared cameras retain the original poses and resolution-scaled calibration. "
                  "The JSON report includes input model SHA-256 hashes, detailed errors, tracks, triangulation angles, "
                  "vertical coverage and voxel overlap.", "",
                  f"[Front plot](<{subject['plots']['front']}>) · [Side plot](<{subject['plots']['side']}>)", ""]
    lines += ["More points or more occupied voxels alone do not establish a more accurate person shape. "
              "There is no independent ground truth here; masking mistakes and subject motion can affect the result. "
              "Fixed bounds can exclude real extremities beyond the original preview, so compare the excluded counts too.", ""]
    return "\n".join(lines)


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "self_test": args.self_test,
              "comparison_script_sha256": sha256(Path(__file__)),
              "method": {"min_distinct_views": args.min_track, "max_mean_error_E1_pixels": args.max_error,
                         "min_maximum_acute_angle_degrees": args.min_angle, "point_marker_area": args.point_size,
                         "pixel_normalization": "Per-observation x/y scale to the same image's E1 camera dimensions before residual norm averaging",
                         "frame_and_bounds": "Original E1 subject_preview analysis, unchanged for all six panels",
                         "sampling": "All comparable points shown, no subsampling or generated geometry",
                         "scope": "Sparse coverage and numerical consistency; no ground-truth anatomical accuracy claim"},
              "measurement_helpers": {str(SFM_ROOT / "work" / "quality-sfm" / name): sha256(SFM_ROOT / "work" / "quality-sfm" / name)
                                      for name in ("compare_models.py", "plot_comparison.py")},
              "subjects": []}
    frozen = {}
    for subject in args.subjects:
        print(f"Measuring {subject}...", flush=True)
        plots, audit, hashes = load_subject(subject, args)
        frozen.update(hashes)
        audit["plots"] = plot_subject(subject, plots, audit, args)
        report["subjects"].append(audit)
    for path, expected in frozen.items():
        file = Path(path)
        if file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
            raise RuntimeError(f"An input model or reference frame changed while comparing: {path}")
    report["input_files_unchanged_after_comparison"] = True
    report["input_file_hashes"] = frozen
    report["self_test_passed"] = True if args.self_test else None
    json_path, md_path = args.output_dir / "comparison.json", args.output_dir / "README.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    md_path.write_text(markdown_report(report))
    print(json.dumps({"report": str(json_path), "summary": str(md_path), "self_test_passed": report["self_test_passed"],
                      "plots": {s["subject"]: s["plots"] for s in report["subjects"]}}, indent=2))


if __name__ == "__main__":
    main()
