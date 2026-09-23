#!/usr/bin/env python3
"""Verify saved reconstructions open in the starter API without recomputing.

With no positional paths, check both quality datasets and their subject previews.
Pass other dataset paths to validate existing saved results. This does not open
a GUI window. Every expensive PyCOLMAP entry point is replaced by a failure.
"""

import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON verification report")
    return parser.parse_args()


def stored_files(dataset):
    """Record canonical saved data so a cache-only check cannot silently alter it."""
    paths = [dataset / "colmap" / "database.db"]
    paths.extend(p for p in (dataset / "colmap" / "sparse").rglob("*") if p.is_file())
    return {str(p.relative_to(dataset)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(paths) if p.is_file()}


def preflight(dataset):
    sparse = dataset / "colmap" / "sparse"
    if not (dataset / "images").is_dir() or not (sparse / "sfm_inputs.json").is_file():
        raise FileNotFoundError(f"Saved images/model metadata not ready: {dataset}")
    candidates = [sparse] + [p for p in sparse.iterdir() if p.is_dir()]
    if not any(all((p / (stem + extension)).is_file()
                   for stem in ("cameras", "images", "points3D"))
               for p in candidates for extension in (".bin", ".txt")):
        raise FileNotFoundError(f"No completed sparse model: {dataset}")


def forbidden(name, calls):
    def fail(*args, **kwargs):
        calls.append(name)
        raise AssertionError(f"Cache miss: forbidden reconstruction operation {name}")
    return fail


def main():
    args = arguments()
    datasets = args.datasets or [
        ROOT / "reconstructions" / name / suffix
        for name in ("light_shirt_quality", "black_shirt_crutches_quality")
        for suffix in ("", "subject_preview")
    ]
    datasets = [p.resolve() for p in datasets]
    # Check every requested artifact before calling into the application.
    for dataset in datasets:
        preflight(dataset)

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "assignment1"))
    import numpy as np
    import pycolmap
    from modules.colmap.api import ColmapAPI

    calls = []
    report = {"all_cached": True, "gui_defaults": {
        "camera_model": "OPENCV", "matcher": "exhaustive_matcher"
    }, "datasets": []}
    with ExitStack() as patches:
        names = ["extract_features", "incremental_mapping", "triangulate_points"]
        names += [name for name in dir(pycolmap) if name.startswith("match_")
                  and callable(getattr(pycolmap, name))]
        for name in names:
            patches.enter_context(mock.patch.object(pycolmap, name, forbidden(name, calls)))
        # Cover the native-crop extraction API as well as normal extraction.
        patches.enter_context(mock.patch.object(pycolmap, "FeatureExtractor",
            SimpleNamespace(create=forbidden("FeatureExtractor.create", calls))))
        for dataset in datasets:
            before = stored_files(dataset)
            started = time.monotonic()
            api = ColmapAPI(0, "OPENCV", "exhaustive_matcher")
            api.data_path = str(dataset)
            messages = io.StringIO()
            with redirect_stdout(messages):
                api.estimate_cameras(recompute=False)
                api._thread.join(timeout=30)
            if api._thread.is_alive():
                raise TimeoutError(f"Cache loading exceeded 30 seconds: {dataset}")
            if api.estimate_error:
                raise AssertionError(f"Cached opening failed for {dataset}: {api.estimate_error}") from api.estimate_error
            if calls:
                raise AssertionError(f"Reconstruction was invoked: {calls}")
            if stored_files(dataset) != before:
                raise AssertionError(f"Cache check changed canonical saved files: {dataset}")
            points = np.asarray(api._pcd.points)
            colors = np.asarray(api._pcd.colors)
            if len(points) == 0 or not np.isfinite(points).all() or colors.shape != points.shape:
                raise AssertionError(f"Invalid displayed point cloud: {dataset}")
            if not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1):
                raise AssertionError(f"Invalid displayed point colors: {dataset}")
            if api.num_cameras < 2 or api.activate_camera_name not in api.camera_names:
                raise AssertionError(f"Invalid displayed cameras: {dataset}")
            if "loaded the cached reconstruction" not in messages.getvalue():
                raise AssertionError(f"API did not report a cache load: {dataset}")
            report["datasets"].append({
                "dataset": str(dataset), "registered_images": api.num_cameras,
                "displayed_points": len(points), "cached": True,
                "canonical_files_unchanged": True,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "application_messages": messages.getvalue().strip().splitlines(),
            })
    report["forbidden_operations_called"] = calls
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.resolve().write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
