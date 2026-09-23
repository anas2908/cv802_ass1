"""Subprocess worker for the official PyCOLMAP 4.2 CUDA dense-MVS API.

The wheel contains the same COLMAP undistortion, PatchMatch and fusion code as
the standalone CLI.  This backend is an explicit fallback for clusters where a
verified Singularity SIF cannot be mounted.  Keeping it in a subprocess retains
the controller's process-group cancellation, combined logs and stage receipts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


EXPECTED_VERSION = "4.2.0"


def _boolean(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise argparse.ArgumentTypeError("boolean options must be numeric 0 or 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv802-pycolmap",
        description="Official PyCOLMAP 4.2 CUDA dense-MVS subprocess backend",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("probe", help="verify the pinned wheel and CUDA support")

    undistort = subcommands.add_parser("image_undistorter")
    undistort.add_argument("--image_path", type=Path, required=True)
    undistort.add_argument("--input_path", type=Path, required=True)
    undistort.add_argument("--output_path", type=Path, required=True)
    undistort.add_argument("--output_type", choices=("COLMAP",), default="COLMAP")
    undistort.add_argument("--max_image_size", type=int, required=True)
    undistort.add_argument("--num_patch_match_src_images", type=int, required=True)
    undistort.add_argument("--num_threads", type=int, required=True)

    patch_match = subcommands.add_parser("patch_match_stereo")
    patch_match.add_argument("--workspace_path", type=Path, required=True)
    patch_match.add_argument("--workspace_format", choices=("COLMAP",), default="COLMAP")
    patch_match.add_argument("--PatchMatchStereo.gpu_index", dest="gpu_index", required=True)
    patch_match.add_argument(
        "--PatchMatchStereo.geom_consistency",
        dest="geom_consistency",
        type=_boolean,
        required=True,
    )
    patch_match.add_argument(
        "--PatchMatchStereo.filter", dest="filter", type=_boolean, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.cache_size", dest="cache_size", type=float, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.max_image_size", dest="max_image_size", type=int, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.num_iterations", dest="num_iterations", type=int, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.num_samples", dest="num_samples", type=int, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.window_radius", dest="window_radius", type=int, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.window_step", dest="window_step", type=int, required=True
    )
    patch_match.add_argument(
        "--PatchMatchStereo.filter_min_num_consistent",
        dest="filter_min_num_consistent",
        type=int,
        required=True,
    )
    patch_match.add_argument(
        "--PatchMatchStereo.num_threads", dest="num_threads", type=int, required=True
    )

    fusion = subcommands.add_parser("stereo_fusion")
    fusion.add_argument("--workspace_path", type=Path, required=True)
    fusion.add_argument("--workspace_format", choices=("COLMAP",), default="COLMAP")
    fusion.add_argument("--input_type", choices=("photometric", "geometric"), required=True)
    fusion.add_argument("--output_path", type=Path, required=True)
    fusion.add_argument("--StereoFusion.cache_size", dest="cache_size", type=float, required=True)
    fusion.add_argument(
        "--StereoFusion.max_image_size", dest="max_image_size", type=int, required=True
    )
    fusion.add_argument(
        "--StereoFusion.min_num_pixels", dest="min_num_pixels", type=int, required=True
    )
    fusion.add_argument(
        "--StereoFusion.max_reproj_error", dest="max_reproj_error", type=float, required=True
    )
    fusion.add_argument(
        "--StereoFusion.max_depth_error", dest="max_depth_error", type=float, required=True
    )
    fusion.add_argument(
        "--StereoFusion.max_normal_error", dest="max_normal_error", type=float, required=True
    )
    fusion.add_argument(
        "--StereoFusion.check_num_images", dest="check_num_images", type=int, required=True
    )
    fusion.add_argument("--StereoFusion.num_threads", dest="num_threads", type=int, required=True)

    poisson = subcommands.add_parser("poisson_mesher")
    poisson.add_argument("--input_path", type=Path, required=True)
    poisson.add_argument("--output_path", type=Path, required=True)
    return parser


def _load_pycolmap() -> Any:
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - exercised by runtime preflight.
        raise RuntimeError("The pinned pycolmap-cuda12 wheel is not installed") from error
    return pycolmap


def _verify_runtime(pycolmap: Any, *, require_cuda: bool) -> dict[str, Any]:
    version = str(getattr(pycolmap, "__version__", "unknown"))
    has_cuda = bool(getattr(pycolmap, "has_cuda", False))
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected PyCOLMAP {EXPECTED_VERSION}, found {version}")
    if require_cuda and not has_cuda:
        raise RuntimeError("PyCOLMAP wheel does not expose CUDA PatchMatch support")
    return {"pycolmap_version": version, "pycolmap_has_cuda": has_cuda}


def run_stage(arguments: argparse.Namespace, pycolmap: Any) -> dict[str, Any]:
    identity = _verify_runtime(pycolmap, require_cuda=arguments.command == "patch_match_stereo")
    if arguments.command == "probe":
        return {**identity, "dense_api": all(
            hasattr(pycolmap, name)
            for name in ("undistort_images", "patch_match_stereo", "stereo_fusion")
        )}
    if arguments.command == "image_undistorter":
        options = pycolmap.UndistortCameraOptions()
        options.max_image_size = arguments.max_image_size
        pycolmap.undistort_images(
            arguments.output_path,
            arguments.input_path,
            arguments.image_path,
            output_type=arguments.output_type,
            num_patch_match_src_images=arguments.num_patch_match_src_images,
            undistort_options=options,
            num_threads=arguments.num_threads,
        )
    elif arguments.command == "patch_match_stereo":
        options = pycolmap.PatchMatchOptions()
        for name in (
            "gpu_index",
            "geom_consistency",
            "filter",
            "cache_size",
            "max_image_size",
            "num_iterations",
            "num_samples",
            "window_radius",
            "window_step",
            "filter_min_num_consistent",
            "num_threads",
        ):
            setattr(options, name, getattr(arguments, name))
        pycolmap.patch_match_stereo(
            arguments.workspace_path,
            workspace_format=arguments.workspace_format,
            options=options,
        )
    elif arguments.command == "stereo_fusion":
        options = pycolmap.StereoFusionOptions()
        for name in (
            "cache_size",
            "max_image_size",
            "min_num_pixels",
            "max_reproj_error",
            "max_depth_error",
            "max_normal_error",
            "check_num_images",
            "num_threads",
        ):
            setattr(options, name, getattr(arguments, name))
        pycolmap.stereo_fusion(
            arguments.output_path,
            arguments.workspace_path,
            workspace_format=arguments.workspace_format,
            input_type=arguments.input_type,
            options=options,
            output_type="PLY",
        )
    elif arguments.command == "poisson_mesher":
        pycolmap.poisson_meshing(arguments.input_path, arguments.output_path)
    else:  # pragma: no cover - argparse restricts the command.
        raise AssertionError(arguments.command)
    return {**identity, "stage": arguments.command, "status": "complete"}


def main(argv: Sequence[str] | None = None, *, pycolmap_module: Any | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run_stage(arguments, pycolmap_module or _load_pycolmap())
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
