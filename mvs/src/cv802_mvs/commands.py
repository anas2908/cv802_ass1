"""Pure COLMAP command construction for auditable dry-runs and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from .config import ExperimentPaths, MVSConfig
from .errors import InputValidationError


@dataclass(frozen=True)
class StageCommand:
    name: str
    subcommand: str
    argv: tuple[str, ...]

    def display(self) -> str:
        return shlex.join(self.argv)


def build_commands(config: MVSConfig, paths: ExperimentPaths) -> tuple[StageCommand, ...]:
    image_path = paths.masked_images if config.masking_mode == "black_background" else paths.images
    def colmap(subcommand: str, *arguments: str) -> tuple[str, ...]:
        if config.colmap_backend == "native":
            return (config.colmap_binary, subcommand, *arguments)
        if config.colmap_backend == "pycolmap":
            assert config.pycolmap_python is not None
            worker = Path(__file__).resolve().parents[2] / "run_pycolmap_stage.py"
            return (
                str(config.pycolmap_python),
                "-B",
                str(worker),
                subcommand,
                *arguments,
            )
        assert config.singularity_binary is not None and config.singularity_image is not None
        # Only the MVS method root is mounted.  Inputs were copied here during
        # staging, so the container cannot write back into the SfM method.
        return (
            config.singularity_binary,
            "exec",
            "--nv",
            "--bind",
            f"{paths.root.parents[1]}:{paths.root.parents[1]}",
            str(config.singularity_image),
            config.colmap_binary,
            subcommand,
            *arguments,
        )

    undistort_arguments = [
        "--image_path",
        str(image_path),
        "--input_path",
        str(paths.model),
        "--output_path",
        str(paths.workspace),
        "--output_type",
        "COLMAP",
        "--max_image_size",
        str(config.max_image_size),
    ]
    if config.colmap_backend == "pycolmap":
        undistort_arguments.extend(
            (
                "--num_patch_match_src_images",
                str(config.source_images_per_view),
                "--num_threads",
                str(config.num_threads),
            )
        )

    commands = [
        StageCommand(
            "undistort",
            "image_undistorter",
            colmap("image_undistorter", *undistort_arguments),
        ),
        StageCommand(
            "patch_match",
            "patch_match_stereo",
            colmap(
                "patch_match_stereo",
                "--workspace_path",
                str(paths.workspace),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.gpu_index",
                config.gpu_index,
                # Numeric booleans avoid COLMAP 3.12's CLI parsing ambiguity for
                # the strings "true"/"false" (COLMAP issue #3279).
                "--PatchMatchStereo.geom_consistency",
                "1" if config.geom_consistency else "0",
                "--PatchMatchStereo.filter",
                "1",
                "--PatchMatchStereo.cache_size",
                str(config.patch_cache_gb),
                "--PatchMatchStereo.max_image_size",
                str(config.max_image_size),
                "--PatchMatchStereo.num_iterations",
                str(config.patch_num_iterations),
                "--PatchMatchStereo.num_samples",
                str(config.patch_num_samples),
                "--PatchMatchStereo.window_radius",
                str(config.patch_window_radius),
                "--PatchMatchStereo.window_step",
                str(config.patch_window_step),
                "--PatchMatchStereo.filter_min_num_consistent",
                str(config.patch_filter_min_consistent),
                "--PatchMatchStereo.num_threads",
                str(config.num_threads),
            ),
        ),
        StageCommand(
            "fusion",
            "stereo_fusion",
            colmap(
                "stereo_fusion",
                "--workspace_path",
                str(paths.workspace),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric" if config.geom_consistency else "photometric",
                "--output_path",
                str(paths.fused),
                "--StereoFusion.cache_size",
                str(config.fusion_cache_gb),
                "--StereoFusion.max_image_size",
                str(config.max_image_size),
                "--StereoFusion.min_num_pixels",
                str(config.fusion_min_num_pixels),
                "--StereoFusion.max_reproj_error",
                str(config.fusion_max_reproj_error),
                "--StereoFusion.max_depth_error",
                str(config.fusion_max_depth_error),
                "--StereoFusion.max_normal_error",
                str(config.fusion_max_normal_error),
                "--StereoFusion.check_num_images",
                str(config.fusion_check_num_images),
                "--StereoFusion.num_threads",
                str(config.num_threads),
            ),
        ),
    ]
    if config.meshing == "poisson":
        commands.append(
            StageCommand(
                "mesh",
                "poisson_mesher",
                colmap(
                    "poisson_mesher",
                    "--input_path",
                    str(paths.fused),
                    "--output_path",
                    str(paths.mesh),
                ),
            )
        )
    elif config.meshing == "delaunay":
        commands.append(
            StageCommand(
                "mesh",
                "delaunay_mesher",
                colmap(
                    "delaunay_mesher",
                    "--input_path",
                    str(paths.workspace),
                    "--output_path",
                    str(paths.mesh),
                ),
            )
        )
    return tuple(commands)


def set_auto_source_count(path: Path, count: int) -> int:
    """Replace each PatchMatch source line with ``__auto__, N``.

    ``image_undistorter`` creates alternating reference/source lines.  The
    explicit cap is the most direct COLMAP-supported control for memory use.
    """
    try:
        original = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InputValidationError(f"Cannot read PatchMatch configuration {path}: {error}") from error
    significant = [index for index, line in enumerate(original) if line.strip() and not line.lstrip().startswith("#")]
    if not significant or len(significant) % 2:
        raise InputValidationError(
            f"Unexpected patch-match.cfg structure in {path}: expected reference/source pairs"
        )
    output = list(original)
    for index in significant[1::2]:
        output[index] = f"__auto__, {count}"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    temporary.replace(path)
    return len(significant) // 2
