"""Input, runtime, stage-output, and colored-PLY validation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from .colmap_model import image_dimensions, load_model_records, referenced_image_paths
from .commands import StageCommand
from .config import ExperimentPaths, MVSConfig
from .constants import IMAGE_SUFFIXES
from .errors import InputValidationError
from .io_utils import sha256_file, stable_digest, utc_now
from .paths import PathPolicy
from .staging import verify_staged_inputs


def _symlinks(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_symlink()]


def validate_inputs(config: MVSConfig, paths: ExperimentPaths) -> dict[str, Any]:
    """Validate copied images against the calibrated COLMAP model and masks."""
    provenance = verify_staged_inputs(paths.root)
    links = _symlinks(paths.root / "inputs")
    if links:
        raise InputValidationError(
            "MVS inputs must be self-contained; found symlinks: "
            + ", ".join(str(path) for path in links[:10])
        )
    model = load_model_records(paths.model)
    images = referenced_image_paths(model, paths.images)
    cameras = {camera.camera_id: camera for camera in model.cameras}
    dimension_counts: dict[str, int] = {}
    for image in model.images:
        dimensions = image_dimensions(images[image.name])
        camera = cameras[image.camera_id]
        expected = (camera.width, camera.height)
        if dimensions != expected:
            raise InputValidationError(
                f"Image dimensions do not match camera {camera.camera_id} for {image.name}: "
                f"{dimensions} vs {expected}"
            )
        key = f"{dimensions[0]}x{dimensions[1]}"
        dimension_counts[key] = dimension_counts.get(key, 0) + 1

    mask_summary: dict[str, Any] | None = None
    if config.masking_mode == "black_background":
        if paths.masks is None or paths.mask_manifest is None:
            raise InputValidationError("Masking is enabled but mask paths are absent")
        try:
            manifest = json.loads(paths.mask_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InputValidationError(f"Cannot read staged mask manifest: {error}") from error
        records = manifest.get("images")
        if not manifest.get("complete") or not isinstance(records, dict):
            raise InputValidationError("Staged mask manifest is incomplete")
        fractions_recorded = 0
        for image in model.images:
            record = records.get(image.name)
            if not isinstance(record, dict) or record.get("usable") is not True:
                raise InputValidationError(f"Missing usable mask record for {image.name}")
            relative = record.get("mask_relative_path")
            if not isinstance(relative, str):
                raise InputValidationError(f"Missing mask path for {image.name}")
            mask = (paths.mask_manifest.parent / relative).resolve(strict=True)
            try:
                mask.relative_to(paths.root.resolve())
            except ValueError as error:
                raise InputValidationError(f"Mask escapes experiment: {mask}") from error
            if sha256_file(mask) != record.get("sha256"):
                raise InputValidationError(f"Mask checksum mismatch for {image.name}")
            if image_dimensions(mask) != image_dimensions(images[image.name]):
                raise InputValidationError(f"Mask dimensions mismatch for {image.name}")
            fractions_recorded += 1
        mask_summary = {
            "count": fractions_recorded,
            "coordinate_system": "original distorted source pixels",
            "polarity": "larger grayscale values retain foreground",
            "application": "before COLMAP image_undistorter",
        }

    return {
        "validated_at": utc_now(),
        "registered_images": len(model.images),
        "cameras": len(model.cameras),
        "model_format": model.format,
        "model_directory": str(model.directory),
        "image_dimensions": dimension_counts,
        "masks": mask_summary,
        "input_provenance_digest": provenance.get("files_digest"),
        "no_input_symlinks": True,
        "finite_calibration_and_poses": True,
    }


def runtime_environment(config: MVSConfig, paths: ExperimentPaths) -> dict[str, str]:
    """Return subprocess environment with every writable cache under DATA_ROOT."""
    environment = dict(os.environ)
    runtime_root = paths.runtime
    redirects = {
        "TMPDIR": runtime_root / "tmp",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "PIP_CACHE_DIR": runtime_root / "pip-cache",
        "CONDA_PKGS_DIRS": runtime_root / "conda-pkgs",
        "TORCH_HOME": runtime_root / "torch-cache",
        "HF_HOME": runtime_root / "huggingface-cache",
        "SINGULARITY_CACHEDIR": runtime_root / "singularity-cache",
        "SINGULARITY_TMPDIR": runtime_root / "singularity-tmp",
    }
    for name, value in redirects.items():
        value.mkdir(parents=True, exist_ok=True)
        environment[name] = str(value)
    environment["OMP_NUM_THREADS"] = str(config.num_threads)
    return environment


def _run_capture(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    return runner(
        argv,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def validate_runtime(
    config: MVSConfig,
    paths: ExperimentPaths,
    policy: PathPolicy,
    commands: tuple[StageCommand, ...],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Prove that the run is on an allocated GPU and COLMAP exposes all flags."""
    if config.require_slurm_job and not os.environ.get("SLURM_JOB_ID"):
        raise InputValidationError(
            "No active Slurm job detected (SLURM_JOB_ID is unset); refusing GPU MVS on a login node"
        )
    policy.require_method(paths.root, "experiment runtime")
    paths.runtime.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment(config, paths)
    # The probe is created and removed only inside the canonical data root.
    with tempfile.NamedTemporaryFile(dir=paths.runtime, prefix="write-probe-", delete=True) as probe:
        probe.write(b"mvs-storage-probe\n")
        probe.flush()
        os.fsync(probe.fileno())

    if config.colmap_backend == "native":
        launcher_name = config.colmap_binary
    elif config.colmap_backend == "singularity":
        launcher_name = str(config.singularity_binary)
    else:
        launcher_name = str(config.pycolmap_python)
    executable = shutil.which(launcher_name, path=environment.get("PATH"))
    if not executable:
        raise InputValidationError(f"MVS launcher executable not found: {launcher_name}")
    executable_path = Path(executable).resolve()
    if str(executable_path).startswith("/home/"):
        raise InputValidationError(
            f"MVS launcher resolves to home storage ({executable_path}); environments/builds belong under {policy.data_root}"
        )
    container_record: dict[str, Any] | None = None
    pycolmap_record: dict[str, Any] | None = None
    if config.colmap_backend == "singularity":
        assert config.singularity_image is not None
        image = policy.require_method(
            config.singularity_image, "COLMAP Singularity image", must_exist=True
        )
        if not image.is_file():
            raise InputValidationError(f"COLMAP Singularity image is not a file: {image}")
        container_record = {
            "path": str(image),
            "size_bytes": image.stat().st_size,
            "sha256": sha256_file(image),
        }

    gpu = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ],
        environment=environment,
        runner=runner,
    )
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise InputValidationError(f"Allocated NVIDIA GPU is unavailable: {gpu.stdout.strip()}")

    representative = commands[0]
    subcommand_index = representative.argv.index(representative.subcommand)
    colmap_prefix = list(representative.argv[:subcommand_index])
    if config.colmap_backend == "pycolmap":
        probe = _run_capture(
            colmap_prefix + ["probe"], environment=environment, runner=runner
        )
        if probe.returncode != 0:
            raise InputValidationError(
                f"PyCOLMAP CUDA runtime probe failed: {probe.stdout.strip()}"
            )
        try:
            pycolmap_record = json.loads(probe.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise InputValidationError(
                f"PyCOLMAP runtime probe returned malformed output: {probe.stdout.strip()}"
            ) from error
        if (
            pycolmap_record.get("pycolmap_version") != "4.2.0"
            or pycolmap_record.get("pycolmap_has_cuda") is not True
            or pycolmap_record.get("dense_api") is not True
        ):
            raise InputValidationError(
                f"PyCOLMAP runtime is not the pinned CUDA dense build: {pycolmap_record}"
            )
        version = probe
    else:
        version = _run_capture(colmap_prefix + ["-h"], environment=environment, runner=runner)
    if version.returncode != 0:
        raise InputValidationError(f"COLMAP failed its version/help probe: {version.stdout.strip()}")
    help_by_command: dict[str, str] = {}
    for command in commands:
        subcommand = command.subcommand
        command_subcommand_index = command.argv.index(subcommand)
        command_prefix = list(command.argv[:command_subcommand_index])
        if subcommand not in help_by_command:
            result = _run_capture(
                command_prefix + [subcommand, "-h"],
                environment=environment,
                runner=runner,
            )
            if result.returncode != 0:
                raise InputValidationError(
                    f"COLMAP command {subcommand!r} is unavailable (CUDA MVS may be disabled): "
                    f"{result.stdout.strip()}"
                )
            help_by_command[subcommand] = result.stdout
        advertised = help_by_command[subcommand]
        missing = sorted(
            {
                argument
                for argument in command.argv[command_subcommand_index + 1 :]
                if argument.startswith("--") and argument not in advertised
            }
        )
        if missing:
            raise InputValidationError(
                f"Installed COLMAP {subcommand} does not advertise configured options: {missing}"
            )

    gpu_rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    return (
        {
            "validated_at": utc_now(),
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_rows": gpu_rows,
            "colmap_executable": str(executable_path),
            "colmap_backend": config.colmap_backend,
            "container": container_record,
            "pycolmap": pycolmap_record,
            "colmap_help_header": version.stdout.splitlines()[:8],
            "available_subcommands": sorted(help_by_command),
            "cache_and_temp_redirects": {
                key: environment[key]
                for key in (
                    "TMPDIR",
                    "XDG_CACHE_HOME",
                    "PIP_CACHE_DIR",
                    "CONDA_PKGS_DIRS",
                    "TORCH_HOME",
                    "HF_HOME",
                    "SINGULARITY_CACHEDIR",
                    "SINGULARITY_TMPDIR",
                )
            },
            "data_root_write_probe": True,
        },
        environment,
    )


_PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def validate_colored_ply(path: Path) -> dict[str, Any]:
    """Scan every vertex for finite XYZ and require RGB color properties."""
    file_size = path.stat().st_size
    if file_size <= 0:
        raise InputValidationError(f"PLY output is empty: {path}")
    with path.open("rb") as stream:
        first = stream.readline()
        if first.strip() != b"ply":
            raise InputValidationError(f"Not a PLY file: {path}")
        file_format = None
        vertex_count = None
        current_element = None
        vertex_properties: list[tuple[str, str]] = []
        header_size = len(first)
        while header_size < 1024 * 1024:
            raw = stream.readline()
            if not raw:
                raise InputValidationError(f"Truncated PLY header: {path}")
            header_size += len(raw)
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise InputValidationError(f"Non-ASCII PLY header: {path}") from error
            fields = line.split()
            if fields[:1] == ["format"] and len(fields) >= 2:
                file_format = fields[1]
            elif fields[:1] == ["element"] and len(fields) == 3:
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and current_element == "vertex":
                if len(fields) != 3 or fields[1] == "list":
                    raise InputValidationError("List-valued vertex properties are unsupported")
                vertex_properties.append((fields[1], fields[2]))
            elif line == "end_header":
                break
        else:
            raise InputValidationError(f"PLY header exceeds 1 MiB: {path}")
        if file_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise InputValidationError(f"Unsupported PLY format {file_format!r}")
        if vertex_count is None or vertex_count <= 0:
            raise InputValidationError(f"PLY contains no vertices: {path}")
        names = [name for _, name in vertex_properties]
        required = {"x", "y", "z", "red", "green", "blue"}
        if not required.issubset(names):
            raise InputValidationError(
                f"PLY is not a colored XYZ cloud; missing {sorted(required - set(names))}"
            )
        xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]
        color_indices = [names.index(channel) for channel in ("red", "green", "blue")]

        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        if file_format == "ascii":
            for vertex_index in range(vertex_count):
                raw = stream.readline()
                if not raw:
                    raise InputValidationError(f"PLY ended at vertex {vertex_index}/{vertex_count}")
                fields = raw.split()
                if len(fields) < len(vertex_properties):
                    raise InputValidationError(f"Malformed ASCII PLY vertex {vertex_index}")
                try:
                    xyz = [float(fields[index]) for index in xyz_indices]
                    rgb = [int(fields[index]) for index in color_indices]
                except ValueError as error:
                    raise InputValidationError(f"Malformed ASCII PLY vertex {vertex_index}") from error
                _validate_vertex(vertex_index, xyz, rgb, minimum, maximum)
        else:
            try:
                codes = [_PLY_TYPES[data_type][0] for data_type, _ in vertex_properties]
            except KeyError as error:
                raise InputValidationError(f"Unsupported PLY scalar type {error.args[0]!r}") from error
            endian = "<" if file_format == "binary_little_endian" else ">"
            record_struct = struct.Struct(endian + "".join(codes))
            vertices_left = vertex_count
            vertex_index = 0
            while vertices_left:
                batch_count = min(vertices_left, 65536)
                block = stream.read(batch_count * record_struct.size)
                if len(block) != batch_count * record_struct.size:
                    raise InputValidationError(
                        f"Binary PLY ended at vertex {vertex_index}/{vertex_count}"
                    )
                for values in struct.iter_unpack(record_struct.format, block):
                    xyz = [float(values[index]) for index in xyz_indices]
                    rgb = [int(values[index]) for index in color_indices]
                    _validate_vertex(vertex_index, xyz, rgb, minimum, maximum)
                    vertex_index += 1
                vertices_left -= batch_count

    return {
        "path": str(path),
        "format": file_format,
        "vertex_count": vertex_count,
        "size_bytes": file_size,
        "sha256": sha256_file(path),
        "has_rgb": True,
        "all_xyz_finite": True,
        "bounds_min_xyz": minimum,
        "bounds_max_xyz": maximum,
    }


def _validate_vertex(
    index: int,
    xyz: list[float],
    rgb: list[int],
    minimum: list[float],
    maximum: list[float],
) -> None:
    if not all(math.isfinite(value) for value in xyz):
        raise InputValidationError(f"PLY vertex {index} has non-finite XYZ")
    if not all(0 <= value <= 255 for value in rgb):
        raise InputValidationError(f"PLY vertex {index} has invalid RGB")
    for axis, value in enumerate(xyz):
        minimum[axis] = min(minimum[axis], value)
        maximum[axis] = max(maximum[axis], value)


def stage_output_summary(
    stage: str, config: MVSConfig, paths: ExperimentPaths
) -> dict[str, Any]:
    """Verify the concrete artifacts that make a stage resumable."""
    model = load_model_records(paths.model)
    expected_images = len(model.images)
    if stage == "mask_inputs":
        images = referenced_image_paths(model, paths.masked_images)
        return {
            "image_count": len(images),
            "expected_image_count": expected_images,
            "manifest_sha256": sha256_file(paths.manifests / "masked_images.json"),
        }
    if stage == "undistort":
        workspace_images = [
            path
            for path in paths.workspace.joinpath("images").rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(workspace_images) != expected_images:
            raise InputValidationError(
                f"Undistorter produced {len(workspace_images)} images; expected {expected_images}"
            )
        workspace_model = load_model_records(paths.workspace / "sparse")
        if len(workspace_model.images) != expected_images:
            raise InputValidationError("Undistorted model lost registered images")
        if {image.name for image in workspace_model.images} != {
            image.name for image in model.images
        }:
            raise InputValidationError("Undistorted model changed the registered image-name set")
        workspace_image_map = referenced_image_paths(
            workspace_model, paths.workspace / "images"
        )
        workspace_cameras = {
            camera.camera_id: camera for camera in workspace_model.cameras
        }
        for image in workspace_model.images:
            dimensions = image_dimensions(workspace_image_map[image.name])
            camera = workspace_cameras[image.camera_id]
            if dimensions != (camera.width, camera.height):
                raise InputValidationError(
                    f"Undistorted image/camera dimensions disagree for {image.name}: "
                    f"{dimensions} vs {(camera.width, camera.height)}"
                )
        patch_config = paths.workspace / "stereo" / "patch-match.cfg"
        if not patch_config.is_file():
            raise InputValidationError("Undistorter did not create stereo/patch-match.cfg")
        return {
            "image_count": len(workspace_images),
            "registered_images": len(workspace_model.images),
            "patch_match_config_sha256": sha256_file(patch_config),
        }
    if stage == "patch_match":
        kind = "geometric" if config.geom_consistency else "photometric"
        depth = sorted((paths.workspace / "stereo" / "depth_maps").rglob(f"*.{kind}.bin"))
        normals = sorted((paths.workspace / "stereo" / "normal_maps").rglob(f"*.{kind}.bin"))
        if len(depth) != expected_images or len(normals) != expected_images:
            raise InputValidationError(
                f"PatchMatch {kind} output incomplete: depth={len(depth)}, normals={len(normals)}, "
                f"expected={expected_images}"
            )
        return {
            "kind": kind,
            "depth_map_count": len(depth),
            "normal_map_count": len(normals),
            "depth_map_bytes": sum(path.stat().st_size for path in depth),
            "normal_map_bytes": sum(path.stat().st_size for path in normals),
        }
    if stage == "fusion":
        return validate_colored_ply(paths.fused)
    if stage == "mesh":
        # Mesher PLY has vertices with color in normal COLMAP output.  Reuse the
        # strict loader so a GUI-visible but colorless artifact is not accepted.
        return validate_colored_ply(paths.mesh)
    raise InputValidationError(f"Unknown stage for output validation: {stage}")
