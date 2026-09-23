"""Strict JSON configuration model for the COLMAP dense pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .constants import DATA_ROOT, DEFAULT_DATA_ROOT, SCHEMA_VERSION
from .errors import ConfigurationError
from .io_utils import stable_digest
from .paths import PathPolicy
from .paths import safe_relative_path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ConfigurationError(f"{label} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class MVSConfig:
    raw: dict[str, Any]
    experiment: str
    images_relative: str
    model_relative: str
    masking_mode: str
    masks_relative: str | None
    mask_manifest_relative: str | None
    mask_threshold: int
    mask_dilation_pixels: int
    colmap_binary: str
    colmap_backend: str
    singularity_binary: str | None
    singularity_image: Path | None
    pycolmap_python: Path | None
    gpu_index: str
    max_image_size: int
    source_images_per_view: int
    geom_consistency: bool
    patch_cache_gb: float
    patch_num_iterations: int
    patch_num_samples: int
    patch_window_radius: int
    patch_window_step: int
    patch_filter_min_consistent: int
    fusion_cache_gb: float
    fusion_min_num_pixels: int
    fusion_max_reproj_error: float
    fusion_max_depth_error: float
    fusion_max_normal_error: float
    fusion_check_num_images: int
    num_threads: int
    meshing: str
    require_slurm_job: bool

    @classmethod
    def load(cls, path: Path, policy: PathPolicy) -> "MVSConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Cannot read configuration {path}: {error}") from error
        return cls.parse(raw, policy)

    @classmethod
    def parse(cls, raw: Any, policy: PathPolicy) -> "MVSConfig":
        obj = _mapping(raw, "configuration")
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise ConfigurationError(f"schema_version must equal {SCHEMA_VERSION}")
        root_value = str(obj.get("data_root", ""))
        # Templates use this token so one config works on the cluster and on
        # another machine after setting CV802_DATA_ROOT.
        if "${CV802_DATA_ROOT}" in root_value:
            portable_root = Path(
                os.environ.get("CV802_DATA_ROOT", str(DEFAULT_DATA_ROOT))
            ).expanduser()
            if not portable_root.is_absolute():
                raise ConfigurationError("CV802_DATA_ROOT must be an absolute path")
            root_value = root_value.replace("${CV802_DATA_ROOT}", str(portable_root))
        configured_root = Path(root_value).expanduser().resolve(strict=False)
        if configured_root != policy.method_root:
            raise ConfigurationError(
                f"data_root must be exactly {policy.method_root}; got {configured_root}"
            )

        experiment = str(obj.get("experiment", ""))
        experiment_root = policy.experiment(experiment)
        inputs = _mapping(obj.get("inputs"), "inputs")
        images_relative = str(inputs.get("images", "inputs/images"))
        model_relative = str(inputs.get("sparse_model", "inputs/sparse"))
        policy.from_experiment_relative(experiment_root, images_relative, "inputs.images")
        policy.from_experiment_relative(experiment_root, model_relative, "inputs.sparse_model")

        masking = _mapping(obj.get("masking", {"mode": "none"}), "masking")
        masking_mode = str(masking.get("mode", "none"))
        if masking_mode not in {"none", "black_background"}:
            raise ConfigurationError("masking.mode must be 'none' or 'black_background'")
        masks_relative = mask_manifest_relative = None
        if masking_mode == "black_background":
            masks_relative = str(masking.get("masks", "inputs/masks"))
            mask_manifest_relative = str(masking.get("manifest", "inputs/mask_manifest.json"))
            policy.from_experiment_relative(experiment_root, masks_relative, "masking.masks")
            policy.from_experiment_relative(experiment_root, mask_manifest_relative, "masking.manifest")

        colmap = _mapping(obj.get("colmap", {}), "colmap")
        binary = str(colmap.get("binary", "colmap"))
        if not binary or any(character.isspace() for character in binary):
            raise ConfigurationError("colmap.binary must be one executable path or name")
        binary_path = Path(binary)
        if binary_path.is_absolute():
            resolved_binary = binary_path.resolve(strict=False)
            trusted_system = any(
                str(resolved_binary).startswith(prefix) for prefix in ("/usr/", "/apps/")
            )
            try:
                resolved_binary.relative_to(policy.method_root)
                in_method_storage = True
            except ValueError:
                in_method_storage = False
            if not trusted_system and not in_method_storage:
                raise ConfigurationError(
                    "An absolute colmap.binary must be a system executable or live under the MVS data root"
                )
        backend = str(colmap.get("backend", "native"))
        if backend not in {"native", "singularity", "pycolmap"}:
            raise ConfigurationError(
                "colmap.backend must be 'native', 'singularity', or 'pycolmap'"
            )
        singularity_binary: str | None = None
        singularity_image: Path | None = None
        pycolmap_python: Path | None = None
        if backend == "singularity":
            singularity_binary = str(colmap.get("singularity_binary", "/usr/bin/singularity"))
            if not singularity_binary or any(character.isspace() for character in singularity_binary):
                raise ConfigurationError("colmap.singularity_binary must be one executable path or name")
            image_relative = safe_relative_path(
                str(colmap.get("image", "dependencies/colmap-4.2.0-20260901.7982.sif")),
                "colmap.image",
            )
            singularity_image = (policy.method_root / image_relative).resolve(strict=False)
            policy.require_method(singularity_image, "COLMAP Singularity image")
        elif backend == "pycolmap":
            python_relative = safe_relative_path(
                str(colmap.get("python", "envs/mvs-engine/bin/python")),
                "colmap.python",
            )
            pycolmap_python = (policy.method_root / python_relative).resolve(strict=False)
            policy.require_method(pycolmap_python, "PyCOLMAP Python executable")
        gpu_index = str(colmap.get("gpu_index", "0"))
        if not re.fullmatch(r"\d+", gpu_index):
            raise ConfigurationError("Exactly one non-negative COLMAP gpu_index is required")

        patch = _mapping(obj.get("patch_match", {}), "patch_match")
        fusion = _mapping(obj.get("fusion", {}), "fusion")
        execution = _mapping(obj.get("execution", {}), "execution")
        meshing = str(obj.get("meshing", "none"))
        if meshing not in {"none", "poisson", "delaunay"}:
            raise ConfigurationError("meshing must be 'none', 'poisson', or 'delaunay'")
        require_slurm = execution.get("require_slurm_job", True)
        if not isinstance(require_slurm, bool):
            raise ConfigurationError("execution.require_slurm_job must be boolean")
        geom = patch.get("geom_consistency", True)
        if not isinstance(geom, bool):
            raise ConfigurationError("patch_match.geom_consistency must be boolean")

        return cls(
            raw=dict(obj),
            experiment=experiment,
            images_relative=images_relative,
            model_relative=model_relative,
            masking_mode=masking_mode,
            masks_relative=masks_relative,
            mask_manifest_relative=mask_manifest_relative,
            mask_threshold=_integer(masking.get("threshold", 128), "masking.threshold", 1, 254),
            mask_dilation_pixels=_integer(
                masking.get("dilation_pixels", 3), "masking.dilation_pixels", 0, 101
            ),
            colmap_binary=binary,
            colmap_backend=backend,
            singularity_binary=singularity_binary,
            singularity_image=singularity_image,
            pycolmap_python=pycolmap_python,
            gpu_index=gpu_index,
            max_image_size=_integer(obj.get("max_image_size", 1600), "max_image_size", 256, 16384),
            source_images_per_view=_integer(
                patch.get("source_images_per_view", 10),
                "patch_match.source_images_per_view",
                1,
                100,
            ),
            geom_consistency=geom,
            patch_cache_gb=_number(patch.get("cache_size_gb", 8), "patch_match.cache_size_gb", 0.25, 512),
            patch_num_iterations=_integer(patch.get("num_iterations", 5), "patch_match.num_iterations", 1, 20),
            patch_num_samples=_integer(patch.get("num_samples", 15), "patch_match.num_samples", 1, 100),
            patch_window_radius=_integer(patch.get("window_radius", 5), "patch_match.window_radius", 1, 32),
            patch_window_step=_integer(patch.get("window_step", 1), "patch_match.window_step", 1, 4),
            patch_filter_min_consistent=_integer(
                patch.get("filter_min_num_consistent", 2),
                "patch_match.filter_min_num_consistent",
                1,
                20,
            ),
            fusion_cache_gb=_number(fusion.get("cache_size_gb", 8), "fusion.cache_size_gb", 0.25, 512),
            fusion_min_num_pixels=_integer(
                fusion.get("min_num_pixels", 5), "fusion.min_num_pixels", 1, 100
            ),
            fusion_max_reproj_error=_number(
                fusion.get("max_reproj_error", 2.0), "fusion.max_reproj_error", 0.01, 100
            ),
            fusion_max_depth_error=_number(
                fusion.get("max_depth_error", 0.01), "fusion.max_depth_error", 0.00001, 1
            ),
            fusion_max_normal_error=_number(
                fusion.get("max_normal_error", 10.0), "fusion.max_normal_error", 0.1, 180
            ),
            fusion_check_num_images=_integer(
                fusion.get("check_num_images", 50), "fusion.check_num_images", 1, 1000
            ),
            num_threads=_integer(execution.get("num_threads", 16), "execution.num_threads", 1, 256),
            meshing=meshing,
            require_slurm_job=require_slurm,
        )

    @property
    def digest(self) -> str:
        return stable_digest(self.raw)

    def paths(self, policy: PathPolicy) -> "ExperimentPaths":
        experiment = policy.experiment(self.experiment)
        return ExperimentPaths(
            root=experiment,
            images=policy.from_experiment_relative(experiment, self.images_relative, "images"),
            model=policy.from_experiment_relative(experiment, self.model_relative, "sparse model"),
            masks=(
                policy.from_experiment_relative(experiment, self.masks_relative, "masks")
                if self.masks_relative
                else None
            ),
            mask_manifest=(
                policy.from_experiment_relative(experiment, self.mask_manifest_relative, "mask manifest")
                if self.mask_manifest_relative
                else None
            ),
        )


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    images: Path
    model: Path
    masks: Path | None
    mask_manifest: Path | None

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def workspace(self) -> Path:
        return self.work / "dense"

    @property
    def masked_images(self) -> Path:
        return self.work / "masked_images"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def fused(self) -> Path:
        return self.outputs / "fused.ply"

    @property
    def mesh(self) -> Path:
        return self.outputs / "mesh.ply"

    @property
    def receipts(self) -> Path:
        return self.root / "receipts"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"
