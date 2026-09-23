"""Fail-closed path policy for the VGGSfM method.

Production code fixes every mutable/heavy location below
``/l/users/anas.khan/cv_802_ass1/vggsfm``.  ``Layout`` accepts another root
only as an injected Python object so unit tests can use a temporary directory;
the CLI never exposes such an override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import METHOD_DATA_ROOT, PROJECT_DATA_ROOT
from .errors import StoragePolicyError


def _resolved(path: Path, *, must_exist: bool = False) -> Path:
    try:
        return path.expanduser().resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise StoragePolicyError(f"Cannot resolve path {path}: {exc}") from exc


def require_within(path: Path, root: Path, *, label: str, must_exist: bool = False) -> Path:
    """Resolve *path* and prove it is inside *root* (symlinks included)."""

    resolved_root = _resolved(root, must_exist=must_exist)
    resolved_path = _resolved(path, must_exist=must_exist)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise StoragePolicyError(
            f"{label} must be inside {resolved_root}; received {resolved_path}"
        ) from exc
    return resolved_path


@dataclass(frozen=True)
class Layout:
    """All generated paths for one method, rooted on the data filesystem."""

    root: Path = METHOD_DATA_ROOT
    enforce_production_root: bool = True

    def __post_init__(self) -> None:
        root = _resolved(Path(self.root), must_exist=False)
        object.__setattr__(self, "root", root)
        if self.enforce_production_root and root != _resolved(METHOD_DATA_ROOT):
            raise StoragePolicyError(
                f"Production VGGSfM data root is fixed at {METHOD_DATA_ROOT}; got {root}"
            )

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def official_root(self) -> Path:
        return self.downloads / "source" / "vggsfm"

    @property
    def env_prefix(self) -> Path:
        return self.root / "envs" / "vggsfm"

    @property
    def python(self) -> Path:
        return self.env_prefix / "bin" / "python"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def dataset_root(self, dataset: str) -> Path:
        return self.inputs / dataset

    def experiment_root(self, run_id: str) -> Path:
        return self.experiments / run_id

    def output_root(self, run_id: str) -> Path:
        return self.outputs / run_id

    def assert_member(self, path: Path, *, label: str, must_exist: bool = False) -> Path:
        return require_within(path, self.root, label=label, must_exist=must_exist)

    def create_runtime_directories(self) -> None:
        """Create only data-root directories; never create mutable state in source."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.assert_member(self.root, label="method data root", must_exist=True)
        for directory in (
            self.inputs,
            self.experiments,
            self.outputs,
            self.downloads,
            self.root / "envs",
            self.cache / "conda",
            self.cache / "pip",
            self.cache / "torch",
            self.cache / "huggingface",
            self.cache / "huggingface" / "datasets",
            self.cache / "xdg",
            self.cache / "cuda",
            self.cache / "matplotlib",
            self.cache / "pycache",
            self.cache / "torch_extensions",
            self.cache / "triton",
            self.cache / "numba",
            self.cache / "gradio",
            self.root / "config" / "xdg",
            self.root / "config" / "xdg-data",
            self.tmp,
            self.logs,
        ):
            self.assert_member(directory, label="runtime directory")
            directory.mkdir(parents=True, exist_ok=True)

    def runtime_environment(self) -> dict[str, str]:
        """Return an environment with every known cache/temp redirected to data."""

        cache = self.cache
        values = {
            "PYTHONUNBUFFERED": "1",
            "CONDA_PKGS_DIRS": str(cache / "conda"),
            "CONDA_ENVS_PATH": str(self.root / "envs"),
            "CONDARC": "/dev/null",
            "PIP_CACHE_DIR": str(cache / "pip"),
            "PIP_CONFIG_FILE": "/dev/null",
            "TORCH_HOME": str(cache / "torch"),
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
            "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "XDG_CONFIG_HOME": str(self.root / "config" / "xdg"),
            "XDG_DATA_HOME": str(self.root / "config" / "xdg-data"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "NUMBA_CACHE_DIR": str(cache / "numba"),
            "GRADIO_TEMP_DIR": str(cache / "gradio"),
            "TMPDIR": str(self.tmp),
            "TEMP": str(self.tmp),
            "TMP": str(self.tmp),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUSERBASE": str(self.root / "envs" / "python-user-base"),
            "HYDRA_FULL_ERROR": "1",
            "MPLBACKEND": "Agg",
            "QT_QPA_PLATFORM": "offscreen",
            "TOKENIZERS_PARALLELISM": "false",
        }
        env = dict(os.environ)
        env.update(values)
        return env


def production_layout() -> Layout:
    """Construct the only layout accepted by the command-line interface."""

    # This also guards accidental changes to the broader project root constant.
    if METHOD_DATA_ROOT.parent != PROJECT_DATA_ROOT:
        raise StoragePolicyError("Internal data-root constants are inconsistent")
    return Layout()
