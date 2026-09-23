"""Headless PyCOLMAP sparse-reconstruction engine.

The original assignment GUI remains available in ``assignment1``. This module
is GUI-free so the same stages run in a Slurm job without an X display. Heavy
inputs, databases, temporary models, logs, and outputs are rejected unless
they live below the configured SfM data root.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
SUPPORTED_PYCOLMAP_VERSION = "4.2.0"
ENGINE_RECIPE = "cv802-headless-sfm-v2"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
MATCHERS = frozenset({"exhaustive", "sequential"})
EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _configured_data_root() -> Path:
    """Return an explicit absolute data root; never fall back to CWD/home."""

    configured = Path(os.environ.get("CV802_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
    if not configured.is_absolute():
        raise RuntimeError("CV802_DATA_ROOT must be an absolute path")
    return configured.resolve()


DATA_ROOT = _configured_data_root()
SFM_DATA_ROOT = DATA_ROOT / "sfm"


@dataclass(frozen=True)
class SfMConfig:
    """Inputs that define one immutable sparse-reconstruction experiment."""

    experiment_name: str
    image_dir: Path
    camera_model: str = "SIMPLE_RADIAL"
    camera_mode: str = "auto"
    matcher: str = "exhaustive"
    device: str = "auto"
    max_image_size: int = 3200
    max_num_features: int = 16384
    num_threads: int = 8
    random_seed: int = 0

    def validate(self) -> "SfMConfig":
        if not EXPERIMENT_NAME.fullmatch(self.experiment_name):
            raise ValueError(
                "experiment_name must be 1-128 characters, start with an "
                "alphanumeric character, and contain only letters, digits, '.', '_', or '-'"
            )
        if self.matcher not in MATCHERS:
            raise ValueError(f"matcher must be one of {sorted(MATCHERS)}")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if self.camera_mode not in {"auto", "per_folder", "single"}:
            raise ValueError("camera_mode must be auto, per_folder, or single")
        if not self.camera_model or not self.camera_model.replace("_", "").isalnum():
            raise ValueError("camera_model must be a non-empty COLMAP model name")
        if self.max_image_size == 0 or self.max_image_size < -1:
            raise ValueError("max_image_size must be -1 or a positive integer")
        if self.max_num_features < 1 or self.num_threads < 1:
            raise ValueError("feature and thread limits must be positive")
        if not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        require_under(self.image_dir, SFM_DATA_ROOT / "inputs", "SfM image input")
        return self


def require_under(path: Path, root: Path, label: str) -> Path:
    """Resolve *path* and fail if it escapes the required storage root."""

    resolved = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must be below {resolved_root}; got {resolved}") from error
    return resolved


def discover_images(image_dir: Path) -> list[Path]:
    """Find ordinary supported image files without following file symlinks."""

    image_dir = require_under(image_dir, SFM_DATA_ROOT / "inputs", "SfM image input")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    images: list[Path] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.is_symlink():
            continue
        cursor = path
        while cursor != image_dir:
            if cursor.is_symlink():
                raise ValueError(f"SfM inputs must be physical copies, not symlinks: {cursor}")
            cursor = cursor.parent
        images.append(require_under(path, image_dir, "SfM image file"))
    if len(images) < 2:
        raise ValueError(f"at least two ordinary image files are required in {image_dir}")
    return images


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def image_manifest(image_dir: Path, images: Sequence[Path]) -> list[dict[str, Any]]:
    """Hash inputs and reject a file that changes while it is being hashed."""

    records: list[dict[str, Any]] = []
    for path in images:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        fingerprint_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        fingerprint_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if fingerprint_before != fingerprint_after:
            raise RuntimeError(f"input changed while it was being hashed: {path}")
        records.append(
            {
                "path": path.relative_to(image_dir).as_posix(),
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    return records


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON durably and atomically on the destination filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _has_cuda(pycolmap: Any) -> bool:
    value = getattr(pycolmap, "has_cuda", False)
    return bool(value() if callable(value) else value)


def _validate_pycolmap(pycolmap: Any) -> None:
    version = str(getattr(pycolmap, "__version__", "missing"))
    if version != SUPPORTED_PYCOLMAP_VERSION:
        raise RuntimeError(
            f"this engine is validated for pycolmap=={SUPPORTED_PYCOLMAP_VERSION}; got {version}"
        )


def _enum_member(enum: Any, name: str) -> Any:
    for candidate in (name, name.lower(), name.upper()):
        if hasattr(enum, candidate):
            return getattr(enum, candidate)
    raise AttributeError(f"{enum!r} has no {name} member")


def _num_cuda_devices(pycolmap: Any) -> int:
    getter = getattr(pycolmap, "get_num_cuda_devices", None)
    if not callable(getter):
        return 0
    try:
        return int(getter())
    except Exception as error:
        raise RuntimeError(f"PyCOLMAP could not enumerate CUDA devices: {error}") from error


def resolve_device(pycolmap: Any, requested: str) -> tuple[str, Any]:
    """Resolve CPU/CUDA without silently downgrading an explicit request."""

    if requested == "auto":
        try:
            cuda_devices = _num_cuda_devices(pycolmap) if _has_cuda(pycolmap) else 0
        except RuntimeError:
            cuda_devices = 0
        selected = "cuda" if cuda_devices > 0 else "cpu"
    else:
        selected = requested
    if selected == "cuda" and not _has_cuda(pycolmap):
        raise RuntimeError(
            "CUDA was requested but this PyCOLMAP build reports has_cuda=False. "
            "Install the pinned Linux pycolmap-cuda12 wheel in DATA_ROOT."
        )
    return selected, _enum_member(pycolmap.Device, selected)


def cuda_runtime_identity(pycolmap: Any, *, runner: Any = subprocess.run) -> dict[str, Any]:
    """Prove exactly one CUDA device is visible and record its identity."""

    count = _num_cuda_devices(pycolmap)
    if count != 1:
        raise RuntimeError(
            f"the SfM CUDA recipe requires exactly one visible GPU; PyCOLMAP sees {count}. "
            "Run inside the one-GPU allocation or narrow CUDA_VISIBLE_DEVICES."
        )
    if str(DATA_ROOT).startswith("/l/") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "CUDA execution on the cluster requires an active Slurm job; "
            "refusing login-node execution"
        )
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot query CUDA device identity with nvidia-smi: {error}") from error
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "unknown error").strip()
        raise RuntimeError(f"nvidia-smi failed while identifying the allocated GPU: {message}")
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi identity row: {line!r}")
        index, uuid, name, driver, memory_mib = fields
        try:
            index_number = int(index)
            memory = int(memory_mib)
        except ValueError as error:
            raise RuntimeError(f"invalid nvidia-smi identity row: {line!r}") from error
        gpus.append(
            {
                "index": index_number,
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "memory_mib": memory,
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU identities")
    slurm_token = os.environ.get("SLURM_JOB_GPUS")
    visible_token = os.environ.get("CUDA_VISIBLE_DEVICES")
    selection_token = slurm_token or visible_token
    selected_identity = None
    if selection_token and "," not in selection_token:
        normalized = selection_token.strip()
        for gpu in gpus:
            if normalized in {str(gpu["index"]), gpu["uuid"]}:
                selected_identity = gpu
                break
    if selected_identity is None and len(gpus) == 1:
        selected_identity = gpus[0]
    if selected_identity is None:
        raise RuntimeError(
            "one CUDA device is visible to PyCOLMAP, but its nvidia-smi identity "
            f"cannot be resolved from SLURM_JOB_GPUS/CUDA_VISIBLE_DEVICES={selection_token!r}"
        )
    return {
        "pycolmap_visible_cuda_devices": count,
        "cuda_visible_devices": visible_token,
        "slurm_job_gpus": slurm_token,
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "selected_gpu": selected_identity,
        "nvidia_smi_inventory": gpus,
    }


def _camera_mode(pycolmap: Any, name: str) -> Any:
    mapping = {"auto": "AUTO", "per_folder": "PER_FOLDER", "single": "SINGLE"}
    return _enum_member(pycolmap.CameraMode, mapping[name])


def _registered_images(reconstruction: Any) -> list[Any]:
    return [reconstruction.images[index] for index in reconstruction.reg_image_ids()]


def _validate_finite_vector(values: Iterable[Any], label: str) -> None:
    numeric = [float(value) for value in values]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"non-finite or empty {label}")


def reconstruction_metrics(reconstruction: Any) -> dict[str, Any]:
    """Validate camera poses, intrinsics, geometry, and point colours."""

    if hasattr(reconstruction, "is_valid") and not reconstruction.is_valid():
        raise ValueError("PyCOLMAP reports an internally invalid reconstruction")
    registered = _registered_images(reconstruction)
    if len(registered) < 2:
        raise ValueError("a valid reconstruction needs at least two registered images")
    points = list(reconstruction.points3D.values())
    if not points:
        raise ValueError("reconstruction contains no 3D points")

    for camera_id, camera in reconstruction.cameras.items():
        if int(camera.width) < 1 or int(camera.height) < 1:
            raise ValueError(f"camera {camera_id} has invalid dimensions")
        _validate_finite_vector(camera.params, f"parameters for camera {camera_id}")
        if not camera.verify_params():
            raise ValueError(f"camera {camera_id} has invalid parameter count or values")

    names: set[str] = set()
    for image in registered:
        if image.name in names:
            raise ValueError(f"duplicate registered image name: {image.name}")
        names.add(image.name)
        pose = image.cam_from_world()
        _validate_finite_vector(pose.translation, f"translation for {image.name}")
        matrix = pose.rotation.matrix()
        _validate_finite_vector(
            (value for row in matrix for value in row), f"rotation for {image.name}"
        )
        if image.camera_id not in reconstruction.cameras:
            raise ValueError(f"image {image.name} references missing camera {image.camera_id}")

    for point_id, point in reconstruction.points3D.items():
        _validate_finite_vector(point.xyz, f"point {point_id}")
        if not math.isfinite(float(point.error)):
            raise ValueError(f"point {point_id} has non-finite reprojection error")
        color = [int(value) for value in point.color]
        if len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise ValueError(f"invalid RGB color for point {point_id}")

    mean_error = float(reconstruction.compute_mean_reprojection_error())
    mean_track = float(reconstruction.compute_mean_track_length())
    if not math.isfinite(mean_error) or not math.isfinite(mean_track) or mean_track <= 0:
        raise ValueError("non-finite or non-positive reconstruction metrics")
    return {
        "registered_images": len(registered),
        "registered_image_names": sorted(names),
        "cameras": len(reconstruction.cameras),
        "points3D": len(points),
        "mean_reprojection_error_px": mean_error,
        "mean_track_length": mean_track,
    }


def model_directories(root: Path) -> list[Path]:
    root = Path(root)
    candidates = [root]
    if root.is_dir():
        candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    return [
        path
        for path in candidates
        if any(
            all((path / f"{stem}{suffix}").is_file() for stem in ("cameras", "images", "points3D"))
            for suffix in (".bin", ".txt")
        )
    ]


def largest_valid_model(pycolmap: Any, root: Path) -> tuple[Any, Path, dict[str, Any]]:
    valid: list[tuple[Any, Path, dict[str, Any]]] = []
    errors: list[str] = []
    for model_dir in model_directories(root):
        try:
            _model_file_manifest(model_dir)
            reconstruction = pycolmap.Reconstruction(model_dir)
            metrics = reconstruction_metrics(reconstruction)
            valid.append((reconstruction, model_dir, metrics))
        except Exception as error:  # retain diagnostics for each disconnected component
            errors.append(f"{model_dir}: {error}")
    if not valid:
        detail = "; ".join(errors) if errors else "no COLMAP model directories found"
        raise RuntimeError(f"no valid sparse reconstruction in {root}: {detail}")
    return max(valid, key=lambda row: (row[2]["registered_images"], row[2]["points3D"]))


def _validate_model_with(
    pycolmap: Any, model_dir: Path, *, expected_images: Sequence[str] | None = None
) -> dict[str, Any]:
    safe_model_dir = require_under(Path(model_dir), SFM_DATA_ROOT, "SfM model")
    _reconstruction, selected, metrics = largest_valid_model(pycolmap, safe_model_dir)
    if expected_images is not None:
        expected = set(expected_images)
        actual = set(metrics["registered_image_names"])
        if not actual <= expected:
            raise ValueError(f"model registers unknown images: {sorted(actual - expected)[:10]}")
    return {
        **metrics,
        "model_dir": str(selected),
        "model_files": _model_file_manifest(selected),
        "pycolmap": pycolmap.__version__,
    }


def _model_file_manifest(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Fingerprint the three files that define a COLMAP sparse model."""

    selected_suffix = next(
        (
            suffix
            for suffix in (".bin", ".txt")
            if all(
                (model_dir / f"{stem}{suffix}").is_file()
                for stem in ("cameras", "images", "points3D")
            )
        ),
        None,
    )
    if selected_suffix is None:
        raise RuntimeError(f"incomplete COLMAP model file set: {model_dir}")
    result: dict[str, dict[str, Any]] = {}
    for stem in ("cameras", "images", "points3D"):
        path = model_dir / f"{stem}{selected_suffix}"
        _safe_destination(path, "SfM model file")
        result[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def validate_model(
    model_dir: Path, *, expected_images: Sequence[str] | None = None
) -> dict[str, Any]:
    """Load and geometrically validate an existing COLMAP sparse model."""

    import pycolmap

    _validate_pycolmap(pycolmap)
    return _validate_model_with(pycolmap, model_dir, expected_images=expected_images)


def validate_colored_ply(path: Path, expected_points: int) -> dict[str, Any]:
    """Check the exported PLY header, colour fields, and vertex count."""

    safe_path = _safe_destination(path, "SfM coloured point cloud")
    if not safe_path.is_file() or safe_path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty coloured PLY: {safe_path}")
    with safe_path.open("rb") as stream:
        header_bytes = stream.read(64 * 1024)
    end = header_bytes.find(b"end_header")
    if not header_bytes.startswith(b"ply\n") and not header_bytes.startswith(b"ply\r\n"):
        raise RuntimeError(f"invalid PLY magic in {safe_path}")
    if end < 0:
        raise RuntimeError(f"PLY header is missing end_header: {safe_path}")
    header = header_bytes[: end + len(b"end_header")].decode("ascii", errors="strict").replace(
        "\r\n", "\n"
    )
    match = re.search(r"^element vertex (\d+)$", header, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"PLY header has no vertex count: {safe_path}")
    vertices = int(match.group(1))
    if vertices != expected_points:
        raise RuntimeError(
            f"PLY/model point-count mismatch: PLY={vertices}, model={expected_points}"
        )
    for channel in ("red", "green", "blue"):
        if re.search(rf"^property (?:uchar|uint8) {channel}$", header, flags=re.MULTILINE) is None:
            raise RuntimeError(f"PLY has no uint8 {channel} channel: {safe_path}")
    return {
        "path": str(safe_path),
        "bytes": safe_path.stat().st_size,
        "sha256": sha256_file(safe_path),
        "vertices": vertices,
    }


def _identity(
    config: SfMConfig, image_records: Sequence[Mapping[str, Any]], pycolmap: Any, device: str
) -> dict[str, Any]:
    recipe = {
        "engine": ENGINE_RECIPE,
        "engine_source_sha256": sha256_file(Path(__file__)),
        "pycolmap": pycolmap.__version__,
        "pycolmap_has_cuda": _has_cuda(pycolmap),
        "device": device,
        "camera_model": config.camera_model,
        "camera_mode": config.camera_mode,
        "matcher": config.matcher,
        "max_image_size": config.max_image_size,
        "max_num_features": config.max_num_features,
        "num_threads": config.num_threads,
        "random_seed": config.random_seed,
        "first_octave": 0,
        "guided_matching": False,
        "sequential_overlap": 10,
        "sequential_loop_detection": False,
        "mapping_extract_colors": True,
        "images": list(image_records),
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "recipe": recipe}


def _call_matcher(pycolmap: Any, config: SfMConfig, database: Path, device: Any) -> None:
    common = {
        "database_path": database,
        "matching_options": {"num_threads": config.num_threads, "guided_matching": False},
        "device": device,
    }
    if config.matcher == "exhaustive":
        pycolmap.match_exhaustive(**common)
    elif config.matcher == "sequential":
        pycolmap.match_sequential(
            **common, pairing_options={"overlap": 10, "loop_detection": False}
        )
    else:  # config validation makes this defensive branch unreachable
        raise ValueError(f"unsupported matcher: {config.matcher}")


@contextmanager
def _experiment_lock(experiment_root: Path) -> Iterator[None]:
    """Serialize writers using an OS lock released after a process crash."""

    experiment_root.mkdir(parents=True, exist_ok=True)
    lock_path = experiment_root / ".run.lock"
    _safe_destination(lock_path, "SfM experiment lock")
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another process is using experiment {experiment_root.name}"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(
            f"pid={os.getpid()} host={socket.gethostname()} "
            f"job={os.environ.get('SLURM_JOB_ID')}\n"
        )
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _safe_destination(path: Path, label: str) -> Path:
    """Resolve each generated destination to reject planted symlink escapes."""

    safe = require_under(path, SFM_DATA_ROOT, label)
    cursor = path
    while cursor != SFM_DATA_ROOT and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError(f"{label} cannot use symlink component: {cursor}")
        cursor = cursor.parent
    return safe


def _next_attempt(attempts_root: Path) -> tuple[str, Path]:
    _safe_destination(attempts_root, "SfM attempts")
    attempts_root.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name)
        for path in attempts_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.isdigit()
    ]
    identifier = f"{max(numbers, default=0) + 1:04d}"
    path = attempts_root / identifier
    _safe_destination(path, "SfM attempt")
    path.mkdir(exist_ok=False)
    return identifier, path


def _base_environment(pycolmap: Any, selected_device: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pycolmap": pycolmap.__version__,
        "pycolmap_has_cuda": _has_cuda(pycolmap),
        "selected_device": selected_device,
    }


def _runtime_environment(pycolmap: Any, selected_device: str) -> dict[str, Any]:
    environment = _base_environment(pycolmap, selected_device)
    if selected_device == "cuda":
        environment["cuda"] = cuda_runtime_identity(pycolmap)
    return environment


def _metrics_agree(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    keys = ("registered_images", "registered_image_names", "cameras", "points3D")
    return all(first.get(key) == second.get(key) for key in keys)


def _cache_artifacts_agree(
    metrics: Mapping[str, Any], ply: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    recorded_metrics = receipt.get("metrics", {})
    recorded_ply = receipt.get("ply", {})
    return (
        _metrics_agree(metrics, recorded_metrics)
        and metrics.get("model_files") == recorded_metrics.get("model_files")
        and ply.get("sha256") == recorded_ply.get("sha256")
        and ply.get("bytes") == recorded_ply.get("bytes")
        and ply.get("vertices") == recorded_ply.get("vertices")
    )


def _validate_outputs(
    pycolmap: Any, output_root: Path, expected_images: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _safe_destination(output_root, "SfM outputs")
    model = output_root / "colmap" / "sparse" / "0"
    metrics = _validate_model_with(pycolmap, model, expected_images=expected_images)
    ply = validate_colored_ply(output_root / "sparse_colored.ply", metrics["points3D"])
    return metrics, ply


def _receipt(
    *,
    config: SfMConfig,
    identity: Mapping[str, Any],
    input_images: int,
    attempt_id: str | None,
    elapsed_seconds: float | None,
    selected_component: str | None,
    output_root: Path,
    database: Path | None,
    metrics: Mapping[str, Any],
    ply: Mapping[str, Any],
    environment: Mapping[str, Any],
    recovered: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "complete": True,
        "cache_hit": False,
        "recovered_after_interruption": recovered,
        "identity_sha256": identity["sha256"],
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed_seconds,
        "input_images": input_images,
        "attempt_id": attempt_id,
        "selected_component": selected_component,
        "outputs": {
            "model": str(output_root / "colmap" / "sparse" / "0"),
            "colored_ply": str(output_root / "sparse_colored.ply"),
            "database": str(database) if database is not None else None,
        },
        "metrics": dict(metrics),
        "ply": dict(ply),
        "environment": dict(environment),
        "config": {**asdict(config), "image_dir": str(config.image_dir.expanduser().resolve())},
    }


def _mark_manifest_complete(manifest_path: Path, receipt: Mapping[str, Any]) -> None:
    """Keep the immutable identity while making completion state unambiguous."""

    manifest = _read_json_object(manifest_path, "experiment manifest")
    if manifest.get("identity", {}).get("sha256") != receipt.get("identity_sha256"):
        raise RuntimeError("cannot complete a manifest whose identity differs from the receipt")
    manifest.update(
        {
            "complete": True,
            "completed_utc": receipt.get("completed_utc"),
            "receipt": str(manifest_path.parent / "receipt.json"),
        }
    )
    atomic_json(manifest_path, manifest)


def run_reconstruction(
    config: SfMConfig, *, resume: bool = True, dry_run: bool = False
) -> dict[str, Any]:
    """Run, recover, or safely reload one sparse-SfM experiment.

    An interrupted attempt is retained for diagnosis. ``resume=True`` starts a
    clean attempt rather than appending to a possibly inconsistent COLMAP
    database. A final output is installed by one same-filesystem rename only
    after model reload, PLY validation, and a second input hash pass succeed.
    """

    config = config.validate()
    image_dir = config.image_dir.expanduser().resolve()
    experiment_root = require_under(
        SFM_DATA_ROOT / "experiments" / config.experiment_name,
        SFM_DATA_ROOT / "experiments",
        "SfM experiment",
    )
    images = discover_images(image_dir)
    records = image_manifest(image_dir, images)

    if dry_run:
        return {
            "dry_run": True,
            "experiment_root": str(experiment_root),
            "input_images": len(images),
            "input_bytes": sum(record["bytes"] for record in records),
        }

    import pycolmap

    _validate_pycolmap(pycolmap)
    selected_device, device = resolve_device(pycolmap, config.device)
    identity = _identity(config, records, pycolmap, selected_device)
    relative_names = [path.relative_to(image_dir).as_posix() for path in images]
    manifest_path = experiment_root / "manifest.json"
    receipt_path = experiment_root / "receipt.json"
    output_root = experiment_root / "outputs"

    with _experiment_lock(experiment_root):
        _safe_destination(manifest_path, "SfM experiment manifest")
        _safe_destination(receipt_path, "SfM completion receipt")
        _safe_destination(output_root, "SfM outputs")
        existing_manifest: dict[str, Any] | None = None
        if manifest_path.is_file():
            existing_manifest = _read_json_object(manifest_path, "experiment manifest")
            stored_identity = existing_manifest.get("identity", {}).get("sha256")
            if stored_identity != identity["sha256"]:
                raise RuntimeError(
                    f"experiment {config.experiment_name!r} already describes different "
                    "inputs/settings; use a new experiment name"
                )
        else:
            unexplained = [path for path in experiment_root.iterdir() if path.name != ".run.lock"]
            if unexplained:
                raise RuntimeError(
                    f"experiment directory has no manifest and is not empty: {experiment_root}"
                )

        if receipt_path.is_file():
            receipt = _read_json_object(receipt_path, "completion receipt")
            if not receipt.get("complete") or receipt.get("identity_sha256") != identity["sha256"]:
                raise RuntimeError(f"invalid or mismatched completion receipt: {receipt_path}")
            metrics, ply = _validate_outputs(pycolmap, output_root, relative_names)
            if not _cache_artifacts_agree(metrics, ply, receipt):
                raise RuntimeError(
                    "cached receipt and saved artifacts disagree; refusing recomputation"
                )
            if not existing_manifest.get("complete"):
                _mark_manifest_complete(manifest_path, receipt)
            return {**receipt, "cache_hit": True, "metrics": metrics, "ply": ply}

        if existing_manifest is not None and not resume:
            raise FileExistsError(f"incomplete experiment requires --resume: {experiment_root}")

        # The output rename may have completed immediately before termination.
        if output_root.exists():
            if existing_manifest is None:
                raise RuntimeError("outputs exist without an identity manifest")
            metrics, ply = _validate_outputs(pycolmap, output_root, relative_names)
            recovered_receipt = _receipt(
                config=config,
                identity=identity,
                input_images=len(images),
                attempt_id=None,
                elapsed_seconds=None,
                selected_component=None,
                output_root=output_root,
                database=None,
                metrics=metrics,
                ply=ply,
                environment=_base_environment(pycolmap, selected_device),
                recovered=True,
            )
            atomic_json(receipt_path, recovered_receipt)
            _mark_manifest_complete(manifest_path, recovered_receipt)
            return recovered_receipt

        environment = _runtime_environment(pycolmap, selected_device)
        if existing_manifest is None:
            atomic_json(
                manifest_path,
                {
                    "schema_version": 2,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "complete": False,
                    "identity": identity,
                    "experiment_root": str(experiment_root),
                    "config": {**asdict(config), "image_dir": str(image_dir)},
                },
            )

        attempt_id, attempt_root = _next_attempt(experiment_root / "attempts")
        work = _safe_destination(attempt_root / "work", "SfM attempt work")
        database = _safe_destination(work / "database.db", "SfM database")
        components = _safe_destination(work / "sparse_components", "SfM components")
        work.mkdir(parents=True, exist_ok=True)
        components.mkdir(parents=True, exist_ok=True)
        attempt_state: dict[str, Any] = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "identity_sha256": identity["sha256"],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": "starting",
            "environment": environment,
        }
        attempt_state_path = attempt_root / "status.json"
        atomic_json(attempt_state_path, attempt_state)
        started = time.monotonic()

        try:
            attempt_state["stage"] = "extracting_features"
            atomic_json(attempt_state_path, attempt_state)
            pycolmap.set_random_seed(config.random_seed)
            pycolmap.extract_features(
                database_path=database,
                image_path=image_dir,
                image_names=relative_names,
                camera_mode=_camera_mode(pycolmap, config.camera_mode),
                reader_options={"camera_model": config.camera_model},
                extraction_options={
                    "max_image_size": config.max_image_size,
                    "num_threads": config.num_threads,
                    "sift": {"max_num_features": config.max_num_features, "first_octave": 0},
                },
                device=device,
            )

            attempt_state["stage"] = "matching_features"
            atomic_json(attempt_state_path, attempt_state)
            _call_matcher(pycolmap, config, database, device)

            attempt_state["stage"] = "incremental_mapping"
            atomic_json(attempt_state_path, attempt_state)
            models = pycolmap.incremental_mapping(
                database_path=database,
                image_path=image_dir,
                output_path=components,
                options={
                    "num_threads": config.num_threads,
                    "extract_colors": True,
                    "random_seed": config.random_seed,
                },
            )
            if not models:
                raise RuntimeError("PyCOLMAP produced no sparse connected component")
            reconstruction, selected_component, metrics = largest_valid_model(pycolmap, components)

            attempt_state["stage"] = "validating_staged_outputs"
            atomic_json(attempt_state_path, attempt_state)
            staged_output = _safe_destination(
                attempt_root / "output_staging", "SfM staged output"
            )
            saved_model = staged_output / "colmap" / "sparse" / "0"
            saved_model.mkdir(parents=True, exist_ok=False)
            reconstruction.write(saved_model)
            sparse_ply = staged_output / "sparse_colored.ply"
            reconstruction.export_PLY(sparse_ply)
            reloaded = _validate_model_with(pycolmap, saved_model, expected_images=relative_names)
            if not _metrics_agree(reloaded, metrics):
                raise RuntimeError("saved sparse model failed round-trip validation")
            staged_ply = validate_colored_ply(sparse_ply, reloaded["points3D"])

            final_records = image_manifest(image_dir, images)
            if final_records != records:
                raise RuntimeError("input images changed during reconstruction")

            atomic_json(
                attempt_root / "success.json",
                {
                    "identity_sha256": identity["sha256"],
                    "selected_component": str(selected_component),
                    "metrics": reloaded,
                    "ply": staged_ply,
                },
            )
            if output_root.exists():
                raise RuntimeError(f"refusing to replace existing outputs: {output_root}")
            staged_output.rename(output_root)
            final_metrics, final_ply = _validate_outputs(pycolmap, output_root, relative_names)
            receipt = _receipt(
                config=config,
                identity=identity,
                input_images=len(images),
                attempt_id=attempt_id,
                elapsed_seconds=time.monotonic() - started,
                selected_component=str(selected_component),
                output_root=output_root,
                database=database,
                metrics=final_metrics,
                ply=final_ply,
                environment=environment,
            )
            atomic_json(receipt_path, receipt)
            _mark_manifest_complete(manifest_path, receipt)
            attempt_state.update(
                {
                    "stage": "complete",
                    "completed_utc": receipt["completed_utc"],
                    "receipt": str(receipt_path),
                }
            )
            atomic_json(attempt_state_path, attempt_state)
            return receipt
        except Exception as error:
            attempt_state.update(
                {
                    "stage": "failed",
                    "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            try:
                atomic_json(attempt_state_path, attempt_state)
            except Exception:
                pass
            raise
