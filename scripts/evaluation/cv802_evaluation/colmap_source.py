"""Read COLMAP models through the project's already-tested source helpers.

No mutable engine state is imported from the data filesystem.  Helper roots
are anchored to this checkout, injected at the front of ``sys.path`` only for
the import, and the imported module paths are verified before use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np


class ModelError(RuntimeError):
    """A model or pose record is invalid for evaluation."""


CODE_ROOT = Path(__file__).resolve().parents[3]
MVS_SOURCE_ROOT = CODE_ROOT / "mvs" / "src"
VGGSFM_SOURCE_ROOT = CODE_ROOT / "vggsfm"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _anchored_import(source_root: Path, module_name: str) -> Any:
    root = source_root.resolve(strict=True)
    sys.path.insert(0, str(root))
    try:
        module = __import__(module_name, fromlist=["*"])
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    module_file = Path(module.__file__).resolve(strict=True)
    if not _within(module_file, root):
        raise ModelError(
            f"refusing unexpected {module_name} import from {module_file}; expected {root}"
        )
    return module


_mvs_colmap = _anchored_import(MVS_SOURCE_ROOT, "cv802_mvs.colmap_model")
_vgg_colmap = _anchored_import(VGGSFM_SOURCE_ROOT, "vggsfm_engine.colmap")


@dataclass(frozen=True)
class CameraPose:
    image_id: int
    name: str
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    center: np.ndarray


def quaternion_rotation(qvec: tuple[float, float, float, float]) -> np.ndarray:
    """Convert COLMAP's Hamilton ``(qw,qx,qy,qz)`` world-to-camera qvec."""

    values = np.asarray(qvec, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ModelError("COLMAP quaternion must contain four finite values")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ModelError("COLMAP quaternion has zero or invalid norm")
    if not 0.99 <= norm <= 1.01:
        raise ModelError(f"COLMAP quaternion norm {norm:.12g} is not approximately 1")
    qw, qx, qy, qz = values / norm
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(rotation).all() or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=1e-7, atol=1e-7
    ):
        raise ModelError("COLMAP quaternion produced an invalid rotation")
    return rotation


def camera_center(
    qvec: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
) -> np.ndarray:
    """Return world-space center ``C=-R^T t`` for COLMAP's ``x_cam=RX+t``."""

    translation = np.asarray(tvec, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ModelError("COLMAP translation must contain three finite values")
    center = -(quaternion_rotation(qvec).T @ translation)
    if not np.isfinite(center).all():
        raise ModelError("computed camera center contains NaN or infinity")
    return center


def load_binary_camera_poses(model_directory: Path) -> tuple[CameraPose, ...]:
    """Parse and validate one complete binary COLMAP model."""

    try:
        model = _mvs_colmap.load_model_records(model_directory)
    except Exception as error:
        raise ModelError(f"cannot read COLMAP model {model_directory}: {error}") from error
    if model.format != "binary":
        raise ModelError(
            f"evaluation requires an actual binary COLMAP model; got {model.format} at "
            f"{model.directory}"
        )
    poses: list[CameraPose] = []
    for image in model.images:
        qvec = tuple(float(value) for value in image.qvec)
        tvec = tuple(float(value) for value in image.tvec)
        poses.append(
            CameraPose(
                image_id=int(image.image_id),
                name=str(image.name),
                camera_id=int(image.camera_id),
                qvec=qvec,
                tvec=tvec,
                center=camera_center(qvec, tvec),
            )
        )
    if len(poses) < 3:
        raise ModelError(
            f"alignment requires at least three registered cameras; found {len(poses)}"
        )
    return tuple(poses)


def point_reader(model_directory: Path) -> tuple[int, Iterator[Any]]:
    """Stream validated colored points using the VGGSfM binary-model helper."""

    points_path = model_directory / "points3D.bin"
    try:
        return _vgg_colmap.iter_points3d(points_path)
    except Exception as error:
        raise ModelError(f"cannot read colored COLMAP points {points_path}: {error}") from error
