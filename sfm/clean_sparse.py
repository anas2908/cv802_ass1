#!/usr/bin/env python3
"""Create a deterministic quality-cleaned display PLY from a sparse model.

This is deliberately a derived visualization: camera poses, calibration and
the source COLMAP model are never modified.  It removes weak two-view points,
high-reprojection-error points and extreme coordinate outliers.  It is not a
semantic person mask and does not claim to isolate clothing or crutches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np
import pycolmap


RECIPE = "cv802-sparse-quality-cleanup-v1"
MAX_REPROJECTION_ERROR = 3.0
MIN_TRACK_VIEWS = 3
ROBUST_IQR_LIMIT = 6.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_under(path: Path, root: Path, label: str, *, exists: bool = False) -> Path:
    resolved_root = root.resolve()
    resolved = path.expanduser().resolve(strict=exists)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {resolved_root}; got {resolved}") from error
    return resolved


def quality_mask(
    xyz: np.ndarray,
    errors: np.ndarray,
    track_views: np.ndarray,
    *,
    max_error: float = MAX_REPROJECTION_ERROR,
    min_track_views: int = MIN_TRACK_VIEWS,
    robust_iqr_limit: float = ROBUST_IQR_LIMIT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select geometrically supported points with a broad robust spatial gate."""

    xyz = np.asarray(xyz, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    track_views = np.asarray(track_views, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or errors.shape != (len(xyz),) or track_views.shape != (len(xyz),):
        raise ValueError("xyz, error and track arrays have incompatible shapes")
    if len(xyz) < 3:
        raise ValueError("cleanup requires at least three sparse points")
    lower_quartile, upper_quartile = np.quantile(xyz, [0.25, 0.75], axis=0)
    iqr = upper_quartile - lower_quartile
    coordinate_range = np.ptp(xyz, axis=0)
    active_axes = iqr > np.maximum(coordinate_range * 1e-6, 1e-9)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(errors)
    reprojection = errors <= max_error
    supported = track_views >= min_track_views
    spatial = np.ones(len(xyz), dtype=bool)
    for axis in range(3):
        if active_axes[axis]:
            spatial &= (
                (xyz[:, axis] >= lower_quartile[axis] - robust_iqr_limit * iqr[axis])
                & (xyz[:, axis] <= upper_quartile[axis] + robust_iqr_limit * iqr[axis])
            )
    keep = finite & reprojection & supported & spatial
    return keep, {
        "finite_points": int(finite.sum()),
        "within_reprojection_limit": int((finite & reprojection).sum()),
        "minimum_track_views_met": int((finite & supported).sum()),
        "within_robust_spatial_gate": int((finite & spatial).sum()),
        "lower_quartile_xyz": lower_quartile.tolist(),
        "upper_quartile_xyz": upper_quartile.tolist(),
        "interquartile_range_xyz": iqr.tolist(),
        "spatial_gate_active_axes": active_axes.tolist(),
    }


def write_binary_ply(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(header)
            for point, color in zip(xyz, colors, strict=True):
                stream.write(struct.pack("<fffBBB", *map(float, point), *map(int, color)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    configured = Path(os.environ.get("CV802_DATA_ROOT", "")).expanduser()
    if not configured.is_absolute():
        parser.error("CV802_DATA_ROOT must be an explicit absolute path")
    method_root = configured.resolve() / "sfm"
    model_path = require_under(args.model, method_root, "source model", exists=True)
    output = require_under(args.output, method_root, "cleaned PLY")
    receipt = require_under(args.receipt, method_root, "cleanup receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or receipt.is_symlink():
        raise ValueError("cleanup destinations cannot be symlinks")

    model_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(model_path.iterdir())
        if path.is_file()
    }
    source_identity = {
        "recipe": RECIPE,
        "pycolmap": pycolmap.__version__,
        "model_files": model_files,
        "thresholds": {
            "max_reprojection_error_px": MAX_REPROJECTION_ERROR,
            "minimum_distinct_track_views": MIN_TRACK_VIEWS,
            "robust_coordinate_iqr_limit": ROBUST_IQR_LIMIT,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or receipt.exists():
        if not output.is_file() or not receipt.is_file():
            raise RuntimeError("Incomplete cleanup result exists; preserving it for inspection")
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint or saved.get("output", {}).get("sha256") != sha256_file(output):
            raise RuntimeError("Existing cleanup result does not match this source model")
        print(json.dumps(saved, indent=2, sort_keys=True))
        return 0

    reconstruction = pycolmap.Reconstruction(model_path)
    point_ids = sorted(reconstruction.points3D)
    points = [reconstruction.points3D[identifier] for identifier in point_ids]
    xyz = np.asarray([point.xyz for point in points], dtype=np.float64)
    colors = np.asarray([point.color for point in points], dtype=np.uint8)
    errors = np.asarray([point.error for point in points], dtype=np.float64)
    tracks = np.asarray(
        [len({element.image_id for element in point.track.elements}) for point in points],
        dtype=np.int64,
    )
    keep, gates = quality_mask(xyz, errors, tracks)
    retained = int(keep.sum())
    if retained < 3 or retained < len(points) // 20:
        raise RuntimeError(f"Cleanup retained an implausibly small cloud: {retained}/{len(points)}")
    write_binary_ply(output, xyz[keep], colors[keep])
    payload = {
        "schema_version": 1,
        "recipe": RECIPE,
        "fingerprint": fingerprint,
        "source_model": str(model_path),
        "source_identity": source_identity,
        "source_points": len(points),
        "retained_points": retained,
        "removed_points": len(points) - retained,
        "selection_gates": gates,
        "semantics": "generic geometric quality cleanup; not person segmentation",
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    atomic_json(receipt, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
