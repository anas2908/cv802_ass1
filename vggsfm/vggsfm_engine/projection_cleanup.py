"""Geometry helpers for derived mask/corridor cleanup in VGGSfM's own frame."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def project_original_pixels(
    image: Any,
    camera: Any,
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world XYZ to original distorted pixels and return an in-bounds mask.

    This follows PyCOLMAP 3.10's API: ``cam_from_world`` is a ``Rigid3d``
    property, while ``img_from_cam`` accepts normalized Nx2 coordinates.
    """

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    camera_xyz = np.asarray(image.cam_from_world * points, dtype=np.float64)
    if camera_xyz.shape != points.shape:
        raise ValueError("cam_from_world returned an unexpected shape")
    positive = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0.0)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    indices = np.flatnonzero(positive)
    if indices.size:
        normalized = camera_xyz[indices, :2] / camera_xyz[indices, 2, None]
        projected = np.asarray(camera.img_from_cam(normalized), dtype=np.float64)
        if projected.shape != (len(indices), 2):
            raise ValueError("img_from_cam returned an unexpected shape")
        pixels[indices] = projected
    finite = np.isfinite(pixels).all(axis=1)
    usable = positive & finite
    usable &= (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(camera.width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(camera.height))
    )
    return pixels, usable


def _distance_squared_to_segments(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Return each point's squared distance to its nearest line segment."""

    direction = ends - starts
    length_squared = np.einsum("ij,ij->i", direction, direction)
    if np.any(length_squared <= 0.0):
        raise ValueError("corridor polylines cannot contain zero-length segments")
    difference = points[:, None, :] - starts[None, :, :]
    parameter = np.einsum("nsi,si->ns", difference, direction) / length_squared[None, :]
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = starts[None, :, :] + parameter[:, :, None] * direction[None, :, :]
    residual = points[:, None, :] - closest
    return np.min(np.einsum("nsi,nsi->ns", residual, residual), axis=1)


def corridor_hits(
    pixels: np.ndarray,
    usable: np.ndarray,
    *,
    width: int,
    height: int,
    groups: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Test continuous original-image pixels against the union of reviewed corridors."""

    points = np.asarray(pixels, dtype=np.float64)
    usable_array = np.asarray(usable, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or usable_array.shape != (len(points),):
        raise ValueError("pixels/usable shapes disagree")
    if width <= 0 or height <= 0 or not groups:
        raise ValueError("invalid camera dimensions or empty corridors")
    hits = np.zeros(len(points), dtype=bool)
    candidates = np.flatnonzero(usable_array)
    if not candidates.size:
        return hits
    query = points[candidates]
    for group in groups.values():
        half_width = float(group["half_width_fraction_of_image_width"]) * width
        if not 0.0 < half_width <= 0.05 * width:
            raise ValueError("corridor half-width is outside the reviewed range")
        segment_starts: list[list[float]] = []
        segment_ends: list[list[float]] = []
        for polyline in group["polylines"]:
            scaled = np.asarray(polyline, dtype=np.float64) * np.array([width, height])
            if scaled.ndim != 2 or scaled.shape[1] != 2 or len(scaled) < 2:
                raise ValueError("invalid normalized corridor polyline")
            if not np.isfinite(scaled).all() or np.any(scaled < 0.0) or np.any(
                scaled > np.array([width, height])
            ):
                raise ValueError("corridor coordinate is outside normalized image bounds")
            segment_starts.extend(scaled[:-1].tolist())
            segment_ends.extend(scaled[1:].tolist())
        distances = _distance_squared_to_segments(
            query,
            np.asarray(segment_starts, dtype=np.float64),
            np.asarray(segment_ends, dtype=np.float64),
        )
        hits[candidates] |= distances <= half_width * half_width
    return hits


def separated_view_confirmation(
    xyz: np.ndarray,
    camera_centers: np.ndarray,
    per_view_hits: np.ndarray,
    *,
    minimum_views: int = 3,
    minimum_pairwise_angle_degrees: float = 15.0,
) -> np.ndarray:
    """Require a hit triple whose camera-to-point rays are pairwise separated.

    Only the reviewed six or similarly small annotation view sets are intended;
    enumerating their triples is explicit and auditable.  This is geometric
    baseline support, not an occlusion/visibility proof.
    """

    points = np.asarray(xyz, dtype=np.float64)
    centers = np.asarray(camera_centers, dtype=np.float64)
    hits = np.asarray(per_view_hits, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("camera_centers must have shape (V, 3)")
    if hits.shape != (len(centers), len(points)):
        raise ValueError("per_view_hits must have shape (V, N)")
    if minimum_views != 3:
        raise ValueError("reviewed crutch policy currently requires exactly a triple")
    if not 0.0 < minimum_pairwise_angle_degrees < 180.0:
        raise ValueError("invalid angular separation")
    if len(centers) < minimum_views:
        return np.zeros(len(points), dtype=bool)

    rays = points[None, :, :] - centers[:, None, :]
    norms = np.linalg.norm(rays, axis=2)
    finite = np.isfinite(rays).all(axis=2) & np.isfinite(norms) & (norms > 1e-12)
    unit = np.divide(
        rays,
        norms[:, :, None],
        out=np.zeros_like(rays),
        where=finite[:, :, None],
    )
    maximum_dot = float(np.cos(np.deg2rad(minimum_pairwise_angle_degrees)))
    confirmed = np.zeros(len(points), dtype=bool)
    for first, second, third in combinations(range(len(centers)), 3):
        candidate = hits[first] & hits[second] & hits[third]
        candidate &= finite[first] & finite[second] & finite[third]
        if not candidate.any():
            continue
        for left, right in ((first, second), (first, third), (second, third)):
            dots = np.einsum("ij,ij->i", unit[left], unit[right])
            candidate &= dots <= maximum_dot + 1e-12
        confirmed |= candidate
    return confirmed
