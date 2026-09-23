"""Proper 3-D similarity alignment for independently reconstructed cameras.

The implementation follows Umeyama's least-squares estimator.  It always
returns a proper rotation (determinant +1); a mirror reflection is never
silently folded into the reported transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np


class AlignmentError(RuntimeError):
    """The camera correspondences cannot define a trustworthy alignment."""


def _points(value: Any, *, label: str, minimum: int = 3) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise AlignmentError(f"{label} must have shape (N, 3); got {points.shape}")
    if points.shape[0] < minimum:
        raise AlignmentError(
            f"{label} needs at least {minimum} points; got {points.shape[0]}"
        )
    if not np.isfinite(points).all():
        raise AlignmentError(f"{label} contains NaN or infinity")
    return points


def _geometry_rank(points: np.ndarray) -> int:
    centered = points - np.mean(points, axis=0)
    return int(np.linalg.matrix_rank(centered))


@dataclass(frozen=True)
class SimilarityTransform:
    """Map a row of source points with ``scale * (R @ x) + translation``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    reflection_correction_applied: bool

    def apply(self, points: Any) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise AlignmentError(
                f"points to transform must end in dimension 3; got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise AlignmentError("points to transform contain NaN or infinity")
        transformed = self.scale * (values @ self.rotation.T) + self.translation
        if not np.isfinite(transformed).all():
            raise AlignmentError("similarity transform produced NaN or infinity")
        return transformed

    def as_dict(self) -> dict[str, Any]:
        determinant = float(np.linalg.det(self.rotation))
        return {
            "convention": "e10_point = scale * rotation @ vggsfm_point + translation",
            "scale_e10_units_per_vggsfm_unit": float(self.scale),
            "rotation": self.rotation.tolist(),
            "rotation_determinant": determinant,
            "translation_e10_units": self.translation.tolist(),
            "reflection_correction_applied": self.reflection_correction_applied,
        }


def umeyama_similarity(source: Any, target: Any) -> SimilarityTransform:
    """Estimate the least-squares proper similarity mapping source to target.

    Source/target rows are corresponding 3-D points.  Two-dimensional
    geometric rank is accepted because camera trajectories are often nearly
    planar, but a line or a single location is underconstrained in 3-D.
    """

    source_points = _points(source, label="source")
    target_points = _points(target, label="target")
    if source_points.shape != target_points.shape:
        raise AlignmentError(
            "source and target must contain the same number of 3-D points"
        )
    if _geometry_rank(source_points) < 2:
        raise AlignmentError("source correspondences are collinear or coincident")
    if _geometry_rank(target_points) < 2:
        raise AlignmentError("target correspondences are collinear or coincident")

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if not math.isfinite(source_variance) or source_variance <= np.finfo(float).eps:
        raise AlignmentError("source correspondences have zero usable variance")

    covariance = (target_centered.T @ source_centered) / source_points.shape[0]
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    unconstrained_determinant = float(np.linalg.det(left @ right_transpose))
    signs = np.ones(3, dtype=np.float64)
    reflection_corrected = unconstrained_determinant < 0.0
    if reflection_corrected:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.dot(singular_values, signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)

    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(scale) or scale <= 0.0:
        raise AlignmentError(f"estimated similarity scale is invalid: {scale!r}")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise AlignmentError("estimated similarity contains NaN or infinity")
    if not math.isclose(determinant, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise AlignmentError(
            f"estimated rotation is not proper; determinant={determinant:.12g}"
        )
    orthogonality = rotation.T @ rotation
    if not np.allclose(orthogonality, np.eye(3), rtol=1e-7, atol=1e-7):
        raise AlignmentError("estimated rotation is not orthonormal")
    return SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        reflection_correction_applied=reflection_corrected,
    )


def residual_statistics(
    residuals: Any, *, reference_radius: float
) -> dict[str, float | int]:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise AlignmentError("residual array must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise AlignmentError("residuals must be finite and non-negative")
    if not math.isfinite(reference_radius) or reference_radius <= 0.0:
        raise AlignmentError("reference radius must be finite and positive")
    rmse = float(np.sqrt(np.mean(values * values)))
    median = float(np.median(values))
    p90 = float(np.percentile(values, 90))
    maximum = float(np.max(values))
    return {
        "count": int(values.size),
        "rmse_e10_units": rmse,
        "median_e10_units": median,
        "p90_e10_units": p90,
        "max_e10_units": maximum,
        "rmse_over_e10_camera_rms_radius": rmse / reference_radius,
        "median_over_e10_camera_rms_radius": median / reference_radius,
        "p90_over_e10_camera_rms_radius": p90 / reference_radius,
        "max_over_e10_camera_rms_radius": maximum / reference_radius,
    }


@dataclass(frozen=True)
class RobustAlignment:
    transform: SimilarityTransform
    residuals: np.ndarray
    inlier_mask: np.ndarray
    threshold_e10_units: float
    reference_radius_e10_units: float
    robust_for_point_cloud: bool
    robustness_reasons: tuple[str, ...]
    all_metrics: dict[str, float | int]
    inlier_metrics: dict[str, float | int]


def _sample_indices(count: int, max_trials: int, seed: int) -> list[tuple[int, int, int]]:
    total = math.comb(count, 3)
    if total <= max_trials:
        return list(itertools.combinations(range(count), 3))
    generator = np.random.default_rng(seed)
    samples: set[tuple[int, int, int]] = set()
    while len(samples) < max_trials:
        sample = tuple(sorted(int(value) for value in generator.choice(count, 3, replace=False)))
        samples.add(sample)
    return sorted(samples)


def robust_similarity_alignment(
    source: Any,
    target: Any,
    *,
    threshold_ratio: float = 0.05,
    min_inlier_ratio: float = 0.60,
    min_inliers: int = 6,
    max_trials: int = 2000,
    seed: int = 0,
) -> RobustAlignment:
    """Fit a deterministic RANSAC+Umeyama alignment and assess PLY safety.

    The inlier threshold is relative to the E10 matched-camera RMS radius, so
    it remains meaningful despite the reconstruction's arbitrary scale.  The
    final reported errors include *all* matches; rejected cameras are not
    hidden from the comparison.
    """

    source_points = _points(source, label="source")
    target_points = _points(target, label="target")
    if source_points.shape != target_points.shape:
        raise AlignmentError(
            "source and target must contain the same number of 3-D points"
        )
    if not (0.0 < threshold_ratio < 1.0):
        raise AlignmentError("threshold_ratio must be between 0 and 1")
    if not (0.0 < min_inlier_ratio <= 1.0):
        raise AlignmentError("min_inlier_ratio must be in (0, 1]")
    if min_inliers < 3:
        raise AlignmentError("min_inliers must be at least 3")
    if max_trials < 1:
        raise AlignmentError("max_trials must be positive")

    target_mean = np.mean(target_points, axis=0)
    reference_radius = float(
        np.sqrt(np.mean(np.sum((target_points - target_mean) ** 2, axis=1)))
    )
    if not math.isfinite(reference_radius) or reference_radius <= np.finfo(float).eps:
        raise AlignmentError("E10 camera centers have zero usable spatial extent")
    threshold = threshold_ratio * reference_radius

    best_transform: SimilarityTransform | None = None
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    for indices in _sample_indices(source_points.shape[0], max_trials, seed):
        selected = np.asarray(indices, dtype=np.int64)
        try:
            candidate = umeyama_similarity(
                source_points[selected], target_points[selected]
            )
        except AlignmentError:
            continue
        candidate_residuals = np.linalg.norm(
            candidate.apply(source_points) - target_points, axis=1
        )
        mask = candidate_residuals <= threshold
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        inlier_rmse = float(np.sqrt(np.mean(candidate_residuals[mask] ** 2)))
        score = (count, -inlier_rmse)
        if best_score is None or score > best_score:
            best_score = score
            best_transform = candidate
            best_mask = mask

    if best_transform is None or best_mask is None:
        best_transform = umeyama_similarity(source_points, target_points)
        best_residuals = np.linalg.norm(
            best_transform.apply(source_points) - target_points, axis=1
        )
        best_mask = best_residuals <= threshold

    # Re-estimate from the consensus and update membership to convergence.
    for _ in range(4):
        if int(np.count_nonzero(best_mask)) < 3:
            break
        try:
            refined = umeyama_similarity(
                source_points[best_mask], target_points[best_mask]
            )
        except AlignmentError:
            break
        residuals = np.linalg.norm(refined.apply(source_points) - target_points, axis=1)
        updated = residuals <= threshold
        best_transform = refined
        if np.array_equal(updated, best_mask):
            best_mask = updated
            break
        best_mask = updated

    residuals = np.linalg.norm(
        best_transform.apply(source_points) - target_points, axis=1
    )
    if not np.isfinite(residuals).all():
        raise AlignmentError("alignment residuals contain NaN or infinity")
    if int(np.count_nonzero(best_mask)) == 0:
        raise AlignmentError("robust alignment found no inlier correspondences")

    all_metrics = residual_statistics(residuals, reference_radius=reference_radius)
    inlier_metrics = residual_statistics(
        residuals[best_mask], reference_radius=reference_radius
    )
    inlier_count = int(np.count_nonzero(best_mask))
    required_count = max(min_inliers, int(math.ceil(min_inlier_ratio * len(residuals))))
    reasons: list[str] = []
    if len(residuals) < min_inliers:
        reasons.append(
            f"only {len(residuals)} matches; at least {min_inliers} are required"
        )
    if inlier_count < required_count:
        reasons.append(
            f"only {inlier_count}/{len(residuals)} matches satisfy the robust threshold; "
            f"at least {required_count} are required"
        )
    if inlier_count >= 3:
        if _geometry_rank(source_points[best_mask]) < 2:
            reasons.append("VGGSfM inlier camera centers are collinear")
        if _geometry_rank(target_points[best_mask]) < 2:
            reasons.append("E10 inlier camera centers are collinear")
    if float(inlier_metrics["p90_over_e10_camera_rms_radius"]) > threshold_ratio:
        reasons.append("inlier p90 residual exceeds the configured relative threshold")
    if float(all_metrics["median_over_e10_camera_rms_radius"]) > 2.0 * threshold_ratio:
        reasons.append("all-match median residual is too large for safe point-cloud alignment")

    return RobustAlignment(
        transform=best_transform,
        residuals=residuals,
        inlier_mask=best_mask,
        threshold_e10_units=threshold,
        reference_radius_e10_units=reference_radius,
        robust_for_point_cloud=not reasons,
        robustness_reasons=tuple(reasons),
        all_metrics=all_metrics,
        inlier_metrics=inlier_metrics,
    )
