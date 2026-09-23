"""End-to-end VGGSfM-to-E10 post-hoc camera-center evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

from .alignment import RobustAlignment, robust_similarity_alignment
from .colmap_source import load_binary_camera_poses, point_reader
from .io_utils import (
    DATA_ROOT,
    EVALUATION_ROOT,
    atomic_write_json,
    evaluation_directory,
    file_record,
    require_within,
    sha256,
)
from .matching import load_official_name_map, match_camera_centers


class EvaluationError(RuntimeError):
    """The requested cross-method evaluation could not be completed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _model_records(model: Path) -> list[dict[str, Any]]:
    return [
        file_record(model / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    ]


def _atomic_aligned_ply(
    source_model: Path,
    destination: Path,
    alignment: RobustAlignment,
) -> dict[str, Any]:
    """Transform every VGGSfM sparse point while preserving its RGB values."""

    destination_parent = require_within(
        destination.parent, EVALUATION_ROOT, label="aligned PLY directory"
    )
    count, points = point_reader(source_model)
    if count == 0:
        raise EvaluationError("VGGSfM model contains no 3-D points to align")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination_parent
    )
    temporary = Path(temporary_name)
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    float32_limit = float(np.finfo(np.float32).max)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "comment VGGSfM sparse points aligned post-hoc to COLMAP E10\n"
                "comment coordinates remain arbitrary E10 reconstruction units, not metres\n"
                f"element vertex {count}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            stream.write(header)
            written = 0
            for point in points:
                source = np.asarray([[point.x, point.y, point.z]], dtype=np.float64)
                transformed = alignment.transform.apply(source)[0]
                if not np.isfinite(transformed).all() or np.any(
                    np.abs(transformed) > float32_limit
                ):
                    raise EvaluationError(
                        f"aligned point {point.point_id} cannot be represented safely"
                    )
                stream.write(
                    struct.pack(
                        "<fffBBB",
                        float(transformed[0]),
                        float(transformed[1]),
                        float(transformed[2]),
                        int(point.red),
                        int(point.green),
                        int(point.blue),
                    )
                )
                bounds_min = np.minimum(bounds_min, transformed)
                bounds_max = np.maximum(bounds_max, transformed)
                written += 1
            if written != count:
                raise EvaluationError(
                    f"COLMAP declared {count} points but the stream yielded {written}"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination_parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "written",
        "path": str(destination),
        "format": "binary_little_endian",
        "has_rgb": True,
        "point_count": count,
        "bounds_min_e10_units": bounds_min.tolist(),
        "bounds_max_e10_units": bounds_max.tolist(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def evaluate(
    *,
    e10_model: Path,
    vggsfm_model: Path,
    vggsfm_request: Path,
    evaluation_id: str,
    aligned_ply: str = "auto",
    threshold_ratio: float = 0.05,
    min_inlier_ratio: float = 0.60,
    min_inliers: int = 6,
    max_trials: int = 2000,
) -> tuple[Path, dict[str, Any]]:
    """Run the comparison and atomically publish ``report.json``."""

    if aligned_ply not in {"auto", "never", "required"}:
        raise EvaluationError("aligned_ply must be auto, never or required")
    e10_model = require_within(e10_model, DATA_ROOT, label="E10 model")
    vggsfm_model = require_within(vggsfm_model, DATA_ROOT, label="VGGSfM model")
    vggsfm_request = require_within(
        vggsfm_request, DATA_ROOT, label="VGGSfM request"
    )
    output = evaluation_directory(evaluation_id)

    e10_poses = load_binary_camera_poses(e10_model)
    vggsfm_poses = load_binary_camera_poses(vggsfm_model)
    official_map, request_payload = load_official_name_map(vggsfm_request)
    independence = request_payload.get("independence")
    if not isinstance(independence, dict):
        raise EvaluationError(
            "VGGSfM request lacks the independence receipt required for post-hoc evaluation"
        )
    if independence.get("sfm_e10_poses_used") is not False:
        raise EvaluationError(
            "VGGSfM request does not prove that E10 poses were excluded from inference"
        )
    if independence.get("alignment_allowed_only_after_reconstruction") is not True:
        raise EvaluationError(
            "VGGSfM request does not authorize alignment only after reconstruction"
        )
    matches = match_camera_centers(e10_poses, vggsfm_poses, official_map)
    alignment = robust_similarity_alignment(
        matches.vggsfm_centers,
        matches.e10_centers,
        threshold_ratio=threshold_ratio,
        min_inlier_ratio=min_inlier_ratio,
        min_inliers=min_inliers,
        max_trials=max_trials,
    )

    matched_records: list[dict[str, Any]] = []
    for index, names in enumerate(matches.names):
        matched_records.append(
            {
                "vggsfm_official_name": names[0],
                "original_e10_name": names[1],
                "residual_e10_units": float(alignment.residuals[index]),
                "ransac_inlier": bool(alignment.inlier_mask[index]),
            }
        )

    point_cloud: dict[str, Any]
    if aligned_ply == "never":
        point_cloud = {
            "status": "not_requested",
            "reason": "--aligned-ply=never",
        }
    elif alignment.robust_for_point_cloud:
        point_cloud = _atomic_aligned_ply(
            vggsfm_model,
            output / "aligned_vggsfm_points.ply",
            alignment,
        )
    else:
        point_cloud = {
            "status": "withheld",
            "reason": "camera-center alignment did not pass the robustness gate",
            "robustness_reasons": list(alignment.robustness_reasons),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "created_at": _utc_now(),
        "evaluation_id": evaluation_id,
        "comparison": "independent official VGGSfM v2 aligned post-hoc to COLMAP E10 light",
        "inputs": {
            "e10_model": _model_records(e10_model),
            "vggsfm_model": _model_records(vggsfm_model),
            "vggsfm_request": file_record(vggsfm_request),
            "vggsfm_request_fingerprint": request_payload.get("request_fingerprint"),
            "vggsfm_independence_receipt": independence,
        },
        "name_matching": {
            "policy": "exact official-to-original mapping from immutable VGGSfM request; no basename guessing",
            "coverage": matches.coverage(),
            "vggsfm_originals_missing_from_e10": list(
                matches.vggsfm_originals_missing_from_e10
            ),
            "e10_images_missing_from_vggsfm": list(
                matches.e10_images_missing_from_vggsfm
            ),
        },
        "alignment": {
            "estimator": "deterministic RANSAC consensus followed by proper Umeyama least-squares refit",
            "transform": alignment.transform.as_dict(),
            "threshold_ratio_of_e10_camera_rms_radius": threshold_ratio,
            "threshold_e10_units": alignment.threshold_e10_units,
            "e10_matched_camera_rms_radius": alignment.reference_radius_e10_units,
            "minimum_inlier_ratio": min_inlier_ratio,
            "minimum_inlier_count": min_inliers,
            "inlier_count": int(np.count_nonzero(alignment.inlier_mask)),
            "outlier_count": int(len(alignment.inlier_mask) - np.count_nonzero(alignment.inlier_mask)),
            "robust_for_point_cloud": alignment.robust_for_point_cloud,
            "robustness_reasons": list(alignment.robustness_reasons),
        },
        "camera_center_errors": {
            "all_matched_cameras": alignment.all_metrics,
            "ransac_inliers": alignment.inlier_metrics,
        },
        "matched_cameras": matched_records,
        "aligned_point_cloud": point_cloud,
        "interpretation": {
            "units": "Distances are in E10's arbitrary monocular-SfM reconstruction units, not metres.",
            "scale": "Scale is E10-units per VGGSfM-unit; neither reconstruction has known metric scale.",
            "reference_not_ground_truth": "E10 is the authoritative assignment reference, not physical ground truth; residuals measure cross-method camera-center disagreement.",
            "post_hoc_only": "The transform was estimated only after independent VGGSfM inference and was not supplied to VGGSfM.",
            "points_not_corresponded": "Sparse point counts/geometries are not scored point-to-point because the methods reconstruct different latent points.",
        },
    }
    report_path = output / "report.json"
    atomic_write_json(report_path, report)
    if aligned_ply == "required" and not alignment.robust_for_point_cloud:
        raise EvaluationError(
            f"alignment report was written to {report_path}, but required PLY was withheld: "
            + "; ".join(alignment.robustness_reasons)
        )
    return report_path, report
