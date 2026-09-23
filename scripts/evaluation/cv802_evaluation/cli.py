"""Command-line interface for the post-hoc reconstruction comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluator import evaluate


DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
DEFAULT_E10 = (
    DATA_ROOT
    / "sfm/historical/transferred/sfm/reconstructions/experiments"
    / "E10_quality_exhaustive_guided_consensus90/light_shirt/colmap/sparse/0"
)
DEFAULT_VGGSFM_RUN = "light-shirt-vggsfm-v1"
DEFAULT_VGGSFM_MODEL = (
    DATA_ROOT / "vggsfm/outputs" / DEFAULT_VGGSFM_RUN / "colmap/sparse/0"
)
DEFAULT_VGGSFM_REQUEST = (
    DATA_ROOT / "vggsfm/experiments" / DEFAULT_VGGSFM_RUN / "request.json"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Align independent official VGGSfM cameras to authoritative E10 "
            "and report camera-center disagreement."
        )
    )
    result.add_argument("--e10-model", type=Path, default=DEFAULT_E10)
    result.add_argument("--vggsfm-model", type=Path, default=DEFAULT_VGGSFM_MODEL)
    result.add_argument("--vggsfm-request", type=Path, default=DEFAULT_VGGSFM_REQUEST)
    result.add_argument(
        "--evaluation-id", default="light-shirt-vggsfm-v1-vs-e10"
    )
    result.add_argument(
        "--aligned-ply",
        choices=("auto", "never", "required"),
        default="auto",
        help="auto writes only after the robustness gate; required fails if withheld",
    )
    result.add_argument("--ransac-threshold-ratio", type=float, default=0.05)
    result.add_argument("--min-inlier-ratio", type=float, default=0.60)
    result.add_argument("--min-inliers", type=int, default=6)
    result.add_argument("--max-trials", type=int, default=2000)
    return result


def main(arguments: list[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    try:
        report_path, report = evaluate(
            e10_model=args.e10_model,
            vggsfm_model=args.vggsfm_model,
            vggsfm_request=args.vggsfm_request,
            evaluation_id=args.evaluation_id,
            aligned_ply=args.aligned_ply,
            threshold_ratio=args.ransac_threshold_ratio,
            min_inlier_ratio=args.min_inlier_ratio,
            min_inliers=args.min_inliers,
            max_trials=args.max_trials,
        )
    except Exception as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2
    summary = {
        "status": report["status"],
        "report": str(report_path),
        "coverage": report["name_matching"]["coverage"],
        "all_camera_errors": report["camera_center_errors"]["all_matched_cameras"],
        "scale_e10_units_per_vggsfm_unit": report["alignment"]["transform"][
            "scale_e10_units_per_vggsfm_unit"
        ],
        "robust_for_point_cloud": report["alignment"]["robust_for_point_cloud"],
        "aligned_point_cloud": report["aligned_point_cloud"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
