"""Command-line interface shared by batch jobs and the future UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import VGGSfMEngine
from .errors import VGGSfMError
from .paths import production_layout
from .profile import InferenceProfile

CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = CODE_ROOT / "configs" / "light_shirt.json"


def _common_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name below DATA_ROOT/vggsfm/inputs (for example light_shirt)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Immutable experiment/output identifier (letters, numbers, dot, dash, underscore)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"Strict JSON inference profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument("--query-frames", type=int, default=None)
    parser.add_argument("--max-query-points", type=int, default=None)
    parser.add_argument(
        "--shared-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use one shared intrinsic camera model for all input frames",
    )
    parser.add_argument(
        "--fine-tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the official fine tracker (more accurate, slower)",
    )
    parser.add_argument(
        "--extra-point-interval",
        type=int,
        default=None,
        help="Pixel sampling interval for official additional colored points",
    )
    parser.add_argument(
        "--extra-point-neighbors",
        type=int,
        default=None,
        help=(
            "Number of temporally adjacent frames used to triangulate each "
            "additional-point grid (-1 uses every frame)"
        ),
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Disable additional-point triangulation and export only BA reconstruction points",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_vggsfm.py",
        description="Headless, reproducible adapter for the pinned official VGGSfM v2 code",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check storage, environment and source revision")
    doctor.add_argument("--require-ready", action="store_true")

    plan = subparsers.add_parser("plan", help="Validate and print a no-write execution plan")
    _common_request_arguments(plan)

    run = subparsers.add_parser("run", help="Run or checksum-validate a reconstruction")
    _common_request_arguments(run)
    run.add_argument(
        "--resume",
        action="store_true",
        help="After a partial failure, create a new attempt without overwriting the old one",
    )
    return parser


def _load_profile(args: argparse.Namespace) -> InferenceProfile:
    profile = InferenceProfile.from_json(args.profile)
    overrides: dict[str, Any] = {
        "query_frame_num": args.query_frames,
        "max_query_pts": args.max_query_points,
        "shared_camera": args.shared_camera,
        "fine_tracking": args.fine_tracking,
        "extra_point_pixel_interval": args.extra_point_interval,
        "extra_point_neighbor_frames": args.extra_point_neighbors,
    }
    if args.sparse_only:
        overrides["extra_point_pixel_interval"] = -1
        overrides["concatenate_extra_points"] = False
    return profile.with_overrides(**overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        layout = production_layout()
        engine = VGGSfMEngine(layout)
        if args.command == "doctor":
            readiness = engine.runtime_readiness()
            print(json.dumps(readiness, indent=2, sort_keys=True))
            ready = all(
                readiness[key]
                for key in (
                    "method_data_root_exists",
                    "method_data_root_writable",
                    "environment_python_exists",
                    "official_demo_exists",
                    "official_revision_matches",
                )
            )
            return 0 if ready or not args.require_ready else 1

        profile = _load_profile(args)
        if args.command == "plan":
            plan = engine.plan(args.dataset, args.run_id, profile)
            print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            result = engine.run(
                args.dataset,
                args.run_id,
                profile,
                resume=args.resume,
            )
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            return 0
        parser.error(f"unknown command {args.command}")
    except VGGSfMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
