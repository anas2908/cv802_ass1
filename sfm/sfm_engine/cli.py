"""Command-line interface for the CV802 headless sparse-SfM engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import SfMConfig, run_reconstruction, validate_model


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run or reload a fresh-camera SfM experiment")
    run.add_argument("--experiment", required=True)
    run.add_argument("--images", required=True, type=Path)
    run.add_argument("--camera-model", default="SIMPLE_RADIAL")
    run.add_argument("--camera-mode", choices=("auto", "per_folder", "single"), default="auto")
    run.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    run.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    run.add_argument("--max-image-size", type=int, default=3200)
    run.add_argument("--max-num-features", type=int, default=16384)
    run.add_argument("--num-threads", type=int, default=8)
    run.add_argument("--random-seed", type=int, default=0)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    validate = commands.add_parser("validate", help="validate an existing COLMAP model")
    validate.add_argument("--model", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "validate":
        result = validate_model(arguments.model)
    else:
        config = SfMConfig(
            experiment_name=arguments.experiment,
            image_dir=arguments.images,
            camera_model=arguments.camera_model,
            camera_mode=arguments.camera_mode,
            matcher=arguments.matcher,
            device=arguments.device,
            max_image_size=arguments.max_image_size,
            max_num_features=arguments.max_num_features,
            num_threads=arguments.num_threads,
            random_seed=arguments.random_seed,
        )
        result = run_reconstruction(
            config, resume=not arguments.no_resume, dry_run=arguments.dry_run
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
