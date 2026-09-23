"""Command-line interface for staging, planning, running, and auditing MVS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import VERSION
from .commands import build_commands
from .config import MVSConfig
from .errors import MVSError
from .paths import PathPolicy
from .runner import MVSRunner
from .staging import stage_inputs
from .validation import validate_colored_ply, validate_inputs, validate_runtime


def _exclusive_lifecycle(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="Verify and reuse completed work")
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="Move prior artifacts into the experiment archive before starting",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="cv802-mvs",
        description="Headless, checksum-audited COLMAP CUDA dense MVS",
    )
    result.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subcommands = result.add_subparsers(dest="command", required=True)

    stage = subcommands.add_parser(
        "stage-inputs", help="Copy calibrated model/images/masks into independent MVS storage"
    )
    stage.add_argument("--experiment", required=True)
    stage.add_argument("--images-source", type=Path, required=True)
    stage.add_argument("--model-source", type=Path, required=True)
    stage.add_argument("--masks-source", type=Path)
    stage.add_argument("--mask-manifest-source", type=Path)
    _exclusive_lifecycle(stage)

    plan = subcommands.add_parser("plan", help="Validate inputs and print exact commands; write nothing")
    plan.add_argument("--config", type=Path, required=True)

    validate = subcommands.add_parser("validate", help="Validate staged inputs or a completed PLY")
    validate_group = validate.add_mutually_exclusive_group(required=True)
    validate_group.add_argument("--config", type=Path)
    validate_group.add_argument("--ply", type=Path)
    validate.add_argument(
        "--runtime",
        action="store_true",
        help="Also require active Slurm/NVIDIA/COLMAP runtime (creates data-root cache dirs)",
    )

    run = subcommands.add_parser("run", help="Execute undistort, PatchMatch, fusion, and optional mesh")
    run.add_argument("--config", type=Path, required=True)
    _exclusive_lifecycle(run)

    status = subcommands.add_parser("status", help="Print state and stage receipts")
    status.add_argument("--config", type=Path, required=True)
    return result


def _load_config(path: Path, policy: PathPolicy) -> MVSConfig:
    return MVSConfig.load(path.resolve(strict=True), policy)


def _status(config: MVSConfig, policy: PathPolicy) -> dict[str, Any]:
    paths = config.paths(policy)
    state_path = paths.root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    receipts: list[dict[str, Any]] = []
    if paths.receipts.exists():
        for receipt_path in sorted(paths.receipts.glob("*.json")):
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                receipt = {"status": "malformed", "error": str(error)}
            receipts.append({"path": str(receipt_path), **receipt})
    return {
        "experiment": config.experiment,
        "experiment_root": str(paths.root),
        "state": state,
        "receipts": receipts,
        "live_logs": [str(path) for path in sorted(paths.logs.glob("*.log"))]
        if paths.logs.exists()
        else [],
    }


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    policy = PathPolicy.production()
    try:
        if arguments.command == "stage-inputs":
            result: Any = stage_inputs(
                policy=policy,
                experiment=arguments.experiment,
                images_source=arguments.images_source,
                model_source=arguments.model_source,
                masks_source=arguments.masks_source,
                mask_manifest_source=arguments.mask_manifest_source,
                resume=arguments.resume,
                overwrite=arguments.overwrite,
            )
        elif arguments.command == "plan":
            config = _load_config(arguments.config, policy)
            result = MVSRunner(config, policy).dry_run()
        elif arguments.command == "validate":
            if arguments.ply:
                ply = policy.require_method(arguments.ply, "PLY output", must_exist=True)
                result = validate_colored_ply(ply)
            else:
                config = _load_config(arguments.config, policy)
                paths = config.paths(policy)
                result = {"inputs": validate_inputs(config, paths)}
                if arguments.runtime:
                    runtime, _ = validate_runtime(
                        config, paths, policy, build_commands(config, paths)
                    )
                    result["runtime"] = runtime
        elif arguments.command == "run":
            config = _load_config(arguments.config, policy)
            result = MVSRunner(config, policy).run(
                resume=arguments.resume, overwrite=arguments.overwrite
            )
        elif arguments.command == "status":
            config = _load_config(arguments.config, policy)
            result = _status(config, policy)
        else:  # pragma: no cover - argparse enforces a known subcommand.
            raise AssertionError(arguments.command)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (MVSError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

