#!/usr/bin/env python3
"""Prove a completed VGGSfM reload is read-only and never invokes inference.

This is an acceptance check on a real saved run, not a reconstruction command.
The receipt is written outside the immutable experiment/output directories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.engine import VGGSfMEngine
from vggsfm_engine.io_utils import validate_identifier
from vggsfm_engine.paths import production_layout
from vggsfm_engine.profile import InferenceProfile


def snapshot(roots: list[Path]) -> dict[str, dict]:
    records = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            records[str(path)] = {
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": digest.hexdigest(),
            }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_identifier(args.run_id, label="run ID")
    validate_identifier(args.dataset, label="dataset")
    layout = production_layout()
    output = args.output.resolve()
    layout.assert_member(output, label="cache validation receipt")
    if output.exists():
        parser.error("receipt already exists; choose a new path")
    experiment = layout.experiment_root(args.run_id)
    published = layout.output_root(args.run_id)
    for immutable in (experiment, published):
        if output.is_relative_to(immutable.resolve()):
            parser.error("receipt must be outside the saved experiment and output")
    manifest = json.loads((published / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        parser.error("only a validated complete run can be checked")
    engine = VGGSfMEngine(layout)
    profile = InferenceProfile.from_json(args.profile)
    before = snapshot([experiment, published])
    started = time.monotonic()
    with (
        patch.object(engine.executor, "run", side_effect=AssertionError("Unexpected inference")) as execute,
        patch.object(engine, "verify_runtime", side_effect=AssertionError("Unexpected runtime/CUDA probe")) as runtime,
        patch("vggsfm_engine.engine.atomic_write_json", side_effect=AssertionError("Unexpected metadata write")) as write,
    ):
        result = engine.run(args.dataset, args.run_id, profile)
    elapsed = time.monotonic() - started
    if result.status != "cached":
        raise AssertionError(f"Expected cached result, received {result.status}")
    after = snapshot([experiment, published])
    if before != after:
        raise AssertionError("Saved experiment/output changed during cache validation")
    receipt = {
        **result.as_dict(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "executor_calls": execute.call_count,
        "runtime_probe_calls": runtime.call_count,
        "metadata_write_calls": write.call_count,
        "experiment_files_unchanged": True,
        "files_unchanged": len(before), "artifacts": before,
        "cache_check_elapsed_seconds": elapsed,
        "original_inference_elapsed_seconds": manifest["official_elapsed_seconds"],
        "scope": "Hashes, sizes and mtimes of every experiment/output file checked before and after a reload; executor and writes fail closed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "artifacts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
