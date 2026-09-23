#!/usr/bin/env python3
"""Prove that a completed fresh-SfM request reloads without computing again.

This is a real integration check, not a substitute for reconstruction tests.
It reconstructs the original immutable request from its saved receipt, mocks
only the expensive PyCOLMAP entry points to fail if called, and compares every
published artifact before/after. The verification receipt is separate from the
experiment so the historical experiment is not rewritten by a cache check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfm_engine.pipeline import SfMConfig, run_reconstruction


DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1/sfm")


def snapshot(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".run.lock":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        result[str(path.relative_to(root))] = {
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = (DATA_ROOT / "experiments" / args.experiment).resolve(strict=True)
    root.relative_to((DATA_ROOT / "experiments").resolve(strict=True))
    output = args.output.resolve()
    output.relative_to(DATA_ROOT.resolve(strict=True))
    if output.is_relative_to(root):
        raise ValueError("cache-check receipt must be outside the experiment")
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((root / "receipt.json").read_text())
    config_data = receipt["config"]
    config = SfMConfig(**{**config_data, "image_dir": Path(config_data["image_dir"])})
    before = snapshot(root)
    started = time.monotonic()
    with patch("pycolmap.extract_features", side_effect=AssertionError("cache re-extracted")) as extract, \
         patch("pycolmap.match_exhaustive", side_effect=AssertionError("cache rematched")) as match, \
         patch("pycolmap.match_sequential", side_effect=AssertionError("cache rematched")) as sequential, \
         patch("pycolmap.incremental_mapping", side_effect=AssertionError("cache remapped")) as mapping:
        result = run_reconstruction(config)
        calls = {
            "extract_features": extract.call_count, "match_exhaustive": match.call_count,
            "match_sequential": sequential.call_count, "incremental_mapping": mapping.call_count,
        }
    after = snapshot(root)
    if not result.get("cache_hit") or before != after or any(calls.values()):
        raise AssertionError("completed reconstruction failed immutable cache check")
    record = {
        "status": "complete", "experiment": str(root), "cache_hit": True,
        "reconstruction_calls": calls, "files_unchanged": len(before),
        "verified_fields": ["bytes", "mtime_ns", "sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "metrics": result["metrics"], "artifacts": before,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "complete", "receipt": str(output), "files_unchanged": len(before)}))


if __name__ == "__main__":
    main()
