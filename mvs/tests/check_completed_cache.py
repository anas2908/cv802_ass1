#!/usr/bin/env python3
"""Instrument read-only reload of an already completed real MVS experiment.

Coordinate with the owning controller before invoking this check. It never
submits a job, reconstructs, queries a GPU, or changes the experiment. Its new
verification report belongs outside the experiment under DATA_ROOT/mvs/runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cv802_mvs.config import MVSConfig
from cv802_mvs.io_utils import atomic_json, sha256_file
from cv802_mvs.paths import PathPolicy
from cv802_mvs.runner import MVSRunner


def snapshot(root: Path) -> dict:
    """Record every file's metadata and hash every JSON scientific receipt.

    The runner independently rehashes input provenance and fused PLY. Avoid
    reading all multi-GB depth-map contents twice merely to prove no writes.
    Access times are intentionally excluded because a read can update atime.
    """
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = {
                'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'inode': stat.st_ino, 'mode': stat.st_mode,
                'sha256': sha256_file(path) if path.suffix == '.json' else None,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    policy = PathPolicy.production()
    config = MVSConfig.load(args.config, policy)
    report = policy.require_method(args.report, 'verification report', must_exist=False)
    report.relative_to(policy.method_root / 'runtime')
    if report.suffix != '.json':
        parser.error('verification report must be a new .json file under MVS/runtime')
    executor = Mock(side_effect=AssertionError('cached reload must not execute COLMAP'))
    runner = MVSRunner(config, policy, command_executor=executor)
    state = json.loads((runner.paths.root / 'run_state.json').read_text())
    if state.get('status') != 'complete':
        parser.error('controller has not published a complete run; wait for completion')
    original = json.loads((runner.paths.manifests / 'final_validation.json').read_text())
    before = snapshot(runner.paths.root)
    started = time.monotonic()
    with patch('cv802_mvs.runner.validate_runtime', side_effect=AssertionError('cached reload must not probe CUDA')) as runtime, \
         patch('cv802_mvs.runner.atomic_json', side_effect=AssertionError('cached reload must not rewrite metadata')) as writes:
        result = runner.run(resume=True)
    if result != {**original, 'cache_hit': True}:
        raise AssertionError('cache reload did not preserve the original final report')
    if before != snapshot(runner.paths.root):
        raise AssertionError('experiment files/metadata changed during cache validation')
    executor.assert_not_called()
    runtime.assert_not_called()
    writes.assert_not_called()
    evidence = {
        'complete': True, 'cache_hit': True, 'experiment': config.experiment,
        'config_digest': config.digest, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'elapsed_validation_seconds': time.monotonic() - started,
        'original_elapsed_reconstruction_seconds_preserved': original['elapsed_seconds'],
        'original_completed_at_preserved': original['completed_at'],
        'executor_calls': executor.call_count, 'runtime_probe_calls': runtime.call_count,
        'metadata_write_calls': writes.call_count, 'experiment_files_unchanged': True,
        'verified_file_metadata_count': len(before),
        'verified_json_hash_count': sum(row['sha256'] is not None for row in before.values()),
        'validated_stage_count': len(result['stage_receipts']),
        'fused_point_cloud': result['colored_dense_point_cloud'],
        'snapshot_scope': 'all file sizes, mtimes, inodes and modes; SHA-256 for JSON; runner validates input provenance, stage summaries and full fused PLY',
    }
    atomic_json(report, evidence)
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
