"""Optional integration check against a completed real fixed-pose experiment.

Run from the SfM code root using ``python tests/check_fixed_pose_cache.py ...``.
The success report belongs in DATA_ROOT/sfm, outside the checked experiment.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfm_engine.fixed_pose import FixedPoseConfig, run
from sfm_engine.pipeline import SFM_DATA_ROOT, atomic_json, require_under, sha256_file


def snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
            for path in sorted(root.rglob('*')) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    for name in ('image_dir', 'baseline_model', 'database', 'feature_database', 'feature_metadata'):
        path = Path(values[name])
        values[name] = path if path.is_absolute() else SFM_DATA_ROOT / path
    config = FixedPoseConfig(**values)
    experiment = SFM_DATA_ROOT / 'experiments' / config.experiment_name
    report = require_under(args.report, SFM_DATA_ROOT, 'cache check report')
    if report.parent != SFM_DATA_ROOT.resolve() or report.suffix != '.json' or report.exists():
        raise ValueError('choose a new .json report directly in DATA_ROOT/sfm')
    before = snapshot(experiment)
    if not before or not (experiment / 'receipt.json').exists():
        raise ValueError('a completed real experiment is required')
    with patch('pycolmap.triangulate_points', side_effect=AssertionError('cached load must not triangulate')) as triangulate:
        result = run(config)
        if not result.get('cached_reuse') or triangulate.call_count:
            raise AssertionError('cache was not reused')
    if snapshot(experiment) != before:
        raise AssertionError('cached validation changed experiment files or created a new attempt')
    payload = {'complete': True, 'experiment': str(experiment), 'cached_reuse': True,
               'triangulation_calls': 0, 'experiment_files_unchanged': True,
               'verified_files': len(before), 'identity_sha256': result['identity_sha256']}
    atomic_json(report, payload)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
