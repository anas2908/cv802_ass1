#!/usr/bin/env python3
"""Run one prepared E8 reconstruction, unchanged rectangle preview and audits."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=['light_shirt', 'black_shirt_crutches'])
    args = parser.parse_args()
    subject = args.subject
    dataset = ROOT / 'reconstructions/experiments/E8_vocab_guided' / (subject + '_quality')
    baseline = ROOT / 'reconstructions' / subject
    work = ROOT / 'work/matching-ablation/E8' / subject
    work.mkdir(parents=True, exist_ok=True)
    attempt = work / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt.mkdir(parents=True)
    status_file = attempt / 'status.json'
    if not (dataset / 'sfm_refine.json').is_file():
        raise FileNotFoundError('Run prepare_e8.py first')
    config = json.loads((dataset / 'sfm_refine.json').read_text())
    py = sys.executable
    commands = []
    if not (dataset / 'reconstruction_report.json').is_file():
        commands.append(
            ('reconstruction', [py, '-u', str(ROOT / 'work/quality-sfm/run_quality.py'),
                                'experiments/E8_vocab_guided/' + subject + '_quality']))
    if not (dataset / 'subject_preview/analysis.json').is_file():
        commands.append(
            ('rectangle_preview', [py, str(ROOT / 'work/full-capture/make_subject_preview.py'),
                                   str(dataset), '--boxes', str((dataset / config['boxes']).resolve()),
                                   '--volume-fraction', '.8' if subject == 'light_shirt' else '.9']))
    commands += [
        ('model_audit', [py, str(ROOT / 'work/quality-sfm/audit_model.py'), str(baseline), str(dataset),
                         '--output', str(attempt / 'model_audit.json')]),
        ('cache_verification', [py, str(ROOT / 'work/quality-sfm/verify_cached_results.py'),
                                str(dataset), str(dataset / 'subject_preview'),
                                '--output', str(attempt / 'cache_verification.json')]),
    ]
    status = {'experiment': 'E8', 'subject': subject, 'dataset': str(dataset),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'steps': []}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for stage, command in commands:
        record = {'stage': stage, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
        status.update(stage=stage, complete=False)
        status['steps'].append(record)
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        with (attempt / (stage + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {attempt}')
        print(subject, stage, 'complete', flush=True)
    status.update(stage='complete', complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    status_file.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
