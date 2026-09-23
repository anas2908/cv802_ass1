#!/usr/bin/env python3
"""Apply the unchanged E4 consensus-90 cleanup to E6 or E8 in separate folders."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / 'reconstructions/experiments'
NAMES = {'E7': ('E6_guided_off', 'E7_guided_off_consensus90'),
         'E9': ('E8_vocab_guided', 'E9_vocab_guided_consensus90')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=NAMES)
    parser.add_argument('subject', choices=['light_shirt', 'black_shirt_crutches'])
    parser.add_argument('--qc-only', action='store_true', help='Create dark candidate crutch projections for inspection')
    args = parser.parse_args()
    if args.qc_only and args.subject != 'black_shirt_crutches':
        parser.error('--qc-only is for crutch protection')
    source_name, target_name = NAMES[args.experiment]
    source = EXPERIMENTS / source_name / (args.subject + '_quality') / 'subject_preview'
    output = EXPERIMENTS / target_name
    work = ROOT / 'work/matching-ablation' / args.experiment
    if not (source / 'analysis.json').is_file():
        raise FileNotFoundError(f'Source preview is not ready: {source}')
    work.mkdir(parents=True, exist_ok=True)
    common = ['--source', str(source), '--output-parent', str(output),
              '--variants', args.subject + ':0.90', '--foreground-threshold', '0.5',
              '--tolerance-baseline-pixels', '3', '--reference-long-edge', '3200',
              '--min-positive-view-fraction', '0.2', '--min-track-views', '3']
    if args.subject == 'light_shirt':
        original = EXPERIMENTS / 'light_person_mask_cleanup'
        command = [sys.executable, '-u', str(original / 'filter_person_masks.py'),
                   '--manifest', str(original / 'mask_manifest.json'), *common]
    else:
        original = EXPERIMENTS / 'black_person_crutches_cleanup'
        command = [sys.executable, '-u', str(original / 'filter_person_crutches.py'),
                   '--manifest', str(original / 'body_mask_manifest.json'),
                   '--protection', str(original / 'crutch_protection/protection_approved.json'),
                   '--candidate-review', str(work / 'crutch_review'), *common]
        if args.qc_only:
            command.append('--qc-only')
    stem = args.subject + ('_crutch_qc' if args.qc_only else '_cleanup')
    attempt = work / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt.mkdir(parents=True)
    log_path = attempt / (stem + '.log')
    record = {'experiment': args.experiment, 'subject': args.subject,
              'source': str(source), 'output': str(output / args.subject),
              'same_frozen_masks_and_parameters_as_E4': True,
              'qc_only': args.qc_only, 'command': command,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    record_path = attempt / (stem + '.json')
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with log_path.open('x') as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=completed.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    if completed.returncode:
        raise RuntimeError(f'Cleanup failed; see {log_path}')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
