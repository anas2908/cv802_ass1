#!/usr/bin/env python3
"""Run E10 with restartable matching; publish only consensus90 subject maps.

Dark crutch candidates require inspection of this run's six-photo overlay before
--finish-dark-cleanup. Source photographs, E1–E9 results and masks are read-only.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
EXPERIMENTS = ROOT / 'reconstructions/experiments'
OUTPUT = EXPERIMENTS / 'E10_quality_exhaustive_guided_consensus90'
SUBJECTS = ('light_shirt', 'black_shirt_crutches')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def cleanup_command(subject, qc_only=False):
    common = ['--source', str(WORK / 'datasets' / (subject + '_quality') / 'subject_preview'),
              '--output-parent', str(OUTPUT), '--variants', subject + ':0.90',
              '--foreground-threshold', '0.5', '--tolerance-baseline-pixels', '3',
              '--reference-long-edge', '3200', '--min-positive-view-fraction', '0.2',
              '--min-track-views', '3']
    if subject == 'light_shirt':
        original = EXPERIMENTS / 'light_person_mask_cleanup'
        return [sys.executable, '-u', str(original / 'filter_person_masks.py'),
                '--manifest', str(original / 'mask_manifest.json'), *common]
    original = EXPERIMENTS / 'black_person_crutches_cleanup'
    command = [sys.executable, '-u', str(original / 'filter_person_crutches.py'),
               '--manifest', str(original / 'body_mask_manifest.json'),
               '--protection', str(original / 'crutch_protection/protection_approved.json'),
               '--candidate-review', str(WORK / 'crutch_review'), *common]
    return command + (['--qc-only'] if qc_only else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', choices=(*SUBJECTS, 'both'), default='light_shirt')
    parser.add_argument('--finish-dark-cleanup', action='store_true')
    parser.add_argument('--wait-for-pid', type=int, help='Wait for an already running reconstruction before resuming its cached result')
    args = parser.parse_args()
    with (WORK / 'runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = WORK / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        run.mkdir(parents=True)
        previous = json.loads((WORK / 'status.json').read_text()) if (WORK / 'status.json').is_file() else {}
        record = {'experiment': 'E10', 'started_utc': now(), 'pid': os.getpid(),
                  'subjects': list(SUBJECTS) if args.subject == 'both' else [args.subject],
                  'final_policy': 'Only consensus90 cleaned outputs for the requested subjects.',
                  'run_directory': str(run), 'steps': [], 'complete': False}
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

        def persist():
            save(run / 'status.json', record)
            save(WORK / 'status.json', record)

        def step(subject, label, command):
            entry = {'subject': subject, 'stage': label, 'command': command,
                     'started_utc': now(), 'log': str(run / (subject + '_' + label + '.log'))}
            record['steps'].append(entry)
            record.update(stage=label, subject=subject)
            persist()
            print(f'{now()} {subject}: {label}', flush=True)
            with Path(entry['log']).open('x') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
            entry.update(finished_utc=now(), returncode=result.returncode)
            persist()
            if result.returncode:
                raise RuntimeError(f'{subject} {label} failed; see {entry["log"]}')

        def verify(subject, paths, label):
            step(subject, label, [sys.executable, str(ROOT / 'work/quality-sfm/verify_cached_results.py'),
                                 *map(str, paths), '--output', str(run / (subject + '_' + label + '.json'))])

        persist()
        # Keep this Mac awake only for the lifetime of this runner.
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        try:
            if args.wait_for_pid:
                record.update(stage='reconstruction', waiting_for_existing_reconstruction_pid=args.wait_for_pid,
                              preceding_run=previous.get('run_directory'),
                              scope_change='User requested light-shirt E10 only; dark run removed before it started.')
                if previous.get('steps'):
                    record['steps'].append(dict(previous['steps'][-1], inherited_running_process=True))
                persist()
                print(f'Continuing existing reconstruction PID {args.wait_for_pid}; dark E10 is not queued.', flush=True)
                while True:
                    try:
                        os.kill(args.wait_for_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(3)
                record.pop('waiting_for_existing_reconstruction_pid', None)
                persist()
            if args.finish_dark_cleanup:
                subject = 'black_shirt_crutches'
                if not (OUTPUT / subject / 'analysis.json').is_file():
                    step(subject, 'cleanup90', cleanup_command(subject))
                verify(subject, [OUTPUT / subject], 'cleaned_cache_verification')
                record.update(stage='dark_cleanup_complete', complete=True)
            else:
                for subject in record['subjects']:
                    dataset = WORK / 'datasets' / (subject + '_quality')
                    config = json.loads((dataset / 'sfm_refine.json').read_text())
                    if config.get('resume_matching') is not True:
                        raise ValueError('E10 must use the reviewed restartable exhaustive guided path')
                    runner_name = os.path.relpath(dataset, ROOT / 'reconstructions')
                    step(subject, 'reconstruction', [sys.executable, '-u',
                         str(ROOT / 'work/quality-sfm/run_quality.py'), runner_name])
                    preview = dataset / 'subject_preview'
                    if not (preview / 'analysis.json').is_file():
                        step(subject, 'rectangle_intermediate', [sys.executable,
                             str(ROOT / 'work/full-capture/make_subject_preview.py'), str(dataset),
                             '--boxes', str((dataset / config['boxes']).resolve()),
                             '--volume-fraction', '.8' if subject == 'light_shirt' else '.9'])
                    step(subject, 'model_audit', [sys.executable,
                         str(ROOT / 'work/quality-sfm/audit_model.py'),
                         str(ROOT / 'reconstructions' / subject), str(dataset),
                         '--output', str(run / (subject + '_model_audit.json'))])
                    verify(subject, [dataset, preview], 'internal_cache_verification')
                    if subject == 'light_shirt':
                        if not (OUTPUT / subject / 'analysis.json').is_file():
                            step(subject, 'cleanup90', cleanup_command(subject))
                        verify(subject, [OUTPUT / subject], 'cleaned_cache_verification')
                    else:
                        if not (WORK / 'crutch_review/review.json').is_file():
                            step(subject, 'crutch_candidate_qc', cleanup_command(subject, qc_only=True))
                        record['dark_crutch_review_required'] = not (OUTPUT / subject / 'analysis.json').is_file()
                record.update(stage='awaiting_crutch_overlay_review' if record.get('dark_crutch_review_required')
                              else 'requested_subjects_complete',
                              complete=not record.get('dark_crutch_review_required', False))
            record['finished_utc'] = now()
            persist()
            print(json.dumps(record, indent=2), flush=True)
        except BaseException as error:
            record.update(stage='failed', error=str(error), finished_utc=now())
            persist()
            raise
        finally:
            awake.terminate()
            awake.wait()


if __name__ == '__main__':
    main()
