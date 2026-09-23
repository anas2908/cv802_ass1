#!/usr/bin/env python3
"""Read-only live Terminal dashboard for E10's durable progress files."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import time

WORK = Path(__file__).resolve().parent
SUBJECTS = {'light_shirt': ('Light shirt', 7750)}
first_seen = {}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    return f'{minutes // 60}h {minutes % 60:02d}m'


def render():
    status = read(WORK / 'status.json')
    lines = ['E10 — HIGH QUALITY · EXHAUSTIVE · GUIDED ON · 90% CLEANUP',
             datetime.now().strftime('Updated %A %H:%M:%S'),
             'Light-shirt experiment only. No dark-shirt run is queued.', '',
             'Stage: ' + status.get('stage', 'Finishing setup and restart checks').replace('_', ' '), '']
    for subject, (label, total) in SUBJECTS.items():
        candidates = list((WORK / 'datasets' / (subject + '_quality') /
                          'colmap/quality_matching_cache').glob('*/progress.json'))
        path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
        progress = read(path) if path else {}
        done = progress.get('completed_pairs', 0)
        rate_key = (subject, progress.get('checkpoint_key'))
        if progress and rate_key not in first_seen:
            first_seen[rate_key] = (time.monotonic(), done)
        observed_at, observed_done = first_seen.get(rate_key, (time.monotonic(), done))
        elapsed = time.monotonic() - observed_at
        recent = done - observed_done
        filled = round(36 * done / total)
        lines += [label, '[' + '#' * filled + '-' * (36 - filled) + ']',
                  f'  Saved matching: {done:,} / {total:,} pairs ({100 * done / total:.1f}%)']
        if progress:
            lines.append(f"  Guided matching workers: {progress.get('matching_threads', '?')} | Mac: 15 CPU cores, 24 GB RAM")
        if done == total:
            lines.append('  Matching finished; reconstruction/cleanup status appears above.')
        elif recent >= 256 and elapsed > 0:
            lines.append('  Rough matching time left: ' + duration((total - done) * elapsed / recent)
                         + ' (changes with image overlap)')
        else:
            lines.append('  Time estimate: waiting for enough measured batches')
        final = WORK.parents[1] / 'reconstructions/experiments/E10_quality_exhaustive_guided_consensus90' / subject
        analysis = read(final / 'analysis.json')
        if analysis:
            lines.append(f"  90% cleaned result: {analysis.get('retained_points', '?'):,} points.")
        lines.append('')
    if status.get('steps'):
        last = status['steps'][-1]
        lines += ['Current subject: ' + last.get('subject', '').replace('_', ' '),
                  'Latest log: ' + last.get('log', ''), '']
        log = Path(last.get('log', ''))
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 1800))
                tail = stream.read().decode('utf-8', errors='replace').splitlines()[-3:]
            lines += ['Recent activity:'] + ['  ' + item[-150:] for item in tail]
    if status.get('error'):
        lines += ['', 'Stopped with an error: ' + status['error']]
    finalization = read(WORK / 'finalization.json')
    if finalization and status.get('complete'):
        lines += ['', 'Final validation and comparison: ' + finalization.get('stage', 'pending').replace('_', ' ')]
        if finalization.get('error'):
            lines.append('  ' + finalization['error'])
    active = read(WORK / 'active_process.json')
    if active.get('pid'):
        result = subprocess.run(['ps', '-p', str(active['pid']), '-o', 'rss=,pcpu=,command='],
                                text=True, capture_output=True)
        fields = result.stdout.strip().split(maxsplit=2)
        if len(fields) == 3 and 'run_quality.py' in fields[2] and 'light_shirt_quality' in fields[2]:
            lines += ['', f'Current SfM process: {int(fields[0]) * 1024 / 1e9:.2f} GB resident RAM; '
                      f'{float(fields[1]):.0f}% CPU (100% = one core).',
                      'Memory includes matching workers, loaded features and caches.']
    lines += ['', f"Counts update after each saved batch of {progress.get('matching_batch_size', 128)} pairs.",
              'Closing this progress window does not stop reconstruction. Ctrl+C closes the display.']
    print('\033[2J\033[H' + '\n'.join(lines), end='\n', flush=True)


try:
    while True:
        render()
        time.sleep(3)
except KeyboardInterrupt:
    print('\nProgress display closed. Reconstruction is unaffected.')
