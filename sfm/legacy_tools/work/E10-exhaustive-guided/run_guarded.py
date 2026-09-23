#!/usr/bin/env python3
"""Run light-only E10 and fall back from four workers on sustained RAM pressure."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CONFIG = WORK / 'datasets/light_shirt_quality/sfm_refine.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def quality_process(parent):
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss=,pcpu=,command='], text=True)
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=4)
        if (len(fields) == 5 and fields[0].isdigit() and fields[1].isdigit()
                and int(fields[1]) == parent and 'run_quality.py' in fields[4] and 'light_shirt_quality' in fields[4]):
            return {'pid': int(fields[0]), 'resident_bytes': int(fields[2]) * 1024,
                    'cpu_percent': float(fields[3])}
    return None


def free_percent():
    result = subprocess.run(['memory_pressure'], text=True, capture_output=True)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', result.stdout)
    return int(match.group(1)) if match else None


def stop(runner, child):
    # Stop the wrapper first so it cannot start or restart another stage.
    if runner.poll() is None:
        runner.terminate()
    if child:
        try:
            os.kill(child, signal.SIGTERM)
        except ProcessLookupError:
            pass
    runner.wait(timeout=15)
    if child:
        for _ in range(100):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Matching process did not stop; checkpoint migration was not attempted')


def main():
    with (WORK / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'pid': os.getpid(), 'started_utc': now(), 'scope': 'light_shirt only',
                 'stage': 'starting', 'attempts': [], 'complete': False,
                 'memory_policy': 'Fallback if RSS >20 GB or free memory <5% immediately; or 3 samples of free<12% / RSS>18 GB plus free<20%. Samples every5seconds.'}
        runner = None
        child_pid = None
        try:
            while True:
                config = read(CONFIG)
                workers = config['matching_threads']
                attempt = {'started_utc': now(), 'workers': workers,
                           'matching_batch_size': config['matching_batch_size'], 'sampled_peak_resident_bytes': 0}
                state['attempts'].append(attempt)
                state.update(stage='running', current_workers=workers)
                runner = subprocess.Popen([sys.executable, '-u', str(WORK / 'run_e10.py'),
                                           '--subject', 'light_shirt'], cwd=ROOT)
                attempt['runner_pid'] = runner.pid
                save(WORK / 'supervisor_status.json', state)
                print(f'{now()} E10 supervisor: {workers} matching workers; runner {runner.pid}', flush=True)
                pressure_streak = 0
                fallback = False
                samples = 0
                while runner.poll() is None:
                    process = quality_process(runner.pid)
                    if process:
                        child_pid = process['pid']
                        process.update(time_utc=now(), matching_workers=workers,
                                       system_free_memory_percent=free_percent())
                        save(WORK / 'active_process.json', dict(pid=child_pid, matching_threads=workers,
                                                               supervisor_pid=os.getpid()))
                        with (WORK / f'resources_pid_{child_pid}.jsonl').open('a') as log:
                            log.write(json.dumps(process) + '\n')
                        rss, free = process['resident_bytes'], process['system_free_memory_percent']
                        attempt['sampled_peak_resident_bytes'] = max(attempt['sampled_peak_resident_bytes'], rss)
                        samples += 1
                        low = free is not None and (free < 12 or (rss > 18e9 and free < 20))
                        pressure_streak = pressure_streak + 1 if low else 0
                        urgent = rss > 20e9 or (free is not None and free < 5)
                        if workers == 4 and (urgent or pressure_streak >= 3):
                            attempt.update(memory_fallback_trigger=process, stopped_utc=now())
                            state['stage'] = 'reducing_workers_for_memory'
                            save(WORK / 'supervisor_status.json', state)
                            print('Sustained memory pressure: preserving the last checkpoint and returning to two workers.', flush=True)
                            stop(runner, child_pid)
                            # A fresh cadence gives the return-to-two recipe a new
                            # immutable key; previous two/four-worker snapshots stay intact.
                            subprocess.run([sys.executable, str(WORK / 'change_workers.py'), '--workers', '2',
                                            '--batch-size', '129'], cwd=ROOT, check=True)
                            fallback = True
                            break
                        if samples % 12 == 0:
                            print(f'{now()} workers={workers}; RAM={rss / 1e9:.2f} GB; '
                                  f'sampled peak={attempt["sampled_peak_resident_bytes"] / 1e9:.2f} GB; free={free}%', flush=True)
                        save(WORK / 'supervisor_status.json', state)
                    time.sleep(5)
                if fallback:
                    child_pid = None
                    continue
                code = runner.wait()
                attempt.update(finished_utc=now(), returncode=code)
                status = read(WORK / 'status.json')
                if code or not status.get('complete'):
                    raise RuntimeError(f'E10 runner failed or is incomplete; see {WORK / "status.json"}')
                state.update(stage='reconstruction_and_cleanup_complete', complete=True, finished_utc=now())
                save(WORK / 'supervisor_status.json', state)
                print(json.dumps(state, indent=2), flush=True)
                break
        except BaseException as error:
            if runner and runner.poll() is None:
                stop(runner, child_pid)
            state.update(stage='failed', error=str(error), finished_utc=now())
            save(WORK / 'supervisor_status.json', state)
            raise


if __name__ == '__main__':
    main()
