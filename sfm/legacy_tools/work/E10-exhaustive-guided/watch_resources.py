#!/usr/bin/env python3
"""Sample the current light reconstruction's resident RAM and CPU; never mutate it."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pid', type=int, required=True)
args = parser.parse_args()
work = Path(__file__).resolve().parent
output = work / f'resources_pid_{args.pid}.jsonl'
maximum = 0
count = 0
with output.open('x') as log:
    while True:
        result = subprocess.run(['ps', '-p', str(args.pid), '-o', 'rss=,pcpu=,command='],
                                text=True, capture_output=True)
        fields = result.stdout.strip().split(maxsplit=2)
        if len(fields) != 3 or 'run_quality.py' not in fields[2] or 'light_shirt_quality' not in fields[2]:
            break
        value = {'time_utc': datetime.now(timezone.utc).isoformat(), 'pid': args.pid,
                 'resident_bytes': int(fields[0]) * 1024, 'cpu_percent': float(fields[1])}
        maximum = max(maximum, value['resident_bytes'])
        count += 1
        log.write(json.dumps(value) + '\n')
        log.flush()
        if count % 12 == 0:
            print(f'SfM RAM {value["resident_bytes"] / 1e9:.2f} GB; sampled peak {maximum / 1e9:.2f} GB', flush=True)
        time.sleep(5)
print(json.dumps({'pid': args.pid, 'samples': count, 'sampled_peak_resident_GB': maximum / 1e9,
                  'ended_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
