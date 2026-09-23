#!/usr/bin/env python3
"""Copy evenly spaced light-shirt images for a clearly labelled CPU-only pilot.

Selection uses the preserved input-manifest order, not camera-folder order.
Only a new DATA_ROOT/sfm/inputs folder is created. Full E11/E12 inputs/results
are read-only and each selected image retains its camera-group subdirectory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfm_engine.pipeline import SFM_DATA_ROOT, atomic_json, discover_images, require_under, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('staging requires an existing approved Slurm compute job')
    source = require_under(SFM_DATA_ROOT / 'inputs/light_shirt', SFM_DATA_ROOT, 'source')
    manifest_path = source / 'input_manifest.json'
    manifest_hash = sha256_file(manifest_path)
    source_rows = json.loads(manifest_path.read_text())
    if args.count < 2 or args.count > len(source_rows):
        parser.error('count must be between 2 and the source image count')
    all_paths = discover_images(source / 'images')
    names = [row['image'] for row in source_rows]
    if len(set(names)) != len(names) or set(names) != {p.relative_to(source / 'images').as_posix() for p in all_paths}:
        raise ValueError('manifest must reference each available input exactly once')
    # Integer nearest-neighbour sampling includes both endpoints, without float
    # rounding ambiguity: 16 picks from 125 gives indices 0,8,...,116,124.
    indices = [(i * (len(names) - 1) + (args.count - 1) // 2) // (args.count - 1) for i in range(args.count)]
    if len(set(indices)) != args.count:
        raise ValueError('selection unexpectedly contains duplicate indices')
    destination = require_under(SFM_DATA_ROOT / 'inputs' / f'light_shirt_cpu_smoke{args.count}', SFM_DATA_ROOT / 'inputs', 'pilot destination')
    if destination.exists():
        raise FileExistsError(f'pilot destination already exists: {destination}')
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for index in indices:
        row = source_rows[index]
        original = require_under(source / 'images' / row['image'], source / 'images', 'source image')
        copied = require_under(destination / 'images' / row['image'], destination, 'pilot image')
        before = original.stat()
        original_hash = sha256_file(original)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, copied)
        after = original.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f'source changed during copy: {original}')
        if copied.stat().st_size != before.st_size or sha256_file(copied) != original_hash or sha256_file(original) != original_hash:
            raise RuntimeError(f'pilot image copy verification failed: {original}')
        if original.stat().st_ino == copied.stat().st_ino and original.stat().st_dev == copied.stat().st_dev:
            raise RuntimeError('pilot inputs must be independent physical copies')
        records.append({'manifest_index_zero_based': index, 'original_record': row,
                        'source': str(original), 'destination_relative': 'images/' + row['image'],
                        'bytes': before.st_size, 'sha256': original_hash})
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError('source manifest changed while copying')
    receipt = {'complete': True, 'pilot_only': True, 'selection': 'evenly spaced indices in preserved manifest order, endpoints included',
               'source_manifest': str(manifest_path), 'source_manifest_sha256': manifest_hash,
               'source_image_count': len(names), 'pilot_image_count': len(records),
               'indices_zero_based': indices, 'preserved_camera_folders': True,
               'physical_copies_sha256_verified': True, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
               'images': records}
    atomic_json(destination / 'copy_provenance.json', receipt)
    atomic_json(destination / 'input_manifest.json', {'source': str(manifest_path), 'pilot_only': True, 'images': [r['original_record'] for r in records]})
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
