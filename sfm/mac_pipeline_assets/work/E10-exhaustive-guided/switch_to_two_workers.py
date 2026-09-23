#!/usr/bin/env python3
"""Preserve a stopped E10 guided checkpoint while changing only worker count."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import closing
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile

WORK = Path(__file__).resolve().parent
DATASET = WORK / 'datasets/light_shirt_quality'
CACHE = DATASET / 'colmap/quality_matching_cache'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    with path.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    config_path = DATASET / 'sfm_refine.json'
    config = json.loads(config_path.read_text())
    if config.get('matching_threads') != 1 or config.get('resume_matching') is not True:
        raise ValueError('Expected the stopped one-worker E10 recipe')
    manifests = list(CACHE.glob('*/manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected one unambiguous source checkpoint')
    source_manifest_path = manifests[0]
    source = source_manifest_path.parent
    old = json.loads(source_manifest_path.read_text())
    source_key = old['checkpoint_key']
    content = {k: v for k, v in old.items() if k != 'checkpoint_key'}
    if digest(content) != source_key or source.name != source_key:
        raise ValueError('Source immutable recipe is invalid')
    options = content['recipe']['matching_options']
    if options['num_threads'] != 1 or options['guided_matching'] is not True or options['use_gpu'] is not False:
        raise ValueError('Source is not a CPU one-worker guided run')
    with (CACHE / (source_key + '.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        database = source / 'checkpoint.db'
        wal = Path(str(database) + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('Stable snapshot must not depend on a WAL')
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('Source checkpoint integrity check failed')
            raw = {r[0] for r in db.execute('SELECT pair_id FROM matches')}
            geometry = {r[0] for r in db.execute('SELECT pair_id FROM two_view_geometries')}
            if raw != geometry or not raw:
                raise ValueError('Source committed pairs are incomplete or empty')
        source_database_hash = sha(database)
        new_content = json.loads(json.dumps(content))
        new_content['recipe']['matching_options']['num_threads'] = 2
        restored = json.loads(json.dumps(new_content))
        restored['recipe']['matching_options']['num_threads'] = 1
        assert restored == content
        target_key = digest(new_content)
        new_manifest = dict(new_content, checkpoint_key=target_key)
        target = CACHE / target_key
        if target.exists():
            raise FileExistsError(f'Preserving existing checkpoint: {target}')
        with tempfile.TemporaryDirectory(prefix='.worker_switch_', dir=CACHE) as temporary:
            staged = Path(temporary) / 'checkpoint'
            staged.mkdir()
            shutil.copy2(database, staged / 'checkpoint.db')
            with (staged / 'checkpoint.db').open('rb') as stream:
                os.fsync(stream.fileno())
            assert sha(staged / 'checkpoint.db') == source_database_hash
            write(staged / 'manifest.json', new_manifest)
            write(staged / 'progress.json', {
                'checkpoint_key': target_key, 'checkpoint_database': str(target / 'checkpoint.db'),
                'updated_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'matching',
                'complete': False, 'total_pairs': old['total_pairs'], 'completed_pairs': len(raw),
                'remaining_pairs': old['total_pairs'] - len(raw), 'matching_threads': 2,
                'matching_batch_size': config['matching_batch_size'], 'guided_matching': True,
                'last_committed_batch_pairs': 0,
                'note': 'Verified inheritance of complete guided pairs from the stopped one-worker run.'})
            sync_dir(staged)
            staged.rename(target)
            sync_dir(CACHE)
        record = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'reason': 'User requested more workers; two selected within the Mac memory budget.',
            'source_checkpoint_dir': str(source), 'target_checkpoint_dir': str(target),
            'source_checkpoint_key': source_key, 'target_checkpoint_key': target_key,
            'source_manifest_sha256': sha(source_manifest_path),
            'target_manifest_sha256': sha(target / 'manifest.json'),
            'source_checkpoint_sha256': source_database_hash,
            'target_initial_checkpoint_sha256': sha(target / 'checkpoint.db'),
            'completed_pairs_transferred': len(raw), 'transferred_pair_ids_sha256': digest(sorted(raw)),
            'only_matching_option_changed': {'num_threads': {'before': 1, 'after': 2}},
            'source_feature_database_sha256': old['feature_database_sha256'],
            'config_before': config, 'config_after': dict(config, matching_threads=2),
            'scope': 'Light shirt only; all 7,750 pairs, guided ON, identical features and thresholds, consensus90 final output.'}
        migrations = WORK / 'worker_migrations'
        migrations.mkdir(exist_ok=True)
        path = migrations / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
        write(path, record)
        sync_dir(migrations)
        temporary_config = config_path.with_suffix('.json.tmp')
        write(temporary_config, record['config_after'])
        os.replace(temporary_config, config_path)
        sync_dir(DATASET)
        assert sha(database) == source_database_hash
        print(json.dumps({'record': str(path), 'preserved_pairs': len(raw),
                          'remaining_pairs': old['total_pairs'] - len(raw),
                          'matching_threads': 2, 'new_checkpoint': str(target)}, indent=2))


if __name__ == '__main__':
    main()
