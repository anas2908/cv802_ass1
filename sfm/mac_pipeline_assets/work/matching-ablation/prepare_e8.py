#!/usr/bin/env python3
"""Create an independent E8 dataset from E3 features and vocabulary pair lists."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pairs(path):
    result = set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        row = line.split()
        if len(row) != 2 or row[0] == row[1]:
            raise ValueError(f'Invalid pair: {line}')
        result.add(tuple(sorted(row)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=['light_shirt', 'black_shirt_crutches'])
    parser.add_argument('--pairs-directory', required=True, type=Path)
    args = parser.parse_args()
    source = ROOT / 'reconstructions' / (args.subject + '_quality')
    parent = ROOT / 'reconstructions/experiments/E8_vocab_guided'
    target = parent / source.name
    pair_dir = args.pairs_directory.resolve()
    if target.exists():
        raise FileExistsError(f'Preserving existing dataset: {target}')
    ordinary = pairs(pair_dir / 'matching_pairs.txt')
    guided = pairs(pair_dir / 'guided_pairs.txt')
    image_root = (source / 'images').resolve()
    names = {p.relative_to(image_root).as_posix() for p in image_root.rglob('*.jpg')}
    if not ordinary or not guided or not guided <= ordinary:
        raise ValueError('E8 requires a nonempty guided subset of vocabulary pairs')
    if {n for p in ordinary for n in p} != names:
        raise ValueError('Vocabulary pairs must reference exactly all input images')
    config = json.loads((source / 'sfm_refine.json').read_text())
    for key in ('baseline_dataset', 'boxes', 'guided_lock'):
        config[key] = os.path.relpath((source / config[key]).resolve(), target)
    checkpoints = list((source / 'colmap/quality_feature_cache').glob('*.json'))
    if len(checkpoints) != 1:
        raise ValueError('Select an unambiguous completed E3 feature checkpoint')
    metadata = checkpoints[0]
    database = metadata.with_suffix('.db')
    saved = json.loads(metadata.read_text())
    if database.stat().st_size != saved['database_size']:
        raise ValueError('Incomplete feature checkpoint')
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + args.subject + '_', dir=parent) as temporary:
        staging = Path(temporary) / 'dataset'
        cache = staging / 'colmap/quality_feature_cache'
        cache.mkdir(parents=True)
        (staging / 'images').symlink_to(os.path.relpath(image_root, target), target_is_directory=True)
        copied = {}
        for source_path, relative in [(source / 'camera_groups.json', 'camera_groups.json'),
                                      (source / 'input_manifest.json', 'input_manifest.json'),
                                      (database, 'colmap/quality_feature_cache/' + database.name),
                                      (metadata, 'colmap/quality_feature_cache/' + metadata.name),
                                      (pair_dir / 'matching_pairs.txt', 'matching_pairs.txt'),
                                      (pair_dir / 'guided_pairs.txt', 'guided_pairs.txt')]:
            shutil.copy2(source_path, staging / relative)
            sha = digest(source_path)
            if digest(staging / relative) != sha:
                raise AssertionError('Copied input does not match its source')
            copied[relative] = {'source': str(source_path), 'sha256': sha}
        (staging / 'sfm_refine.json').write_text(json.dumps(config, indent=2) + '\n')
        provenance = {
            'experiment': 'E8', 'subject': args.subject,
            'comparison_reference': str(source), 'input_images': len(names),
            'actual_pair_selection': 'PyCOLMAP vocabulary-tree retrieval; frozen explicit image pairs',
            'pair_preparation_directory': str(pair_dir),
            'ordinary_pairs': len(ordinary), 'guided_matching': True,
            'guided_pairs': len(guided), 'guided_policy': 'See vocabulary pair selection provenance; guided subset remains within retrieved pairs',
            'features': 'Byte-identical E3 completed checkpoint, independent database copy',
            'images': 'Same native image files, linked without modification',
            'cameras': 'Same fixed baseline poses and native-scaled calibration as E3',
            'triangulation': 'Same E3 settings; no dense reconstruction',
            'copied_inputs': copied,
            'gui_matcher_note': 'The GUI dropdown does not describe this saved quality recipe. The quality branch uses these explicit vocabulary-retrieved pair lists.'
        }
        (staging / 'experiment_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        (staging / 'README.md').write_text(
            '# E8: vocabulary-tree pairing with guided matching\n\n'
            'Uses the same native images, cached affine/DSP SIFT features, fixed cameras and triangulation as E3. '
            'Vocabulary retrieval chooses the candidate image pairs. A selected subset uses guided matching. '
            'See experiment_provenance.json and the referenced retrieval audit for actual settings.\n\n'
            'Open subject_preview in the starter GUI for the person view, or this folder for the full sparse model. '
            'The GUI matcher dropdown is not the provenance of this saved reconstruction.\n')
        staging.rename(target)
    print(json.dumps({'dataset': str(target), 'images': len(names), 'ordinary_pairs': len(ordinary), 'guided_pairs': len(guided)}, indent=2))


if __name__ == '__main__':
    main()
