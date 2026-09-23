#!/usr/bin/env python3
"""Prepare internal E10 exhaustive+guided inputs without running matching.

All E3 and baseline assets stay read-only. Only new internal E10 datasets and
preparation reports are written; final consensus90 results belong to the runner.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import argparse
import hashlib,json,math,os,shutil,sys

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
OUT=WORK/'datasets'
SUBJECTS={'light_shirt':(125,7750),'black_shirt_crutches':(290,41905)}
sys.path.insert(0,str(ROOT/'work/quality-sfm'))
from audit_database import _select_checkpoint
import pycolmap


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subjects',nargs='*',choices=tuple(SUBJECTS),default=list(SUBJECTS))
    args=parser.parse_args();subjects={name:SUBJECTS[name] for name in args.subjects}
    # Preflight every destination before creating either subject.
    for subject in subjects:
        destination=OUT/f'{subject}_quality'
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Preserving existing E10 preparation: {destination}')
    if (WORK/'preparation.json').exists():
        raise FileExistsError(f'Preserving existing report: {WORK/"preparation.json"}')
    report={'experiment':'E10_quality_exhaustive_guided','scope':'Preparation only; internal intermediates, no matching or reconstruction started.',
            'prepared_utc':datetime.now(timezone.utc).isoformat(),'subjects':{},'old_results_preserved':True,
            'final_result_policy':'Only consensus90 cleaned results are published; these datasets stay under work.',
            'intended_final_root':str(ROOT/'reconstructions/experiments/E10_quality_exhaustive_guided_consensus90')}
    for subject,(expected_images,expected_pairs) in subjects.items():
        source=ROOT/'reconstructions'/f'{subject}_quality';destination=OUT/f'{subject}_quality'
        source_config=json.loads((source/'sfm_refine.json').read_text());config=dict(source_config)
        for key in ('baseline_dataset','boxes','guided_lock'):
            config[key]=os.path.relpath((source/source_config[key]).resolve(),destination)
        config.update(matching_pairs='matching_pairs.txt',guided_pairs='guided_pairs.txt',
            resume_matching=True,matching_batch_size=128,matching_threads=1)
        baseline=(destination/config['baseline_dataset']).resolve();boxes=(destination/config['boxes']).resolve()
        image_root=(source/'images').resolve()
        assert baseline==ROOT/'reconstructions'/subject
        assert boxes==(source/source_config['boxes']).resolve()
        source_metadata,cache_metadata,feature_key,ignored=_select_checkpoint(source,source_config,
            json.loads((source/'colmap/sparse/sfm_inputs.json').read_text()))
        source_cache=source_metadata.with_suffix('.db');signature=cache_metadata['signature']
        images=sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
        names=[p.relative_to(image_root).as_posix() for p in images]
        assert len(names)==len(set(names))==expected_images
        assert not any(any(c.isspace() for c in name) for name in names)
        assert signature['images']==[[name,p.stat().st_size,p.stat().st_mtime_ns] for name,p in zip(names,images)]
        baseline_files=[baseline/'colmap/database.db']+sorted(p for p in (baseline/'colmap/sparse').rglob('*') if p.is_file())
        assert signature['baseline']==[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in baseline_files]
        assert signature['boxes_sha256']==sha(boxes)
        assert signature['pycolmap']==pycolmap.__version__
        assert signature['max_features']==config['max_features']==18000
        assert feature_key==hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        assert source_cache.stat().st_size==cache_metadata['database_size']
        assert source_cache.with_name(source_cache.name+'-wal').stat().st_size==0 if source_cache.with_name(source_cache.name+'-wal').exists() else True
        pairs=list(combinations(names,2))
        assert len(pairs)==len(set(pairs))==expected_pairs==math.comb(expected_images,2)
        assert all(a<b and a!=b for a,b in pairs)
        degrees=Counter(name for pair in pairs for name in pair)
        assert set(degrees)==set(names) and set(degrees.values())=={expected_images-1}
        pair_bytes=''.join(f'{a} {b}\n' for a,b in pairs).encode('utf-8')
        watched=[source/'sfm_refine.json',source/'matching_pairs.txt',source/'guided_pairs.txt',
            source/'camera_groups.json',source/'input_manifest.json',source_cache,source_metadata,boxes,*baseline_files,
            *(p for p in (source/'colmap/sparse').rglob('*') if p.is_file())]
        old_hashes={str(p):sha(p) for p in dict.fromkeys(watched)}
        image_records=[{'name':name,'resolved_path':str(p.resolve()),'size_bytes':p.stat().st_size,
            'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)} for name,p in zip(names,images)]
        cache=destination/'colmap/quality_feature_cache';cache.mkdir(parents=True)
        (destination/'images').symlink_to(os.path.relpath(image_root,destination),target_is_directory=True)
        for filename in ('camera_groups.json','input_manifest.json'):
            shutil.copy2(source/filename,destination/filename)
            assert sha(source/filename)==sha(destination/filename)
        (destination/'matching_pairs.txt').write_bytes(pair_bytes)
        (destination/'guided_pairs.txt').write_bytes(pair_bytes)
        write(destination/'sfm_refine.json',config)
        copies=[]
        for original in (source_cache,source_metadata):
            copied=cache/original.name;shutil.copy2(original,copied)
            first,last=original.stat(),copied.stat()
            same_inode=(first.st_dev,first.st_ino)==(last.st_dev,last.st_ino)
            assert not copied.is_symlink() and not same_inode
            assert sha(original)==sha(copied)
            copies.append({'source':str(original),'copy':str(copied),'sha256':sha(copied),
                'source_device_inode':[first.st_dev,first.st_ino],'copy_device_inode':[last.st_dev,last.st_ino],
                'independent_inode':True,'byte_identical':True})
        assert (destination/'images').resolve()==image_root
        assert (destination/'matching_pairs.txt').read_bytes()==(destination/'guided_pairs.txt').read_bytes()
        parsed=[tuple(line.split()) for line in (destination/'matching_pairs.txt').read_text().splitlines()]
        assert set(parsed)==set(combinations(names,2)) and len(parsed)==expected_pairs
        for path,digest in old_hashes.items():
            assert sha(path)==digest,f'Existing source changed during preparation: {path}'
        for row in image_records:
            p=Path(row['resolved_path']);assert p.stat().st_size==row['size_bytes'] and p.stat().st_mtime_ns==row['mtime_ns']
        provenance={'experiment':'E10_quality_exhaustive_guided','subject':subject,'prepared_utc':report['prepared_utc'],
            'internal_intermediate':True,'matching_started':False,'source_E3_dataset':str(source),'new_dataset':str(destination),
            'intended_final_consensus90_dataset':str(Path(report['intended_final_root'])/subject),
            'change':'Every unordered image pair is requested in ordinary and guided matching. Inputs, features, crops, calibrated poses and triangulation constants stay identical to E3.',
            'image_count':expected_images,'pair_count_formula':f'{expected_images}*({expected_images}-1)/2',
            'ordinary_pairs':expected_pairs,'guided_pairs':expected_pairs,'guided_byte_identical_to_ordinary':True,
            'pair_sha256':sha(destination/'matching_pairs.txt'),
            'pair_coverage_assertions':{'all_unordered_combinations_exactly_once':True,'no_self_pairs':True,'no_duplicate_pairs':True,
                'all_views_covered':True,'all_pairs_lexically_canonical':True,'minimum_degree':min(degrees.values()),'maximum_degree':max(degrees.values())},
            'config':config,'resolved_baseline':str(baseline),'resolved_boxes':str(boxes),'resolved_images':str(image_root),
            'feature_checkpoint_key':feature_key,'feature_checkpoint_signature':signature,'feature_copies':copies,
            'high_quality_constants':{'max_features':18000,'extraction':signature['extraction'],'crop_padding':signature['crop_padding'],
                'spatial_cells':signature['spatial_cells'],'fixed_poses':True,'refine_intrinsics':False,'max_reprojection_error':3.5,'min_angle':1.5},
            'immutable_image_manifest':image_records,'old_files_sha256':old_hashes,'old_watched_files_unchanged_after_preparation':True,
            'no_model_points_added_changed_or_deleted':True,'application_code_modified':False}
        write(destination/'experiment_provenance.json',provenance)
        report['subjects'][subject]={key:provenance[key] for key in ('new_dataset','image_count','ordinary_pairs','guided_pairs',
            'pair_sha256','feature_checkpoint_key','resolved_baseline','resolved_boxes','resolved_images','pair_coverage_assertions','feature_copies')}
        print(f'PREPARED {subject}: {expected_images} images, {expected_pairs} ordinary pairs, {expected_pairs} guided pairs; matching not started.',flush=True)
    report.update(complete=True,matching_started=False)
    write(WORK/'preparation.json',report)
    print('E10 PREPARATION COMPLETE; no full matching job was launched.',flush=True)

if __name__=='__main__':main()
