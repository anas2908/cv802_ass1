#!/usr/bin/env python3
"""Compare final consensus90 E10 results with E4 in the common E1 frame.

Run after the final light-shirt E10 dataset exists. Only a new comparison directory is
written. --self-test independently repeats E4 in both columns and never reads
unfinished E10 work. No reconstruction, alignment, or input mutation occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys,tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'work/quality-sfm'))
sys.path.insert(0,str(ROOT/'work/matching-ablation'))
from compare_models import image_states,model_at,summarize
from plot_comparison import comparable_points,frame_from_baseline
from compare_experiments import check_cameras,model_hashes,sha256

SUBJECTS=('light_shirt',)
LABELS={'light_shirt':'Light shirt'}


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects',nargs='+',choices=SUBJECTS,default=list(SUBJECTS))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--point-size',type=float,default=1.8)
    parser.add_argument('--dpi',type=int,default=180)
    args=parser.parse_args(argv)
    if len(set(args.subjects))!=len(args.subjects):parser.error('Subjects must be unique')
    if not np.isfinite(args.point_size) or args.point_size<=0 or args.dpi<=0:parser.error('Invalid plotting settings')
    args.min_track,args.max_error,args.min_angle=3,3.,1.5
    args.output_dir=(args.output_dir or (WORK/'self_tests/repeated_e4_light_only' if args.self_test else WORK/'comparison')).resolve()
    try:args.output_dir.relative_to(WORK)
    except ValueError:parser.error('Comparison outputs must stay under work/E10-exhaustive-guided')
    return args


def source_paths(subject,self_test=False,root=ROOT):
    if subject not in SUBJECTS:raise ValueError('Unknown subject')
    experiments=root/'reconstructions/experiments'
    e4=experiments/'light_person_mask_cleanup/mask_consensus_90'
    e10=e4 if self_test else experiments/'E10_quality_exhaustive_guided_consensus90'/subject
    return e4,e10


def preflight(args):
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f'Preserving existing comparison: {args.output_dir}')
    for subject in args.subjects:
        for dataset in (ROOT/'reconstructions'/subject/'subject_preview',*source_paths(subject,args.self_test)):
            if not dataset.is_dir() or not (dataset/'analysis.json').is_file():
                raise FileNotFoundError(f'Final saved input is not ready: {dataset}')


def check_repeated(first,second,first_metrics,second_metrics,first_grids,second_grids):
    checks={'identical_display_coordinates':bool(np.array_equal(first['xyz'],second['xyz'])),
        'identical_rgb':bool(np.array_equal(first['rgb'],second['rgb'])),
        'identical_independently_measured_metrics':first_metrics==second_metrics,
        'identical_voxel_sets':first_grids==second_grids}
    if not all(checks.values()):raise AssertionError(f'Repeated E4 self-test failed: {checks}')
    return checks


def check_cleanup_policy(first,second):
    if not isinstance(first,dict) or not isinstance(second,dict):
        raise ValueError('Both final inputs must record their cleanup filter settings')
    for policy in (first,second):
        agreement=policy.get('foreground_agreement')
        if not isinstance(agreement,(int,float)) or not np.isfinite(agreement) or abs(agreement-.9)>1e-12:
            raise ValueError('Only final 90% mask-consensus inputs may be compared')
    if first!=second:
        raise ValueError('E4 and E10 cleanup filter settings differ; the same-policy comparison cannot proceed')
    return {'passed':True,'same_saved_filter_settings':True,'foreground_agreement':.9,
            'policy':'Foreground votes pooled over all usable in-image positive-depth projections'}


def load_subject(subject,args):
    e1=ROOT/'reconstructions'/subject/'subject_preview'
    e1_model_path,baseline=model_at(e1)
    basis,origin,low,high,analysis_path=frame_from_baseline(e1)
    reference=image_states(baseline)
    frozen=model_hashes(e1_model_path)
    frozen[str(analysis_path.resolve())]={'bytes':analysis_path.stat().st_size,'sha256':sha256(analysis_path)}
    plots,rows,grids=[],[],[]
    for label,dataset in zip(('E4','E10'),source_paths(subject,args.self_test)):
        path,model=model_at(dataset)
        input_hashes=model_hashes(path)
        cleanup_path=dataset/'analysis.json'
        cleanup=json.loads(cleanup_path.read_text())
        input_hashes[str(cleanup_path.resolve())]={'bytes':cleanup_path.stat().st_size,'sha256':sha256(cleanup_path)}
        for filename,signature in input_hashes.items():
            if filename in frozen and frozen[filename]!=signature:raise RuntimeError(f'Input changed between reads: {filename}')
            frozen[filename]=signature
        cameras=check_cameras(baseline,model,reference)
        metrics,grid=summarize(model,reference,basis,origin,low,high,args)
        plot=comparable_points(model,reference,basis,origin,low,high,args)
        for key,value in plot['summary'].items():
            if metrics[key]!=value:raise AssertionError(f'Metric/plot selection mismatch: {subject} {label} {key}')
        if len(plot['xyz'])!=metrics['retained_comparable_points']:raise AssertionError('Plot point count disagrees with metrics')
        rows.append({'experiment':label,'actual_input_experiment':'E4' if args.self_test else label,
            'dataset':str(dataset),'model':str(path),'input_model_and_analysis_hashes':input_hashes,
            'camera_check_against_E1':cameras,'metrics':metrics,
            'input_cleanup_metadata':{key:cleanup[key] for key in ('experiment','filter','group_consensus','group_filter','limitations') if key in cleanup}})
        plots.append(plot);grids.append(grid)
    cleanup_check=check_cleanup_policy(rows[0]['input_cleanup_metadata'].get('filter'),rows[1]['input_cleanup_metadata'].get('filter'))
    checks=check_repeated(plots[0],plots[1],rows[0]['metrics'],rows[1]['metrics'],grids[0],grids[1]) if args.self_test else None
    first,second=rows[0]['metrics'],rows[1]['metrics']
    audit={'subject':subject,'E1_reference_model':str(e1_model_path),'E1_frame_analysis':str(analysis_path),
        'models':rows,'self_test_checks':checks,'cleanup_policy_check':cleanup_check,
        'comparable_point_change':second['retained_comparable_points']-first['retained_comparable_points'],
        'comparable_point_ratio_E10_to_E4':second['retained_comparable_points']/first['retained_comparable_points'],
        'voxel_comparison':{key:{'E4':len(grids[0][key]),'E10':len(grids[1][key]),
            'shared':len(grids[0][key]&grids[1][key]),'E10_only':len(grids[1][key]-grids[0][key]),
            'E4_only':len(grids[0][key]-grids[1][key])} for key in grids[0]},
        'fixed_frame':{'world_to_upright_row_matrix':basis.tolist(),'origin_world':origin.tolist(),
            'selection_upright_min':low.tolist(),'selection_upright_max':high.tolist(),
            'display_center':((low+high)/2).tolist(),'equal_axis_radius':float(np.max(high-low))*.535}}
    return plots,audit,frozen


def draw_subject(staging,subject,plots,audit,args):
    center=np.asarray(audit['fixed_frame']['display_center']);radius=audit['fixed_frame']['equal_axis_radius']
    paths={}
    for horizontal,view in ((0,'front'),(1,'side')):
        figure,axes=plt.subplots(1,2,figsize=(11,7.5),layout='constrained')
        figure.suptitle(f'{LABELS[subject]} — {view} view'+(' — SELF-TEST: E4 repeated' if args.self_test else ''),fontsize=18)
        for column,(axis,points,row) in enumerate(zip(axes,plots,audit['models'])):
            xyz=points['xyz'];metrics=row['metrics']
            axis.scatter(xyz[:,horizontal],xyz[:,2],c=points['rgb'],s=args.point_size,linewidths=0,alpha=1,rasterized=True)
            axis.set(xlim=(center[horizontal]-radius,center[horizontal]+radius),ylim=(center[2]-radius,center[2]+radius),
                xlabel=('Horizontal' if horizontal==0 else 'Depth')+' (arbitrary units)',ylabel='Up (arbitrary units)')
            axis.set_aspect('equal',adjustable='box');axis.grid(alpha=.13)
            label=('E4 · earlier 90% cleanup' if column==0 else 'E10 · exhaustive + guided · 90% cleanup')
            if args.self_test:label='E4 · identical input independently measured'
            axis.set_title(label+f'\n{metrics["input_subject_points"]:,} input points → {len(xyz):,} comparable',fontsize=11,pad=10)
        figure.supxlabel('Same E1 frame, scale, bounds and marker size; all comparable points shown.\n'
            'Distinct views ≥ 3 · mean error ≤ 3 E1 pixels · maximum acute triangulation angle ≥ 1.5°.\n'
            'Outside-bound points are counted separately. Sparse coverage does not establish anatomical accuracy.',fontsize=9)
        name=f'{subject}_{view}.png';figure.savefig(staging/name,dpi=args.dpi);plt.close(figure)
        paths[view]=str(args.output_dir/name)
    return paths


def readme(report):
    lines=['# Light-shirt E10 compared with E4','']
    if report['self_test']:lines+=['**Self-test only: both columns independently measure the same E4 input. These are not E10 results.**','']
    lines+=['Only the final light-shirt 90% cleanup models are compared. Both use the same pooled-view mask-consensus policy, verified from their saved filter settings. E1 supplies the same camera pixel units, upright frame, origin and bounds for both columns. Comparable points require at least three distinct image views, mean error ≤ 3 E1 pixels, and maximum acute triangulation angle ≥ 1.5°. Repeated features in one photograph do not count as additional views.','',
        '| Subject | Input | Saved points | Comparable points | Mean error (E1 px) | Occupied voxels (height/100) | Quality points outside bounds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for subject in report['subjects']:
        for row in subject['models']:
            m=row['metrics'];label=row['experiment'] if not report['self_test'] else 'E4 repeat'
            lines.append(f'| {LABELS[subject["subject"]]} | {label} | {m["input_subject_points"]:,} | {m["retained_comparable_points"]:,} | {m["baseline_pixel_mean_point_error_retained"]["mean"]:.4f} | {m["occupied_voxels"]["height/100"]:,} | {m["quality_points_outside_fixed_baseline_bounds"]:,} |')
    lines+=['','The light-shirt comparison has front and side PNGs with identical axes and marker size. comparison.json includes camera checks, source hashes, detailed track/error/angle statistics, vertical coverage, voxel overlap and each input’s cleanup policy.','',
        'The 90% setting describes mask consensus, not reconstruction accuracy. E4 and E10 use identical saved cleanup settings, including the same pooled 90% foreground-consensus threshold. More points or occupied voxels may also reflect noise. Fixed E1 bounds can exclude real extremities, and no ground-truth body geometry is available. No models were aligned or edited.','']
    return '\n'.join(lines)


def main(argv=None):
    args=arguments(argv);preflight(args)
    helper_paths=[ROOT/'work/quality-sfm/compare_models.py',ROOT/'work/quality-sfm/plot_comparison.py',ROOT/'work/matching-ablation/compare_experiments.py']
    frozen={str(p):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in helper_paths}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'self_test':args.self_test,
        'comparison_script_sha256':sha256(Path(__file__)),
        'method':{'reference':'Light shirt only: E1 original subject preview; E4 and final E10 only',
            'cleanup_policy':'Same pooled-view 90% mask consensus; complete saved filter dictionaries must be identical',
            'min_distinct_views':3,'max_mean_error_E1_pixels':3.,'min_maximum_acute_angle_degrees':1.5,
            'pixel_normalization':'Per-observation residual x/y scaled to the matching E1 camera dimensions before averaging norms',
            'fixed_bounds':'Same saved E1 upright min/max; outside points excluded equally and counted separately',
            'point_marker_area':args.point_size,'sampling':'Every accepted point; original colors; no subsampling or geometry generation'},
        'measurement_helpers':dict(frozen),'subjects':[]}
    loaded=[]
    for subject in args.subjects:
        plots,audit,signatures=load_subject(subject,args);loaded.append((subject,plots,audit));frozen.update(signatures)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.E10_compare_',dir=args.output_dir.parent) as temp:
        staging=Path(temp)/'comparison';staging.mkdir()
        for subject,plots,audit in loaded:
            audit['plots']=draw_subject(staging,subject,plots,audit,args);report['subjects'].append(audit)
        for filename,signature in frozen.items():
            p=Path(filename)
            if p.stat().st_size!=signature['bytes'] or sha256(p)!=signature['sha256']:
                raise RuntimeError(f'Read-only comparison input changed: {filename}')
        report['all_input_hashes_unchanged']=True
        (staging/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (staging/'README.md').write_text(readme(report))
        if args.output_dir.exists():raise FileExistsError(f'Comparison appeared while running; preserving it: {args.output_dir}')
        staging.rename(args.output_dir)
    print(json.dumps({'output_dir':str(args.output_dir),'self_test':args.self_test,'all_input_hashes_unchanged':True,
        'comparable_points':{s['subject']:{row['experiment']:row['metrics']['retained_comparable_points'] for row in s['models']} for s in report['subjects']}},indent=2))

if __name__=='__main__':main()
