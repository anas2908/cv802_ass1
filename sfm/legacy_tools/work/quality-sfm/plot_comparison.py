#!/usr/bin/env python3
"""Plot baseline and quality sparse SfM with identical frames, bounds, and filters.

Default: compare both subjects under reconstructions/ and write
reconstructions/quality_comparison.png plus per-subject PNGs and a JSON audit.
Use --self-test to compare baseline against itself in work/quality-sfm/.
No reconstruction, alignment, image editing, or input/model mutation occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare_models import model_at, image_states, point_measurements, quantiles


SUBJECTS=('light_shirt','black_shirt_crutches')
LABELS={'light_shirt':'Light shirt','black_shirt_crutches':'Black shirt and crutches'}


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--subjects',nargs='+',choices=SUBJECTS,default=list(SUBJECTS))
    parser.add_argument('--output',type=Path,help='Combined PNG; default reconstructions/quality_comparison.png')
    parser.add_argument('--self-test',action='store_true',help='Plot baseline against itself; defaults to a separate work-folder output')
    parser.add_argument('--min-track',type=int,default=3,help='Minimum distinct image IDs observing each point; repeated observations in one image count once')
    parser.add_argument('--max-error',type=float,default=3.,help='Mean reprojection error in each observation\'s corresponding baseline-image pixel units')
    parser.add_argument('--min-angle',type=float,default=1.5,help='Minimum maximum acute track triangulation angle, in degrees')
    parser.add_argument('--point-size',type=float,default=1.8,help='Same marker area in points squared for all panels')
    parser.add_argument('--dpi',type=int,default=180)
    args=parser.parse_args()
    if args.min_track<2 or not 0<args.max_error or not 0<args.min_angle<=90 or args.point_size<=0:
        parser.error('Invalid quality or plotting settings')
    args.sfm_root=args.sfm_root.resolve()
    if args.output is None:
        args.output=args.sfm_root/('work/quality-sfm/quality_comparison_self_test.png' if args.self_test else 'reconstructions/quality_comparison.png')
    args.output=args.output.resolve()
    return args


def frame_from_baseline(preview):
    path=preview/'analysis.json'
    analysis=json.loads(path.read_text())
    basis=np.asarray(analysis['display_orientation']['world_to_display_row_matrix'],dtype=float)
    origin=np.asarray(analysis['plot']['display_origin_world'],dtype=float)
    bounds=analysis['plot']['upright_bounds']
    low=np.asarray(bounds['min'],dtype=float);high=np.asarray(bounds['max'],dtype=float)
    if basis.shape!=(3,3) or not np.isfinite(basis).all() or not np.allclose(basis.T@basis,np.eye(3),atol=1e-8):
        raise ValueError(f'Baseline display basis must be finite and orthonormal: {path}')
    if not np.isfinite([origin,low,high]).all() or not np.all(high>low):
        raise ValueError(f'Invalid fixed baseline origin/bounds: {path}')
    return basis,origin,low,high,path


def comparable_points(model,reference,basis,origin,low,high,args):
    # Imported measurement code is shared with compare_models.py. It recalculates
    # residuals with distortion and normalizes each observation's x/y separately.
    measurements=point_measurements(model,reference)
    display=(measurements['xyz']-origin)@basis
    finite=np.isfinite(display).all(axis=1)
    quality=finite&(measurements['distinct_views']>=args.min_track)&measurements['all_observations_valid']
    quality &= np.isfinite(measurements['normalized'])&(measurements['normalized']<=args.max_error)
    quality &= measurements['angles']>=args.min_angle
    inside=np.all((display>=low-1e-10)&(display<=high+1e-10),axis=1)
    keep=quality&inside
    colors=np.asarray([model.points3D[i].color for i in sorted(model.points3D)],dtype=float).reshape(-1,3)/255.
    summary={'input_subject_points':len(display),'points_with_track_at_least_minimum':int((measurements['tracks']>=args.min_track).sum()),
             'points_with_distinct_views_at_least_minimum':int((measurements['distinct_views']>=args.min_track).sum()),
             'track_length_retained':quantiles(measurements['tracks'][keep]),'distinct_image_views_retained':quantiles(measurements['distinct_views'][keep]),
             'points_passing_quality_before_fixed_bounds':int(quality.sum()),
             'quality_points_outside_fixed_baseline_bounds':int((quality&~inside).sum()),
             'retained_comparable_points':int(keep.sum()),'registered_images':int(model.num_reg_images()),
             'invalid_or_behind_camera_observations':measurements['invalid_observations'],
             'observations_without_baseline_image':measurements['observations_without_baseline_image']}
    if not keep.any():raise ValueError('No points pass the common quality filters inside fixed baseline bounds')
    # Every accepted point is plotted. No subsampling or synthetic points.
    return {'xyz':display[keep],'rgb':colors[keep],'summary':summary}


def pose_check(first,second):
    common=sorted(first.keys()&second.keys())
    if len(common)<2:raise ValueError('Models do not share enough registered image names')
    maximum=max(max(float(np.max(np.abs(first[name][key]-second[name][key]))) for key in ('rotation','translation')) for name in common)
    if maximum>1e-10:
        raise ValueError(f'Camera poses changed (max element difference {maximum:.6g}); a fixed-world-frame plot needs independent review. No alignment applied.')
    return {'common_images':len(common),'baseline_only_images':sorted(first.keys()-second.keys()),
            'quality_only_images':sorted(second.keys()-first.keys()),'max_pose_element_difference':maximum,
            'same_world_frame_verified_by_fixed_camera_poses':True}


def load_subject(subject,args):
    baseline_preview=args.sfm_root/'reconstructions'/subject/'subject_preview'
    quality_preview=baseline_preview if args.self_test else args.sfm_root/'reconstructions'/(subject+'_quality')/'subject_preview'
    baseline_path,baseline=model_at(baseline_preview)
    quality_path,quality=(baseline_path,baseline) if args.self_test else model_at(quality_preview)
    basis,origin,low,high,analysis_path=frame_from_baseline(baseline_preview)
    reference=image_states(baseline)
    poses=pose_check(reference,image_states(quality))
    first=comparable_points(baseline,reference,basis,origin,low,high,args)
    second=first if args.self_test else comparable_points(quality,reference,basis,origin,low,high,args)
    center=(low+high)/2
    # One spatial scale within a subject for front and side, and for both models.
    radius=float(np.max(high-low))*.535
    audit={'subject':subject,'baseline_model':str(baseline_path),'quality_model':str(quality_path),'baseline_analysis':str(analysis_path),
           'camera_pose_check':poses,'baseline':first['summary'],'quality':second['summary'],
           'fixed_frame':{'world_to_upright_row_matrix':basis.tolist(),'origin_world':origin.tolist(),
                          'quality_selection_upright_min':low.tolist(),'quality_selection_upright_max':high.tolist(),
                          'display_center':center.tolist(),'equal_axis_radius':radius},
           'comparable_point_ratio':second['summary']['retained_comparable_points']/first['summary']['retained_comparable_points']}
    return {'subject':subject,'first':first,'second':second,'center':center,'radius':radius,'audit':audit}


def draw_subject(axes,data,args,small=False):
    radius=data['radius'];center=data['center']
    for row,(horizontal,label) in enumerate(((0,'Front reference'),(1,'Side'))):
        for column,point_data in enumerate((data['first'],data['second'])):
            axis=axes[row,column]
            xyz=point_data['xyz']
            axis.scatter(xyz[:,horizontal],xyz[:,2],c=point_data['rgb'],s=args.point_size,
                         linewidths=0,alpha=1,rasterized=True)
            axis.set_xlim(center[horizontal]-radius,center[horizontal]+radius)
            axis.set_ylim(center[2]-radius,center[2]+radius)
            axis.set_aspect('equal',adjustable='box')
            axis.set_xlabel(('Horizontal' if horizontal==0 else 'Depth')+' (arbitrary units)',fontsize=10)
            axis.set_ylabel('Up (arbitrary units)',fontsize=10)
            axis.grid(alpha=.13);axis.tick_params(labelsize=9)
            model_label='Baseline' if column==0 else ('Baseline repeat' if args.self_test else 'Quality')
            axis.set_title(f'{model_label} · {label}\n{len(xyz):,} comparable points',fontsize=12,pad=8)


def caption(args):
    return (f'Identical filters: distinct views ≥ {args.min_track}; mean error ≤ {args.max_error:g} baseline pixels; '
            f'acute triangulation angle ≥ {args.min_angle:g}°.\n'
            'Fixed baseline orientation, origin and bounds; identical marker size; every accepted point shown.\n'
            'Scale is arbitrary. More points do not establish anatomical accuracy.')


def main():
    args=arguments()
    data=[load_subject(subject,args) for subject in args.subjects]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    n=len(data)
    figure,axes=plt.subplots(2*n,2,figsize=(11,9*n),layout='constrained',squeeze=False)
    title='Sparse SfM: baseline and quality comparison'
    if args.self_test:title+=' — self-test'
    figure.suptitle(title,fontsize=19)
    for index,subject_data in enumerate(data):
        section=axes[2*index:2*index+2,:]
        draw_subject(section,subject_data,args)
        section[0,0].set_title(LABELS[subject_data['subject']]+'\n'+section[0,0].get_title(),fontsize=13,pad=9)
        section[0,1].set_title(LABELS[subject_data['subject']]+'\n'+section[0,1].get_title(),fontsize=13,pad=9)
    figure.supxlabel(caption(args),fontsize=10)
    figure.savefig(args.output,dpi=args.dpi);plt.close(figure)
    per_subject={}
    for subject_data in data:
        path=args.output.with_name(args.output.stem+'_'+subject_data['subject']+'.png')
        figure,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
        figure.suptitle(LABELS[subject_data['subject']]+' — fixed-view SfM comparison'+(' (self-test)' if args.self_test else ''),fontsize=18)
        draw_subject(axes,subject_data,args)
        figure.supxlabel(caption(args),fontsize=9)
        figure.savefig(path,dpi=args.dpi);plt.close(figure)
        per_subject[subject_data['subject']]=str(path)
    report={'combined_png':str(args.output),'per_subject_pngs':per_subject,'self_test':args.self_test,
            'method':{'min_track_length':args.min_track,'min_distinct_image_views':args.min_track,
                      'track_gate':'Minimum distinct image IDs; repeated features in one image do not count as independent views. Raw track statistics are reported separately.',
                      'max_mean_error_baseline_pixels':args.max_error,
                      'min_maximum_acute_triangulation_angle_degrees':args.min_angle,'marker_area_points_squared':args.point_size,
                      'residual_implementation':'compare_models.point_measurements; per-observation x/y baseline pixel normalization',
                      'bounds_policy':'Both models use the exact min/max of the saved baseline upright cloud; out-of-bounds counts reported separately',
                      'display_policy':'Same baseline frame, centered equal-axis display limits, actual colors, opacity1, same marker size; no subsampling or geometry synthesis'},
            'subjects':[item['audit'] for item in data],
            'limitations':['A sparse point-cloud comparison, not a dense mesh or ground-truth accuracy measurement.',
                          'New extremities outside baseline bounds are excluded equally by the fixed comparison volume and counted separately.',
                          'Each subject uses its own arbitrary reconstruction scale; cross-subject apparent sizes are not metric comparisons.']}
    audit_path=args.output.with_suffix('.json');audit_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'combined_png':str(args.output),'audit_json':str(audit_path),'per_subject_pngs':per_subject,
                      'comparable_counts':{item['subject']:[item['first']['summary']['retained_comparable_points'],item['second']['summary']['retained_comparable_points']] for item in data}},indent=2))


if __name__=='__main__':
    main()
