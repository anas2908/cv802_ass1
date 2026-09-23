#!/usr/bin/env python3
"""Derive a reversible, GUI-readable subject-focused SfM preview offline.

Run only after the source reconstruction has finished. This does not re-run
SfM or change source images, cameras, points, or database. Rectangle consensus
is a heuristic; retained points can still include background.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Dataset with images/ and colmap/sparse/')
    parser.add_argument('--model', type=Path, help='Explicit sparse model; otherwise choose largest by registered images, then points')
    parser.add_argument('--boxes', type=Path, help='Box JSON dictionary; defaults to subject_boxes/<dataset-name>_boxes.json beside this script')
    parser.add_argument('--output', type=Path, help='New output folder; defaults to dataset/subject_preview. Must not exist.')
    parser.add_argument('--min-observations', type=int, default=2, help='Minimum distinct track images with detected boxes (default: 2)')
    parser.add_argument('--min-fraction', type=float, default=.65, help='Minimum fraction of boxed observations inside padded boxes (default: .65)')
    parser.add_argument('--min-track-length', type=int, default=3, help='Minimum number of distinct images supporting a 3D point (default: 3)')
    parser.add_argument('--max-error', type=float, default=4., help='Maximum mean point reprojection error in pixels (default: 4)')
    parser.add_argument('--min-confidence', type=float, default=.5, help='Ignore lower-confidence detections (default: .5)')
    parser.add_argument('--max-plot-points', type=int, default=60000)
    parser.add_argument('--volume-fraction', type=float, default=.8, help='Required inside-box fraction among valid in-image projections (default: .8)')
    parser.add_argument('--volume-min-view-fraction', type=float, default=.2, help='Minimum inside-box projections as a fraction of reliable cameras (default: .2)')
    args = parser.parse_args()
    if args.min_observations < 1 or args.min_track_length < 2 or not 0 <= args.min_fraction <= 1 or args.max_error <= 0 or not 0<=args.volume_fraction<=1 or not 0<=args.volume_min_view_fraction<=1:
        parser.error('Invalid filter settings')
    return args


def contains_model(directory):
    return any(all((directory / (name + ext)).is_file() for name in ('cameras','images','points3D')) for ext in ('.bin','.txt'))


def choose_model(dataset, explicit, pycolmap):
    if explicit:
        path = explicit.resolve()
        if not contains_model(path):
            raise ValueError(f'Incomplete source model: {path}')
        return path, pycolmap.Reconstruction(path)
    root = dataset / 'colmap' / 'sparse'
    if not root.is_dir():
        raise FileNotFoundError(f'Source reconstruction is not ready: {root}')
    candidates = [p for p in [root, *sorted(root.iterdir())] if p.is_dir() and contains_model(p)]
    best_path, best_model = None, None
    for path in candidates:
        model = pycolmap.Reconstruction(path)
        if model.num_reg_images() < 2 or model.num_points3D() == 0:
            continue
        if best_model is None or (model.num_reg_images(),model.num_points3D()) > (best_model.num_reg_images(),best_model.num_points3D()):
            best_path,best_model = path,model
    if best_model is None:
        raise ValueError(f'No non-empty connected source model found in {root}')
    return best_path,best_model


def coordinate_stats(xyz, np):
    if len(xyz) == 0:
        return {'count':0}
    low,high = np.quantile(xyz,[.01,.99],axis=0)
    return {'count':len(xyz), 'min':xyz.min(axis=0).tolist(), 'max':xyz.max(axis=0).tolist(),
            'median':np.median(xyz,axis=0).tolist(), 'robust_percentiles':[1,99],
            'robust_min':low.tolist(),'robust_max':high.tolist(),
            'robust_axis_spans':(high-low).tolist(),'robust_diagonal':float(np.linalg.norm(high-low))}


def write_ply(path,xyz,rgb,np):
    # Structured binary output avoids loading Open3D and preserves all retained points.
    vertices=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,key in enumerate(('x','y','z')): vertices[key]=xyz[:,i]
    for i,key in enumerate(('red','green','blue')): vertices[key]=rgb[:,i]
    header=('ply\nformat binary_little_endian 1.0\ncomment subject-focused sparse SfM preview; arbitrary scale\n'
            f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as file:
        file.write(header.encode('ascii'));file.write(vertices.tobytes())


def volume_consensus(xyz,model,image_boxes,args,np):
    """Intersect camera rectangles approximately, counting only projections inside images."""
    known=np.zeros(len(xyz),dtype=np.int32)
    inside=np.zeros(len(xyz),dtype=np.int32)
    for image_id,box in image_boxes.items():
        image=model.images[image_id];camera=model.cameras[image.camera_id]
        pose=image.cam_from_world();rotation=pose.rotation.matrix();translation=np.asarray(pose.translation)
        camera_xyz=xyz@rotation.T+translation
        positive=np.isfinite(camera_xyz).all(axis=1)&(camera_xyz[:,2]>1e-8)
        indices=np.flatnonzero(positive)
        if not len(indices): continue
        projected=camera.img_from_cam(np.ascontiguousarray(camera_xyz[indices]))
        in_image=np.isfinite(projected).all(axis=1)&(projected[:,0]>=0)&(projected[:,0]<camera.width)&(projected[:,1]>=0)&(projected[:,1]<camera.height)
        ix=indices[in_image];uv=projected[in_image]
        known[ix]+=1
        hit=(uv[:,0]>=box[0])&(uv[:,0]<=box[2])&(uv[:,1]>=box[1])&(uv[:,1]<=box[3])
        inside[ix[hit]]+=1
    fraction=np.divide(inside,known,out=np.zeros(len(xyz),dtype=float),where=known>0)
    minimum=max(args.min_observations,math.ceil(args.volume_min_view_fraction*len(image_boxes)))
    keep=(inside>=minimum)&(fraction>=args.volume_fraction)
    thresholds=sorted(set([.7,.8,.9,args.volume_fraction]))
    sensitivity={f'{v:.3f}':{'points':int(((inside>=minimum)&(fraction>=v)).sum()),'bounds':coordinate_stats(xyz[(inside>=minimum)&(fraction>=v)],np)} for v in thresholds}
    info={'reliable_cameras':len(image_boxes),'min_inside_views':minimum,'required_inside_fraction':args.volume_fraction,
          'min_inside_view_fraction':args.volume_min_view_fraction,'candidate_points':len(xyz),'retained_points':int(keep.sum()),
          'ignored_projections':'negative depth, outside image bounds, non-finite, or missing detector; these contribute no negative vote',
          'camera_projection':'PyCOLMAP Camera.img_from_cam including calibrated distortion',
          'sensitivity':sensitivity}
    return keep,info,{'in_image_votes':known,'inside_votes':inside,'fraction':fraction}


def upright_basis(model,image_boxes,np):
    """Display-only basis: robust camera up, earliest still-photo horizontal axis."""
    images=[model.images[i] for i in image_boxes]
    up_vectors=np.asarray([-im.cam_from_world().rotation.matrix().T[:,1] for im in images])
    initial=np.median(up_vectors,axis=0);initial/=np.linalg.norm(initial)
    consistent=up_vectors[up_vectors@initial>math.cos(math.radians(30))]
    up=np.mean(consistent if len(consistent) else up_vectors,axis=0);up/=np.linalg.norm(up)
    def capture_key(im):
        name=Path(im.name).name
        token=name.split('_',1)[0]
        return (0,int(token),name) if token.isdigit() else (1,0,name)
    reference=min(images,key=capture_key)
    right=reference.cam_from_world().rotation.matrix().T[:,0]
    right=right-up*np.dot(right,up);right/=np.linalg.norm(right)
    depth=np.cross(up,right);depth/=np.linalg.norm(depth)
    basis=np.column_stack([right,depth,up])
    return basis,{'reference_image':reference.name,'world_up':up.tolist(),'world_to_display_row_matrix':basis.tolist(),
                  'definition':'display rows = (world rows - display origin) @ matrix; X right, Y depth, Z up',
                  'scope':'PNG display only; COLMAP model and PLY remain in original coordinates'}


def plot_points(path,xyz,rgb,title,stats,args,np,basis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin=np.median(xyz,axis=0);display=(xyz-origin)@basis
    low,high=np.quantile(display,[.005,.995],axis=0)
    center=(low+high)/2
    radius=max(float((high-low).max())*.55,1e-6)
    visible=np.all((display>=center-radius)&(display<=center+radius),axis=1)
    samples=np.flatnonzero(visible)
    if len(samples)>args.max_plot_points:
        samples=np.random.default_rng(42).choice(samples,args.max_plot_points,replace=False)
    pts,col=display[samples],rgb[samples]/255.
    fig=plt.figure(figsize=(12,10),layout='constrained')
    fig.suptitle(f'{title} — subject-focused sparse SfM\n{len(xyz):,} retained points; {stats["registered_images"]} registered cameras',fontsize=16)
    labels=['Horizontal','Depth','Up']
    for panel,(a,b,label) in enumerate(((0,2,'Front reference'),(1,2,'Side'),(0,1,'Top')),1):
        ax=fig.add_subplot(2,2,panel)
        ax.scatter(pts[:,a],pts[:,b],c=col,s=2.8,linewidths=0,rasterized=True)
        ax.set(xlim=(center[a]-radius,center[a]+radius),ylim=(center[b]-radius,center[b]+radius),
               xlabel=labels[a]+' (arbitrary units)',ylabel=labels[b]+' (arbitrary units)',title=label+' orthographic view')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.16)
    ax=fig.add_subplot(2,2,4,projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=col,s=2.8,linewidths=0,depthshade=False,rasterized=True)
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='Horizontal',ylabel='Depth',zlabel='Up',title='Oblique view')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=-65)
    fig.supxlabel('Display oriented upright from camera rotations; output model coordinates unchanged. Scale is not metric.\nMulti-view rectangle filtering is approximate; this is sparse SfM, not a surface mesh.',fontsize=10)
    fig.savefig(path,dpi=180);plt.close(fig)
    return {'plotted_points':len(samples),'within_plot_bounds':int(visible.sum()),'total_retained':len(xyz),'display_origin_world':origin.tolist(),
            'display_center':center.tolist(),'equal_axis_radius':radius,'display_bounds_percentiles':[.5,99.5],'upright_bounds':coordinate_stats(display,np)}


def plot_sensitivity(path,xyz,rgb,votes,volume_info,basis,np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    thresholds=[.7,.8,.9]
    masks=[(votes['inside_votes']>=volume_info['min_inside_views'])&(votes['fraction']>=t) for t in thresholds]
    union=masks[0]
    if not union.any(): return
    origin=np.median(xyz[union],axis=0);display=(xyz-origin)@basis
    low,high=np.quantile(display[union],[.005,.995],axis=0);center=(low+high)/2;radius=max(float((high-low).max())*.55,1e-6)
    fig,axes=plt.subplots(1,3,figsize=(12,5),layout='constrained')
    for ax,threshold,mask in zip(axes,thresholds,masks):
        points=display[mask];colors=rgb[mask]/255
        ax.scatter(points[:,0],points[:,2],c=colors,s=2.5,linewidths=0,rasterized=True)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-radius,center[2]+radius),title=f'{threshold:.0%} consistency: {mask.sum():,} points',xlabel='Horizontal',ylabel='Up')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.16)
    fig.suptitle('Sensitivity to multi-view rectangle consistency — same scale and view',fontsize=14)
    fig.supxlabel(f'At least {volume_info["min_inside_views"]} reliable views must contain each point. Out-of-image projections ignored.',fontsize=10)
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    args=arguments()
    import numpy as np
    import pycolmap
    dataset=args.dataset.resolve()
    image_root=(dataset/'images').resolve()
    if not image_root.is_dir(): raise FileNotFoundError(f'Images folder not found: {image_root}')
    output=(args.output or dataset/'subject_preview').resolve()
    if output.exists(): raise FileExistsError(f'Preserving existing folder: {output}. Choose a different --output.')
    boxes_path=(args.boxes or Path(__file__).parent/'subject_boxes'/(dataset.name+'_boxes.json')).resolve()
    boxes=json.loads(boxes_path.read_text())
    if isinstance(boxes,list): boxes={r['image_name']:r for r in boxes if r.get('dataset')==dataset.name}
    model_path,model=choose_model(dataset,args.model,pycolmap)
    source_files={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in model_path.iterdir() if p.is_file()}
    image_boxes={}
    for image_id,image in model.images.items():
        r=boxes.get(image.name)
        if r and r.get('status')=='detected' and r.get('selected',{}).get('confidence',0)>=args.min_confidence and not r.get('ambiguity',False):
            box=np.asarray(r['padded_bbox_xyxy'],dtype=float)
            if np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]: image_boxes[int(image_id)]=box
    if len(image_boxes)<2: raise ValueError('Fewer than two registered images match reliable detected boxes; check image-name paths.')
    ids=[];xyz=[];colors=[];qualities=[];rejections=Counter();thresholds=sorted(set([.5,.65,.8,args.min_fraction]));sensitivity=Counter()
    for point_id,point in model.points3D.items():
        position=np.asarray(point.xyz,dtype=float)
        if not np.isfinite(position).all(): rejections['non_finite_position']+=1;continue
        xyz.append(position.copy());colors.append(np.asarray(point.color,dtype=np.uint8).copy());ids.append(int(point_id))
        track=list(point.track.elements);error=float(point.error)
        distinct_views=len({int(observation.image_id) for observation in track})
        quality_ok=distinct_views>=args.min_track_length and math.isfinite(error) and 0<=error<=args.max_error
        image_votes={}
        for observation in track:
            image_id=int(observation.image_id);box=image_boxes.get(image_id)
            if box is None: continue
            xy=np.asarray(model.images[image_id].point2D(observation.point2D_idx).xy)
            if not np.isfinite(xy).all(): continue
            vote=bool(box[0]<=xy[0]<=box[2] and box[1]<=xy[1]<=box[3])
            # Repeated SIFT detections in one image do not provide independent support.
            image_votes[image_id]=image_votes.get(image_id,True) and vote
        known=len(image_votes);inside=sum(image_votes.values())
        fraction=inside/known if known else 0.
        keep=quality_ok and known>=args.min_observations and fraction>=args.min_fraction
        qualities.append((keep,known,inside,fraction,error,distinct_views))
        if quality_ok and known>=args.min_observations:
            for threshold in thresholds:
                if fraction>=threshold: sensitivity[f'{threshold:.3f}']+=1
        if not quality_ok: rejections['track_or_error_limit']+=1
        elif known<args.min_observations: rejections['too_few_detected_observations']+=1
        elif not keep: rejections['outside_subject_rectangle_consensus']+=1
    xyz=np.asarray(xyz,dtype=float);colors=np.asarray(colors,dtype=np.uint8)
    keep=np.array([row[0] for row in qualities],dtype=bool)
    if keep.sum()<3: raise ValueError(f'Only {keep.sum()} points passed; no misleading preview written. Adjust thresholds after inspecting boxes/model.')
    candidate_indices=np.flatnonzero(keep)
    volume_keep,volume_info,volume_votes=volume_consensus(xyz[keep],model,image_boxes,args,np)
    observed_retained=int(keep.sum())
    keep[candidate_indices]=volume_keep
    rejections['outside_multi_view_subject_volume']=observed_retained-int(keep.sum())
    if keep.sum()<3: raise ValueError(f'Only {keep.sum()} points passed volume filtering; no preview written.')
    basis,orientation_info=upright_basis(model,image_boxes,np)
    accepted_ids={ids[i] for i in np.flatnonzero(keep)}
    centers=np.asarray([model.images[i].projection_center() for i in model.reg_image_ids()])
    stats={'dataset':str(dataset),'source_model':str(model_path),'boxes':str(boxes_path),'output':str(output),
           'registered_images':int(model.num_reg_images()),'registered_images_with_boxes':len(image_boxes),
           'source_points':int(model.num_points3D()),'retained_points':int(keep.sum()),'retained_fraction':float(keep.sum()/model.num_points3D()),
           'filter':{'min_detected_observations':args.min_observations,'min_inside_fraction':args.min_fraction,'min_track_length':args.min_track_length,'min_distinct_views':args.min_track_length,'observation_votes':'one per distinct image; all repeated detections must be inside its box','max_point_error_pixels':args.max_error,'min_detector_confidence':args.min_confidence},
           'rejection_counts':dict(rejections),'observed_track_filter_points':observed_retained,'volume_filter':volume_info,'display_orientation':orientation_info,'sensitivity_retained_counts':{f'{v:.3f}':sensitivity[f'{v:.3f}'] for v in thresholds},
           'source_point_bounds':coordinate_stats(xyz,np),'subject_point_bounds':coordinate_stats(xyz[keep],np),'camera_center_bounds':coordinate_stats(centers,np),
           'scale':'arbitrary SfM units; no metric scale reference supplied','coordinate_transform':'none; original model coordinates and every camera retained',
           'limitation':'Observed-track plus multi-view rectangle consensus approximates a subject volume; it is not segmentation. Background can remain and valid subject points may be excluded. Missing/out-of-frame detections contribute no vote.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.subject_preview_build_',dir=output.parent))
    try:
        (staging/'images').symlink_to(os.path.relpath(image_root,staging),target_is_directory=True)
        if (dataset/'camera_groups.json').is_file(): shutil.copy2(dataset/'camera_groups.json',staging/'camera_groups.json')
        sparse=staging/'colmap'/'sparse';(sparse/'0').mkdir(parents=True)
        db=pycolmap.Database.open(staging/'colmap'/'database.db');db.close()
        metadata=dataset/'colmap'/'sparse'/'sfm_inputs.json'
        if metadata.is_file(): stats['source_sfm_inputs']=json.loads(metadata.read_text())
        # A filtered preview is a finished model, with no refinement recipe of its own.
        # Fingerprint its linked inputs so the starter GUI opens it from the cache.
        extensions={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
        images=sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
        signature={'images':[[p.relative_to(image_root).as_posix(),p.stat().st_size,p.stat().st_mtime_ns] for p in images]}
        if (image_root.parent/'camera_groups.json').is_file(): signature['camera_mode']='PER_FOLDER'
        (sparse/'sfm_inputs.json').write_text(json.dumps(signature))
        write_ply(staging/'subject.ply',xyz[keep],colors[keep],np)
        stats['plot']=plot_points(staging/'subject_views.png',xyz[keep],colors[keep],dataset.name.replace('_',' ').title(),stats,args,np,basis)
        plot_sensitivity(staging/'volume_sensitivity.png',xyz[candidate_indices],colors[candidate_indices],volume_votes,volume_info,basis,np)
        np.savez_compressed(staging/'volume_votes.npz',point_ids=np.asarray(ids)[candidate_indices],**volume_votes)
        # Only the in-memory model is changed. Never write into model_path.
        for point_id in list(model.points3D.keys()):
            if int(point_id) not in accepted_ids: model.delete_point3D(point_id)
        model.write_binary(sparse/'0')
        stats['written_model_points']=int(model.num_points3D())
        (staging/'analysis.json').write_text(json.dumps(stats,indent=2))
        (staging/'README.md').write_text(
            f'# Subject-focused sparse preview\n\nSource: `{dataset}`\n\n'
            f'{stats["retained_points"]:,} of {stats["source_points"]:,} sparse points retained; '
            f'all {stats["registered_images"]} registered cameras preserved.\n\n'
            f'In the starter choose **File → Open existing result** and select this `{output.name}` folder itself. '
            'The `images` entry links to the original prepared photographs. '
            'This preview owns its database and filtered model; the full scene remains in the source dataset. '
            'Do not use Fit Colmap merely to view this result: that recomputes a full scene in this preview folder.\n\n'
            '`subject.ply` is a portable point cloud; `subject_views.png` shows equal-axis views. '
            '`analysis.json` reports filtering parameters and model bounds. `volume_sensitivity.png` compares 70%, 80%, and 90% projected rectangle consistency.\n\n'
            'This is an optional rectangle-filtered sparse visualization. Background can remain; no dense mesh was generated. '
            'Missing or out-of-image detections do not count against points. Model scale is arbitrary. PNGs are oriented upright for readability; PLY and COLMAP coordinates are preserved.\n')
        after={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in model_path.iterdir() if p.is_file()}
        if after!=source_files: raise RuntimeError('Source model changed during preview generation; wait for reconstruction to finish before retrying.')
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    print(json.dumps({'output':str(output),'retained_points':stats['retained_points'],'source_points':stats['source_points'],'registered_images':stats['registered_images'],'sensitivity':stats['sensitivity_retained_counts'],'volume_sensitivity':stats['volume_filter']['sensitivity']},indent=2))


if __name__=='__main__':
    main()
