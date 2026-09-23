#!/usr/bin/env python3
"""Prepare native SfM refinement inputs and body-aware matching graphs.

No extraction, matching, reconstruction, or application-code edits are performed.
The baseline reconstructions and baseline detection boxes stay untouched.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import json, itertools, math, os, shutil
import numpy as np
import pycolmap
from scipy.sparse import coo_matrix

SFM=Path(__file__).resolve().parents[2]
WORK=SFM/'work/quality-sfm'
NATIVE=WORK/'photos_native'
SUBJECTS=('light_shirt','black_shirt_crutches')


def summarize(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'count':0}
    return {'count':len(a),'min':float(a.min()),'p10':float(np.quantile(a,.1)),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max()),'mean':float(a.mean())}


def graph_components(n,pairs):
    neighbors=[set() for _ in range(n)]
    for i,j in pairs:neighbors[i].add(j);neighbors[j].add(i)
    unseen=set(range(n));components=[]
    while unseen:
        seed=unseen.pop();part={seed};queue=[seed]
        while queue:
            for other in neighbors[queue.pop()]&unseen:
                unseen.remove(other);part.add(other);queue.append(other)
        components.append(sorted(part))
    return components,neighbors


def budget_guided(candidates,n,angle,shared,groups,budget):
    """Keep at least two guided edges per image, then spend budget on diversity."""
    selected=set();degree=np.zeros(n,dtype=int);cross=np.zeros(n,dtype=int);bins=[set() for _ in range(n)]
    def angle_bin(i,j):
        value=angle[i,j]
        return next((k for k,(lo,hi) in enumerate(((3,10),(10,20),(20,40))) if lo<=value<hi),None)
    max_log=max(math.log1p(int(shared[i,j])) for i,j in candidates)
    def quality(i,j):
        moderate=3<=angle[i,j]<=25
        return math.log1p(int(shared[i,j]))/max_log+(0.4 if moderate else 0)
    def add(pair):
        selected.add(pair);i,j=pair;degree[i]+=1;degree[j]+=1
        if groups[i]!=groups[j]:cross[i]+=1;cross[j]+=1
        category=angle_bin(i,j)
        if category is not None:bins[i].add(category);bins[j].add(category)
    while np.any(degree<2):
        choices=[]
        for pair in candidates-selected:
            i,j=pair;gain=int(degree[i]<2)+int(degree[j]<2)
            if gain:
                category=angle_bin(i,j)
                new_bins=int(category is not None and category not in bins[i])+int(category is not None and category not in bins[j])
                new_cross=(int(cross[i]==0)+int(cross[j]==0)) if groups[i]!=groups[j] else 0
                choices.append(((gain,int(degree[i]==0)+int(degree[j]==0),new_cross+new_bins,quality(i,j),-max(degree[i],degree[j]),-i,-j),pair))
        if not choices:raise ValueError('Strict guided candidates cannot give every image two edges')
        add(max(choices)[1])
        if len(selected)>budget:raise ValueError('Guided budget insufficient for minimum degree2')
    while len(selected)<min(budget,len(candidates)):
        choices=[]
        for pair in candidates-selected:
            i,j=pair;category=angle_bin(i,j)
            new_bins=int(category is not None and category not in bins[i])+int(category is not None and category not in bins[j])
            new_cross=(int(cross[i]==0)+int(cross[j]==0)) if groups[i]!=groups[j] else 0
            value=4.0*new_cross+2.2*new_bins+quality(i,j)-.035*(degree[i]+degree[j])
            choices.append(((value,quality(i,j),-i,-j),pair))
        add(max(choices)[1])
    available_bins=[set() for _ in range(n)];available_cross=np.zeros(n,dtype=bool)
    for i,j in candidates:
        category=angle_bin(i,j)
        if category is not None:available_bins[i].add(category);available_bins[j].add(category)
        if groups[i]!=groups[j]:available_cross[i]=True;available_cross[j]=True
    audit={'budget':budget,'strict_candidates':len(candidates),'selected_pairs':len(selected),'minimum_guided_degree':int(degree.min()),
           'images_with_two_or_more_edges':int((degree>=2).sum()),'images_with_cross_group_edge':int((cross>0).sum()),'images_with_available_cross_group_edge':int(available_cross.sum()),
           'images_with_two_or_more_angle_bins':sum(len(v)>=2 for v in bins),'images_with_all_available_angle_bins':sum(bins[i]>=available_bins[i] for i in range(n)),
           'bin_coverage':{f'{lo}_{hi}deg':{'images_covered':sum(k in b for b in bins),'images_available':sum(k in b for b in available_bins)} for k,(lo,hi) in enumerate(((3,10),(10,20),(20,40)))},
           'moderate_3_25deg_pairs':int(sum(3<=angle[i,j]<=25 for i,j in selected)),
           'ordinary_matching_preserves_all_strict_candidates':True,
           'selection_policy':'First cover degree>=2 for every image, preferring mutual coverage, cross-group and angular diversity. Then add edges by missing cross-group/bin coverage and shared-body strength, with preference for3–25deg and gentle degree balancing. No images removed.'}
    return selected,audit


def prepare(subject):
    baseline=SFM/'reconstructions'/subject
    native=NATIVE/subject
    target=SFM/'reconstructions'/(subject+'_quality')
    target.mkdir(parents=True,exist_ok=True)
    if (target/'colmap').exists():raise RuntimeError(f'Will not alter a quality dataset already being processed: {target}')
    images=native/'images'
    if not (target/'images').exists():
        (target/'images').symlink_to(os.path.relpath(images,target),target_is_directory=True)
    elif (target/'images').resolve()!=images.resolve():raise RuntimeError('Existing images link points elsewhere')
    shutil.copy2(native/'camera_groups.json',target/'camera_groups.json')
    shutil.copy2(baseline/'input_manifest.json',target/'input_manifest.json')
    model_path=baseline/'subject_preview/colmap/sparse/0'
    model=pycolmap.Reconstruction(model_path)
    names=sorted(p.relative_to(images).as_posix() for p in images.rglob('*.jpg'))
    if set(names)!={im.name for im in model.images.values()}:raise ValueError('Baseline poses do not cover exactly every prepared input')
    n=len(names);name_index={name:i for i,name in enumerate(names)}
    image_by_name={im.name:im for im in model.images.values()};imageid_index={int(im.image_id):name_index[im.name] for im in model.images.values()}
    xyz=np.asarray([point.xyz for point in model.points3D.values()]);person_center=np.median(xyz,axis=0)
    centers=np.asarray([image_by_name[name].projection_center() for name in names])
    directions=centers-person_center;directions/=np.linalg.norm(directions,axis=1)[:,None]
    angle=np.degrees(np.arccos(np.clip(directions@directions.T,-1,1)));np.fill_diagonal(angle,np.inf)
    groups=np.asarray([str(Path(name).parent) for name in names]);group_names=sorted(set(groups))
    # Sparse incidence multiplication counts shared retained body tracks efficiently.
    track_rows=[];track_columns=[]
    for row,point in enumerate(model.points3D.values()):
        for column in {imageid_index[int(e.image_id)] for e in point.track.elements}:
            track_rows.append(row);track_columns.append(column)
    incidence=coo_matrix((np.ones(len(track_rows),dtype=np.int32),(track_rows,track_columns)),shape=(len(xyz),n)).tocsr()
    shared=(incidence.T@incidence).toarray();np.fill_diagonal(shared,0)
    # Feature-only fallback crops for missed detections; the detection status stays missing.
    boxes_path=native/'boxes.json';boxes=json.loads(boxes_path.read_text());inferred=[]
    for name in names:
        record=boxes[name]
        if record.get('status')=='detected':continue
        image=image_by_name[name];camera=pycolmap.Camera(model.cameras[image.camera_id].todict())
        camera.rescale(int(record['width']),int(record['height']))
        pose=image.cam_from_world();camera_xyz=xyz@pose.rotation.matrix().T+pose.translation
        good=np.isfinite(camera_xyz).all(axis=1)&(camera_xyz[:,2]>1e-8)
        uv=camera.img_from_cam(np.ascontiguousarray(camera_xyz[good]));uv=uv[np.isfinite(uv).all(axis=1)]
        if len(uv)<10:raise ValueError(f'Insufficient person projections for missing detection: {name}')
        low,high=np.quantile(uv,[.01,.99],axis=0);pad=(high-low)*.15
        low=np.maximum(low-pad,0);high=np.minimum(high+pad,[camera.width,camera.height])
        if np.any(high<=low):raise ValueError(f'Invalid projected fallback crop: {name}')
        crop=[float(low[0]),float(low[1]),float(high[0]),float(high[1])]
        record['feature_crop_bbox_xyxy']=crop
        inferred.append({'image':name,'status_preserved':record.get('status'),'feature_crop_bbox_xyxy':crop,'positive_finite_person_projections':len(uv),'method':'baseline subject points projected through native-scaled camera; coordinate quantiles1–99%, padding15%, image clamp'})
    boxes_path.write_text(json.dumps(boxes,indent=2))
    (NATIVE/(subject+'_boxes.json')).write_text(json.dumps(boxes,indent=2))
    # Undirected pairs are canonical index tuples. Reasons provide an audit trail.
    ordinary=set();reasons=defaultdict(set)
    def add(i,j,reason):
        if i==j:return
        pair=tuple(sorted((int(i),int(j))));ordinary.add(pair);reasons[pair].add(reason)
    for i in range(n):
        for j in np.argsort(angle[i])[:min(12,n-1)]:add(i,j,'nearest_angle_12')
        candidates=[j for j in range(n) if 2<=angle[i,j]<=65 and shared[i,j]>0]
        candidates.sort(key=lambda j:(-int(shared[i,j]),angle[i,j],names[j]))
        for j in candidates[:8]:add(i,j,'shared_body_top8_2_65deg')
        for group in group_names:
            if group==groups[i]:continue
            candidates=[j for j in range(n) if groups[j]==group and angle[i,j]<=65]
            candidates.sort(key=lambda j:(angle[i,j],-int(shared[i,j]),names[j]))
            for j in candidates[:4]:add(i,j,'other_group_nearest4_under65deg')
    manifest={r['image']:r for r in json.loads((baseline/'input_manifest.json').read_text())}
    for group in group_names:
        members=[i for i in range(n) if groups[i]==group]
        def time_key(i):
            r=manifest[names[i]]
            return (r.get('captured',''),r.get('time_seconds',0),Path(names[i]).name)
        members.sort(key=time_key)
        for k,i in enumerate(members):
            for j in members[max(0,k-2):min(len(members),k+3)]:add(i,j,'same_group_chronological_plusminus2')
    excluded_wide=[]
    for i,j in list(ordinary):
        if angle[i,j]>85:
            excluded_wide.append({'image1':names[i],'image2':names[j],'angle_degrees':float(angle[i,j]),'reasons':sorted(reasons[(i,j)])})
            ordinary.remove((i,j));del reasons[(i,j)]
    # Strict guided picks: best body overlap per angular bin and a distinct cross-group pick.
    guided=set();guided_reasons=defaultdict(set);per_image_guided={}
    bins=((3,10),(10,20),(20,40))
    for i in range(n):
        chosen=set();selection=[]
        for lo,hi in bins:
            candidates=[j for j in range(n) if lo<=angle[i,j]<hi and shared[i,j]>0]
            candidates.sort(key=lambda j:(-int(shared[i,j]),angle[i,j],names[j]))
            if candidates:
                j=candidates[0];chosen.add(j);pair=tuple(sorted((i,j)));guided.add(pair);guided_reasons[pair].add(f'bin_{lo}_{hi}');selection.append({'neighbor':names[j],'reason':f'best_shared_{lo}_{hi}deg'})
        candidates=[j for j in range(n) if j not in chosen and groups[j]!=groups[i] and 4<=angle[i,j]<=45 and shared[i,j]>0]
        candidates.sort(key=lambda j:(-int(shared[i,j]),angle[i,j],names[j]))
        if candidates:
            j=candidates[0];pair=tuple(sorted((i,j)));guided.add(pair);guided_reasons[pair].add('distinct_cross_group');selection.append({'neighbor':names[j],'reason':'best_shared_distinct_cross_group_4_45deg'})
        per_image_guided[names[i]]=selection
    strict_guided=set(guided)
    for i,j in strict_guided:add(i,j,'strict_guided_candidate')
    budget={'light_shirt':238,'black_shirt_crutches':552}[subject]
    guided,guided_audit=budget_guided(strict_guided,n,angle,shared,groups,budget)
    for i,j in guided:add(i,j,'guided_subset')
    components,neighbors=graph_components(n,ordinary)
    bridges=[]
    # Usually unnecessary; if a capture has an angular gap, connect components at closest directions.
    while len(components)>1:
        candidates=[]
        for a,b in itertools.combinations(components,2):
            block=angle[np.ix_(a,b)];x,y=np.unravel_index(np.argmin(block),block.shape)
            i,j=a[x],b[y];candidates.append((angle[i,j],-shared[i,j],i,j))
        _,_,i,j=min(candidates);add(i,j,'connectivity_bridge');bridges.append([names[i],names[j]])
        components,neighbors=graph_components(n,ordinary)
    assert len(components)==1 and all(neighbors) and guided<=ordinary
    def write_pairs(filename,pairs):
        (target/filename).write_text(''.join(f'{names[i]} {names[j]}\n' for i,j in sorted(pairs)))
    write_pairs('matching_pairs.txt',ordinary);write_pairs('guided_pairs.txt',guided);write_pairs('guided_candidates_strict.txt',strict_guided)
    guided_components,guided_neighbors=graph_components(n,guided)
    config={'baseline_dataset':'../'+subject,'boxes':'../../work/quality-sfm/photos_native/'+subject+'/boxes.json','matching_pairs':'matching_pairs.txt','guided_pairs':'guided_pairs.txt','guided_lock':'../../work/quality-sfm/guided.lock','max_features':18000}
    (target/'sfm_refine.json').write_text(json.dumps(config,indent=2))
    stats={'dataset':subject+'_quality','images':n,'source_person_points':len(xyz),'person_center':person_center.tolist(),'ordinary_pairs':len(ordinary),'guided_pairs':len(guided),'ordinary_components':len(components),'ordinary_all_images_have_edges':all(neighbors),'guided_components':len(guided_components),'guided_pairs_are_ordinary_subset':guided<=ordinary,'ordinary_degree':summarize([len(v) for v in neighbors]),'guided_degree':summarize([len(v) for v in guided_neighbors]),'ordinary_angles_degrees':summarize([angle[i,j] for i,j in ordinary]),'guided_angles_degrees':summarize([angle[i,j] for i,j in guided]),'ordinary_shared_body_points':summarize([shared[i,j] for i,j in ordinary]),'guided_shared_body_points':summarize([shared[i,j] for i,j in guided]),'ordinary_reason_counts':dict(Counter(reason for values in reasons.values() for reason in values)),'guided_reason_counts':dict(Counter(reason for pair,values in guided_reasons.items() if pair in guided for reason in values)),'guided_budget_audit':guided_audit,'connectivity_bridges':bridges,'ordinary_angle_cap_degrees':85,'excluded_pairs_above_angle_cap':excluded_wide,'angle_cap_exceptions':[{'image1':names[i],'image2':names[j],'angle_degrees':float(angle[i,j])} for i,j in ordinary if angle[i,j]>85],'inferred_feature_crop_count':len(inferred),'inferred_feature_crops':inferred,'group_image_counts':dict(Counter(groups)),'guided_selection':'Budgeted subset of strict best-shared candidates; see guided_budget_audit and guided_candidates_strict.txt','pair_policy':'12 nearest angular + top8 sharedbody2–65deg +4 nearest per other group within65deg + samegroup chronological±2 + guidedsubset','per_image':{names[i]:{'group':str(groups[i]),'ordinary_degree':len(neighbors[i]),'guided_degree':len(guided_neighbors[i]),'guided_strict_selections':per_image_guided[names[i]],'guided_selected_neighbors':[names[j] for j in sorted(guided_neighbors[i])]} for i in range(n)}}
    (target/'graph_statistics.json').write_text(json.dumps(stats,indent=2))
    (target/'guided_selection_audit.json').write_text(json.dumps({'summary':guided_audit,'strict_candidate_edges':[{'image1':names[i],'image2':names[j],'angle_degrees':float(angle[i,j]),'shared_body_points':int(shared[i,j]),'cross_group':bool(groups[i]!=groups[j]),'selected_for_guided':(i,j) in guided,'strict_reasons':sorted(guided_reasons[(i,j)])} for i,j in sorted(strict_guided)]},indent=2))
    # Small edge metadata table supports review without reloading native imagery.
    (target/'pair_metadata.json').write_text(json.dumps([{'image1':names[i],'image2':names[j],'angle_degrees':float(angle[i,j]),'shared_body_points':int(shared[i,j]),'guided':(i,j) in guided,'reasons':sorted(reasons[(i,j)])} for i,j in sorted(ordinary)],indent=2))
    return {k:stats[k] for k in ('dataset','images','ordinary_pairs','guided_pairs','ordinary_components','ordinary_all_images_have_edges','inferred_feature_crop_count')}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subjects',nargs='*',choices=SUBJECTS,default=list(SUBJECTS))
    args=parser.parse_args()
    results=[prepare(subject) for subject in args.subjects]
    print(json.dumps(results,indent=2))
    print('Total guided pairs',sum(r['guided_pairs'] for r in results))
