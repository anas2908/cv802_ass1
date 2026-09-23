#!/usr/bin/env python3
"""Resize local Vision mattes to native inputs and report explicit mask QC."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
SFM=ROOT.parents[2]
INPUT=json.loads((ROOT/'input_manifest.json').read_text())
BOXES=json.loads((SFM/'work/quality-sfm/photos_native/light_shirt/boxes.json').read_text())
records={}
for row in INPUT:
    name=row['image_name'];mask_path=Path(row['mask']);source=Path(row['source'])
    record={'source_path':str(source),'source_resolved_path':str(source.resolve()),'source_size_bytes':source.stat().st_size,'source_mtime_ns':source.stat().st_mtime_ns,
            'mask_path':str(mask_path),'mask_relative_path':mask_path.relative_to(ROOT).as_posix(),'status':'pending','usable':False}
    raw_meta_path=Path(row['raw_metadata'])
    if not raw_meta_path.exists():records[name]=record;continue
    raw_meta=json.loads(raw_meta_path.read_text())
    if raw_meta.get('status')!='generated':
        record.update(status='error',error=raw_meta.get('error','Unknown Vision failure'));records[name]=record;continue
    width,height=raw_meta['source_width'],raw_meta['source_height']
    record.update(width=width,height=height,raw_mask_width=raw_meta['raw_mask_width'],raw_mask_height=raw_meta['raw_mask_height'],raw_metadata_path=str(raw_meta_path))
    with Image.open(row['raw_mask']) as raw:
        if raw.mode!='L':raise ValueError(f'Unexpected mask representation: {name} {raw.mode}')
        native=raw.resize((width,height),Image.Resampling.BILINEAR)
        mask_path.parent.mkdir(parents=True,exist_ok=True)
        native.save(mask_path,'PNG',compress_level=4)
        record['mask_sha256']=hashlib.sha256(mask_path.read_bytes()).hexdigest()
        # QC runs on a bounded thumbnail; full-resolution matte is preserved unchanged.
        small=native.copy();small.thumbnail((640,640),Image.Resampling.BILINEAR)
        values=np.asarray(small,dtype=np.uint8)
    foreground=values>=128
    fraction=float(foreground.mean())
    labels,components=ndimage.label(foreground)
    areas=np.bincount(labels.ravel())[1:]
    largest=float(areas.max()/max(1,foreground.sum())) if len(areas) else 0.
    ys,xs=np.nonzero(foreground)
    mask_bbox=[float(xs.min()/values.shape[1]*width),float(ys.min()/values.shape[0]*height),float((xs.max()+1)/values.shape[1]*width),float((ys.max()+1)/values.shape[0]*height)] if len(xs) else None
    box_record=BOXES[name];box=box_record.get('padded_bbox_xyxy',box_record.get('feature_crop_bbox_xyxy'))
    outside_fraction=None;roi_occupancy=None
    if box:
        sx,sy=values.shape[1]/width,values.shape[0]/height
        x0,y0=max(0,int(box[0]*sx)),max(0,int(box[1]*sy));x1,y1=min(values.shape[1],math.ceil(box[2]*sx)),min(values.shape[0],math.ceil(box[3]*sy))
        inside=int(foreground[y0:y1,x0:x1].sum());outside_fraction=float(1-inside/max(1,foreground.sum()));roi_occupancy=float(inside/max(1,(x1-x0)*(y1-y0)))
    flags=[]
    if fraction<.002:flags.append('insufficient_foreground')
    if fraction>.95:flags.append('nearly_all_foreground')
    if largest<.55:flags.append('fragmented_foreground')
    if outside_fraction is not None and outside_fraction>.25:flags.append('foreground_mostly_outside_expected_subject_box')
    if roi_occupancy is not None and roi_occupancy<.12:flags.append('low_expected_roi_occupancy')
    record.update(status='needs_review' if flags else 'valid',usable=not flags,
                  qc={'flags':flags,'threshold_128_foreground_fraction':fraction,'foreground_fraction_threshold64':float((values>=64).mean()),'foreground_fraction_threshold192':float((values>=192).mean()),
                      'minimum_value':int(values.min()),'maximum_value':int(values.max()),'mean_value':float(values.mean()),'connected_components':int(components),'largest_component_foreground_fraction':largest,
                      'foreground_bbox_xyxy':mask_bbox,'expected_subject_bbox_xyxy':box,'outside_expected_box_foreground_fraction':outside_fraction,'expected_roi_occupancy':roi_occupancy,'qc_thumbnail_size':[values.shape[1],values.shape[0]]})
    records[name]=record
counts=Counter(r['status'] for r in records.values());complete=counts['pending']==0
manifest={'schema_version':1,'source_dataset':str((SFM/'reconstructions/light_shirt_quality').resolve()),'complete':complete,'ready_for_filter':complete and counts['valid']>=3,
          'method':{'detector':'Apple Vision VNGeneratePersonSegmentationRequest','quality':'accurate','request_revision':1,'independent_request_per_image':True,
                    'input_orientation':'All source JPEGs physically upright with EXIF1; Vision handler orientation .up','raw_mask_format':'OneComponent8 grayscale; PNG modeL uint8',
                    'native_resize':'Pillow bilinear from Vision mask dimensions to source image width/height; no threshold during resize','value_semantics':'0–255 foreground matte/confidence-like values; normalize by255. Not binary and not calibrated probabilities.',
                    'pixel_coordinates':'native source dimensions, top-left origin; no crop, rotation, mirror, or pad','source_pixels_modified':False,
                    'docs':['https://developer.apple.com/documentation/Vision/VNGeneratePersonSegmentationRequest','https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest/outputpixelformat']},
          'summary':{'total':len(records),'statuses':dict(counts),'usable':counts['valid'],'failed_or_review_names':[n for n,r in records.items() if r['status'] in ('error','needs_review')]},'images':records}
(ROOT/'mask_manifest.json').write_text(json.dumps(manifest,indent=2))
# Representative original/matte/overlay triplets are analytical QC, not altered inputs.
available=[row for row in INPUT if records[row['image_name']]['status'] in ('valid','needs_review')]
priority=[r['image_name'] for r in json.loads((ROOT/'pilot_manifest.json').read_text())]
chosen=[r for name in priority for r in available if r['image_name']==name]
if complete:
    chosen += available[::max(1,len(available)//18)]
    chosen += [r for r in available if records[r['image_name']]['status']=='needs_review']
chosen=list({r['image_name']:r for r in chosen}.values())
for start in range(0,len(chosen),12):
    batch=chosen[start:start+12];tw,th=525,310;sheet=Image.new('RGB',(2*tw,math.ceil(len(batch)/2)*th),'white');draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(batch):
        x=index%2*tw;y=index//2*th
        with Image.open(row['source']) as image:
            original=image.convert('RGB');original.thumbnail((168,252),Image.Resampling.LANCZOS)
        with Image.open(row['mask']) as image:mask=image.convert('L').resize(original.size,Image.Resampling.BILINEAR)
        rgb=np.asarray(original,dtype=float);alpha=np.asarray(mask,dtype=float)/255*.45
        green=np.zeros_like(rgb);green[:]=[0,230,90]
        overlay=Image.fromarray(np.uint8(rgb*(1-alpha[:,:,None])+green*alpha[:,:,None]))
        for column,(visual,label) in enumerate(((original,'Original'),(mask.convert('RGB'),'Vision matte'),(overlay,'Matte overlay'))):
            xx=x+column*175;sheet.paste(visual,(xx,y+18));draw.text((xx+3,y+2),label,fill='black')
        record=records[row['image_name']]
        draw.text((x+3,y+272),Path(row['image_name']).name,fill='black');draw.text((x+3,y+288),record['status']+'; foreground '+f"{record['qc']['threshold_128_foreground_fraction']:.1%}",fill='black')
    sheet.save(ROOT/f'mask_qc_{start:03d}.jpg',quality=94)
print(json.dumps(manifest['summary'],indent=2))
