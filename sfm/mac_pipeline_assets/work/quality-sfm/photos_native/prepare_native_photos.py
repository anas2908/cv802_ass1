"""Build full-resolution copies of selected photos and unchanged frame links for SfM."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy,json,os,re,shutil,subprocess
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
SFM=ROOT.parents[2]
ORIGINALS=SFM/'work/full-capture/photos/heic'
META={r['filename']:r for r in json.loads((SFM/'work/full-capture/photos/metadata.json').read_text())}
rows=[];source_files=[]
for name in ('black_shirt_crutches','light_shirt'):
 source_root=SFM/'reconstructions'/name/'images'
 for src in sorted(source_root.rglob('*.jpg')):
  relative=src.relative_to(source_root)
  is_photo=relative.parts[0].startswith('photos_')
  heic=ORIGINALS/(re.search(r'IMG_\d+',src.stem).group()+'.HEIC') if is_photo else None
  if is_photo:
   if not heic.is_file(): raise FileNotFoundError(heic)
   source_files.append(heic)
  with Image.open(src) as image: old_size=image.size
  rows.append({'dataset':name,'image_name':relative.as_posix(),'kind':'photo' if is_photo else 'video_frame','baseline_image':str(src),'source_heic':str(heic) if heic else None,'output_image':str(ROOT/name/'images'/relative),'baseline_width':old_size[0],'baseline_height':old_size[1]})
# Inspect native dimensions before conversion; all originals remain untouched.
result=subprocess.run(['/usr/bin/sips','-g','pixelWidth','-g','pixelHeight',*[str(p) for p in source_files]],check=True,text=True,capture_output=True)
native={};current=None
for line in result.stdout.splitlines():
 if line.startswith('/'):
  current=line.strip();native[current]={}
 elif current and ':' in line:
  key,value=line.strip().split(':',1)
  if key in ('pixelWidth','pixelHeight'):native[current][key]=int(value.strip())
for row in rows:
 if row['kind']=='photo':
  src_meta=META[Path(row['source_heic']).name];dim=native[row['source_heic']]
  w,h=dim['pixelWidth'],dim['pixelHeight']
  row['native_encoded_width'],row['native_encoded_height']=w,h
  row['source_orientation']=src_meta['original_orientation']
  row['camera_model']=src_meta['camera_model'];row['focal_length_35mm']=src_meta['focal_length_35mm'];row['captured']=src_meta['captured']
  if row['source_orientation'] in (5,6,7,8):w,h=h,w
 else:w,h=row['baseline_width'],row['baseline_height']
 row['width'],row['height']=w,h
 row['scale_x'],row['scale_y']=w/row['baseline_width'],h/row['baseline_height']
(ROOT/'catalog.json').write_text(json.dumps(rows,indent=2))
print('Catalog:',dict(Counter(r['kind'] for r in rows)),'native encoded dimensions:',dict(Counter((r.get('native_encoded_width'),r.get('native_encoded_height')) for r in rows if r['kind']=='photo')),flush=True)
def prepare(row):
 dst=Path(row['output_image']);dst.parent.mkdir(parents=True,exist_ok=True)
 if row['kind']=='video_frame':
  if not dst.exists():dst.symlink_to(os.path.relpath(row['baseline_image'],dst.parent))
  return
 tmp=ROOT/'.temporary'/row['dataset']/Path(row['image_name'])
 tmp.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['/usr/bin/sips','-s','format','jpeg','-s','formatOptions','100',row['source_heic'],'--out',str(tmp)],check=True,capture_output=True)
 with Image.open(tmp) as image:
  original=image.getexif();original_ifd=original.get_ifd(34665)
  upright=ImageOps.exif_transpose(image).convert('RGB')
  if upright.size!=(row['width'],row['height']):raise ValueError(f'Unexpected upright size: {dst} {upright.size}')
  exif=Image.Exif()
  for key in (271,272,305,306):
   if key in original:exif[key]=original[key]
  exif[274]=1
  ed={k:v for k,v in original_ifd.items() if k in (36867,36868,36880,36881,36882,37521,37522,33434,33437,34855,37386,41989,42034,42035,42036)}
  ed[40962],ed[40963]=upright.size;exif[34665]=ed
  upright.save(dst,'JPEG',quality=95,subsampling=0,exif=exif)
 tmp.unlink()
with ThreadPoolExecutor(max_workers=2) as pool:
 for i,_ in enumerate(pool.map(prepare,rows),1):
  if i%25==0: print(f'Prepared {i}/{len(rows)} native/frame inputs',flush=True)
for name in ('black_shirt_crutches','light_shirt'):
 target=ROOT/name
 groups=json.loads((SFM/'reconstructions'/name/'camera_groups.json').read_text())
 for group,record in groups.items():
  sample=next(r for r in rows if r['dataset']==name and r['image_name'].split('/')[0]==group)
  record['width'],record['height']=sample['width'],sample['height']
 (target/'camera_groups.json').write_text(json.dumps(groups,indent=2))
 boxes=json.loads((SFM/'work/full-capture/subject_boxes'/(name+'_boxes.json')).read_text())
 transformed={}
 for row in (r for r in rows if r['dataset']==name):
  record=copy.deepcopy(boxes[row['image_name']]);sx,sy=row['scale_x'],row['scale_y']
  record['baseline_path']=record['path'];record['path']=row['output_image'];record['width'],record['height']=row['width'],row['height']
  for candidate in record.get('candidates',[]):
   if 'bbox_xyxy' in candidate:candidate['bbox_xyxy']=[v*([sx,sy,sx,sy][i]) for i,v in enumerate(candidate['bbox_xyxy'])]
  if record.get('selected'):
   record['selected']['bbox_xyxy']=[v*([sx,sy,sx,sy][i]) for i,v in enumerate(record['selected']['bbox_xyxy'])]
  if 'padded_bbox_xyxy' in record:record['padded_bbox_xyxy']=[v*([sx,sy,sx,sy][i]) for i,v in enumerate(record['padded_bbox_xyxy'])]
  record['native_scaling']={'scale_x':sx,'scale_y':sy,'source_width':row['baseline_width'],'source_height':row['baseline_height'],'method':'Scaled existing reviewed macOS Vision rectangles; no new detector inference.'}
  transformed[row['image_name']]=record
 (target/'boxes.json').write_text(json.dumps(transformed,indent=2))
 (ROOT/(name+'_boxes.json')).write_text(json.dumps(transformed,indent=2))
print('Native preparation complete',flush=True)
