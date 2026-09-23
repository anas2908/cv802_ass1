from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'pilot_results.json').read_text())['images'];summary=[]
for number,r in enumerate(rows):
 if r['status']!='generated':summary.append({'image':r['image_name'],'status':r['status']});continue
 with Image.open(r['source']) as image:
  width,height=image.size;source=image.convert('RGB');source.thumbnail((340,530),Image.Resampling.LANCZOS)
 with Image.open(r['raw_mask']) as raw:
  native=raw.convert('L').resize((width,height),Image.Resampling.BILINEAR);target=Path(r['mask']);target.parent.mkdir(parents=True,exist_ok=True);native.save(target,'PNG',compress_level=4)
  person=np.asarray(native.resize(source.size,Image.Resampling.BILINEAR),dtype=float)/255
 with Image.open(r['foreground_all']) as fg:
  assert fg.size==(width,height)
  foreground=np.asarray(fg.resize(source.size,Image.Resampling.BILINEAR),dtype=float)/255
 rgb=np.asarray(source,dtype=float)
 def overlay(mask,color):
  alpha=mask[:,:,None]*.6;return Image.fromarray(np.uint8(rgb*(1-alpha)+np.asarray(color)[None,None,:]*alpha))
 person_overlay=overlay(person,[0,235,100]);fg_overlay=overlay(foreground,[0,180,255]);fg_image=Image.fromarray(np.uint8(foreground*255)).convert('RGB')
 sheet=Image.new('RGB',(4*350,source.height+75),'white');d=ImageDraw.Draw(sheet)
 for col,(im,label) in enumerate([(source,'Original'),(person_overlay,'Person-only matte (green)'),(fg_overlay,'Foreground instance matte (blue)'),(fg_image,'Foreground matte')]):
  sheet.paste(im,(col*350,25));d.text((col*350+3,5),label,fill='black')
 d.text((3,source.height+32),r['image_name'],fill='black');d.text((3,source.height+49),f"Foreground instances: {r['instance_ids']}; native{width}x{height}; no instance selection or geometric cleanup yet",fill='black')
 sheet.save(ROOT/f'pilot_overlay_{number:02d}.jpg',quality=95)
 summary.append({'image':r['image_name'],'status':'generated','instance_ids':r['instance_ids'],'width':width,'height':height,
                 'thumbnail_person_foreground_fraction':float((person>=.5).mean()),'thumbnail_instance_foreground_fraction':float((foreground>=.5).mean()),
                 'instance_only_fraction':float(((foreground>=.5)&(person<.5)).mean()),'person_only_fraction':float(((person>=.5)&(foreground<.5)).mean())})
(ROOT/'pilot_qc_metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
