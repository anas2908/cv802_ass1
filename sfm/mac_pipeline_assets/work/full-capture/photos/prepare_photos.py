"""Prepare read-only HEIC archive stills for SfM; preserve camera EXIF and bake orientation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, math, subprocess, zipfile
from PIL import Image, ImageOps, ImageDraw, ExifTags
ROOT=Path(__file__).resolve().parent
for d in ('heic','jpeg','contact_sheets','temporary'): (ROOT/d).mkdir(exist_ok=True)
ARCHIVE=ROOT.parents[3]/'images_videos'/'Photos-1-001.zip'
with zipfile.ZipFile(ARCHIVE) as archive:
 for member in archive.namelist():
  if member.lower().endswith('.heic'):
   destination=ROOT/'heic'/Path(member).name
   if not destination.exists(): destination.write_bytes(archive.read(member))
files=sorted((ROOT/'heic').glob('*.HEIC'))
def convert(src):
 dst=ROOT/'jpeg'/src.with_suffix('.jpg').name
 tmp=ROOT/'temporary'/dst.name
 subprocess.run(['/usr/bin/sips','-s','format','jpeg','-s','formatOptions','100','-Z','3200',str(src),'--out',str(tmp)],check=True,capture_output=True)
 with Image.open(tmp) as im:
  original=im.getexif(); exif_original=original.get_ifd(34665)
  upright=ImageOps.exif_transpose(im).convert('RGB')
  exif=Image.Exif()
  for k in (271,272,305,306):
   if k in original: exif[k]=original[k]
  exif[274]=1
  ed={k:v for k,v in exif_original.items() if k in (36867,36868,36880,36881,36882,37521,37522,33434,33437,34855,37386,41989,42034,42035,42036)}
  ed[40962],ed[40963]=upright.size
  exif[34665]=ed
  upright.save(dst,'JPEG',quality=95,subsampling=0,exif=exif)
  meta={'filename':src.name,'jpeg':str(dst),'width':upright.width,'height':upright.height,'original_orientation':original.get(274),'camera_make':original.get(271),'camera_model':original.get(272),'captured':exif_original.get(36867),'timezone':exif_original.get(36881),'focal_length_mm':float(exif_original.get(37386,0)),'focal_length_35mm':exif_original.get(41989),'lens_model':exif_original.get(42036),'exposure_seconds':float(exif_original.get(33434,0)),'iso':exif_original.get(34855)}
 tmp.unlink()
 return meta
with ThreadPoolExecutor(max_workers=4) as pool:
 results=[]
 for i,m in enumerate(pool.map(convert,files),1):
  results.append(m)
  if i%25==0: print(f'Prepared {i}/{len(files)} photos',flush=True)
(ROOT/'metadata.json').write_text(json.dumps(results,indent=2))
for start in range(0,len(results),48):
 batch=results[start:start+48]; cols=8; tw,th=190,230
 sheet=Image.new('RGB',(cols*tw,math.ceil(len(batch)/cols)*th),'#222222'); draw=ImageDraw.Draw(sheet)
 for n,record in enumerate(batch):
  with Image.open(record['jpeg']) as im:
   im.thumbnail((tw-8,th-34))
   x=(n%cols)*tw; y=(n//cols)*th
   sheet.paste(im,(x+(tw-im.width)//2,y))
   draw.text((x+5,y+th-32),record['filename'],fill='white')
   draw.text((x+5,y+th-18),record['captured'][11:] if record['captured'] else '',fill='#cccccc')
 sheet.save(ROOT/'contact_sheets'/f'stills_{start+1:03d}_{start+len(batch):03d}.jpg',quality=90)
print('Complete',len(results),flush=True)
