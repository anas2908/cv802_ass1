from pathlib import Path
import json, subprocess, imageio_ffmpeg, shutil
import numpy as np
from scipy.ndimage import laplace
from PIL import Image, ImageDraw
base=Path(__file__).resolve().parent
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
subjects={'IMG_6760':'black_shirt_crutches','IMG_8288':'black_shirt_crutches','IMG_8338':'light_shirt'}
manifest=[]
for stem,subject in subjects.items():
 out=base/'candidates'/stem;out.mkdir(parents=True,exist_ok=True)
 subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(base/'sources'/(stem+'.MOV')),'-vf','fps=6','-q:v','2','-y',str(out/'candidate_%05d.jpg')],check=True)
 frames=sorted(out.glob('*.jpg'));scored=[]
 for i,p in enumerate(frames):
  im=Image.open(p).convert('L'); w,h=im.size
  arr=np.asarray(im.crop((int(w*.2),int(h*.08),int(w*.8),int(h*.93))).resize((384,960)),dtype=np.float32)
  sharpness=float(laplace(arr).var())
  scored.append({'file':p,'time_seconds':(i+.5)/6,'sharpness':sharpness})
 selected=[]
 dest=base/'frames'/subject/stem;dest.mkdir(parents=True,exist_ok=True)
 for b in range(0,len(scored),3):
  best=max(scored[b:b+3],key=lambda x:x['sharpness'])
  op=dest/f'{stem}_t{round(best["time_seconds"]*1000):06d}.jpg';shutil.copy2(best['file'],op)
  entry={'path':str(op),'source':stem+'.MOV','subject':subject,'time_seconds':best['time_seconds'],'sharpness':best['sharpness'],'width':1080,'height':1920,'method':'Sharpest central-crop Laplacian frame from each half-second, sampled at 6 fps; no upscale.'}
  manifest.append(entry);selected.append(entry)
 tw,th=160,315;cols=8;rows=(len(selected)+cols-1)//cols
 sheet=Image.new('RGB',(tw*cols,th*rows),(240,240,240));d=ImageDraw.Draw(sheet)
 for i,e in enumerate(selected):
  im=Image.open(e['path']);im.thumbnail((150,270));x=i%cols*tw;y=i//cols*th
  sheet.paste(im,(x+(tw-im.width)//2,y+40));d.text((x+3,y+3),f'{e["time_seconds"]:.2f}s',fill='black');d.text((x+3,y+19),f'sharp {e["sharpness"]:.1f}',fill='black')
 sheet.save(base/(stem+'_frames.jpg'))
 print(stem,len(selected),'sharpness min/median/max',min(e['sharpness'] for e in selected),np.median([e['sharpness'] for e in selected]),max(e['sharpness'] for e in selected),flush=True)
(base/'frame_manifest.json').write_text(json.dumps(manifest,indent=2))
