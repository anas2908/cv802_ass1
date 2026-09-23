from pathlib import Path
from PIL import Image,ImageDraw
from collections import Counter
import json,math
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'boxes.json').read_text())
summary={}
for dataset in sorted(set(r['dataset'] for r in rows)):
 batch=[r for r in rows if r['dataset']==dataset]
 summary[dataset]={'total':len(batch),'statuses':dict(Counter(r['status'] for r in batch)),'ambiguous':[r['image_name'] for r in batch if r.get('ambiguity')],'low_confidence':[r['image_name'] for r in batch if r.get('selected',{}).get('confidence',0)<0.5]}
 (ROOT/(dataset+'_boxes.json')).write_text(json.dumps({r['image_name']:r for r in batch},indent=2))
 selected=[]
 # Cover every camera group evenly and show every missing or ambiguous case.
 groups=sorted(set(str(Path(r['image_name']).parent) for r in batch))
 for group in groups:
  gb=[r for r in batch if str(Path(r['image_name']).parent)==group]
  step=max(1,len(gb)//10)
  selected.extend(gb[::step])
 selected += [r for r in batch if r.get('ambiguity') or r.get('status')!='detected']
 selected=list({r['image_name']:r for r in selected}.values())
 for start in range(0,len(selected),40):
  piece=selected[start:start+40];cols=8;tw,th=190,278
  sheet=Image.new('RGB',(cols*tw,math.ceil(len(piece)/cols)*th),'#222222');sd=ImageDraw.Draw(sheet)
  for i,r in enumerate(piece):
   with Image.open(r['path']) as im:
    im=im.convert('RGB');im.thumbnail((tw-4,th-52)); d=ImageDraw.Draw(im); sx,sy=im.width/r['width'],im.height/r['height']
    if r.get('selected'):
     bb=r['selected']['bbox_xyxy'];d.rectangle([bb[0]*sx,bb[1]*sy,bb[2]*sx,bb[3]*sy],outline='#00ff77',width=2)
     bb=r['padded_bbox_xyxy'];d.rectangle([bb[0]*sx,bb[1]*sy,bb[2]*sx,bb[3]*sy],outline='#ffff00',width=2)
    x=i%cols*tw;y=i//cols*th;sheet.paste(im,(x+(tw-im.width)//2,y))
    sd.text((x+3,y+th-49),Path(r['image_name']).name,fill='white')
    sd.text((x+3,y+th-35),str(Path(r['image_name']).parent)[-29:],fill='#dddddd')
    sd.text((x+3,y+th-20),f"{r.get('selected',{}).get('confidence',0):.2f} {r['status']}"+(' AMBIGUOUS' if r.get('ambiguity') else ''),fill='#ffaa77')
  sheet.save(ROOT/(dataset+f'_boxes_{start:03d}.jpg'),quality=92)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
