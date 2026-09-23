from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'boxes.json').read_text())
for r in rows:
 r['detector']='macOS Vision VNDetectHumanRectanglesRequest (upperBodyOnly=false)'
 r['coordinate_system']='pixel coordinates; origin top-left; bbox [x_min,y_min,x_max,y_max]'
 if not r.get('selected'): continue
 b=r['selected']['bbox_xyxy'];bw=b[2]-b[0];bh=b[3]-b[1]
 if r['dataset']=='black_shirt_crutches':
  px=max(bw*.30,r['width']*.035);top=bh*.055;bottom=max(bh*.14,r['height']*.025)
 else: px=bw*.10;top=bottom=bh*.055
 r['padded_bbox_xyxy']=[max(0,b[0]-px),max(0,b[1]-top),min(r['width'],b[2]+px),min(r['height'],b[3]+bottom)]
 r['padding_note']='Generous rectangle for optional 3D point filtering; includes background and is not a pixel-accurate person mask.'
(ROOT/'boxes.json').write_text(json.dumps(rows,indent=2))
