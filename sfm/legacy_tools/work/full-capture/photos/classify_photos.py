from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, Counter
from PIL import Image
import json, hashlib
ROOT=Path(__file__).resolve().parent
records=json.loads((ROOT/'metadata.json').read_text())
def examine(r):
 n=int(Path(r['filename']).stem.split('_')[1])
 r['subject']='black_shirt_crutches' if (6653<=n<=6759 or 8212<=n<=8287 or 8391<=n<=8488) else 'light_shirt'
 r['capture_group']=r['subject']+'_'+r['camera_model'].replace(' ','_').lower()
 r['lens_group']=r['capture_group']+'_'+str(r['focal_length_35mm'])+'mm'
 with Image.open(r['jpeg']) as im:
  exif=im.getexif().get_ifd(34665)
  r['subsecond']=str(exif.get(37521,'0'))
  r['pixel_sha256']=hashlib.sha256(im.tobytes()).hexdigest()
 r['source_sha256']=hashlib.sha256((ROOT/'heic'/r['filename']).read_bytes()).hexdigest()
 r['exclude_recommended']=r['filename'] in ('IMG_6761.HEIC','IMG_6762.HEIC')
 r['notes']='Hands initially on hips or moving into arms-crossed pose; later light-shirt photos have arms crossed.' if r['exclude_recommended'] else ''
 return r
with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(examine,records))
(ROOT/'metadata.json').write_text(json.dumps(records,indent=2))
dups=defaultdict(list)
for r in records: dups[r['pixel_sha256']].append(r['filename'])
dups=[v for v in dups.values() if len(v)>1]
report={'total_photos':len(records),'subject_counts':dict(Counter(r['subject'] for r in records)),'capture_group_counts':dict(Counter(r['capture_group'] for r in records)),'lens_group_counts':dict(Counter(r['lens_group'] for r in records)),'exact_pixel_duplicate_groups':dups,'groups':{}}
for group in sorted(set(r['subject'] for r in records)):
 batch=sorted((r for r in records if r['subject']==group),key=lambda r:(r['captured'],r['subsecond'],r['filename']))
 report['groups'][group]={'filenames':[r['filename'] for r in batch],'jpeg_paths':[r['jpeg'] for r in batch],'recommended_exclusions':[r['filename'] for r in batch if r['exclude_recommended']]}
 (ROOT/(group+'_images.txt')).write_text('\n'.join(r['jpeg'] for r in batch)+'\n')
(ROOT/'classification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
