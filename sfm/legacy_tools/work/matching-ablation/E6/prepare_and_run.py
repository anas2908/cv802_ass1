#!/usr/bin/env python3
"""E6: reuse E3 features and settings, changing only guided matching to off.

Every output is new. Old images, features, models, configuration and code are read only.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3]
WORK=Path(__file__).resolve().parent
OUT=ROOT/'reconstructions/experiments/E6_guided_off'
SUBJECTS=('light_shirt','black_shirt_crutches')
PYTHON=ROOT/'.venv/bin/python'
sys.path.insert(0,str(ROOT/'work/quality-sfm'))
from audit_database import _select_checkpoint
import pycolmap

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def run(stage,subject,command):
 log=WORK/f'{subject}_{stage}.log'; started=time.monotonic()
 print(f'{stamp()} START {subject} {stage}',flush=True)
 with log.open('x') as stream:
  process=subprocess.Popen([str(PYTHON),*map(str,command)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  write(WORK/'status.json',{'subject':subject,'stage':stage,'pid':process.pid,'log':str(log),'started_utc':stamp()})
  code=process.wait()
 if code:raise RuntimeError(f'{subject} {stage} failed with exit{code}; see{log}')
 result={'stage':stage,'elapsed_seconds':time.monotonic()-started,'log':str(log),'exit_code':code}
 print(f'{stamp()} DONE {subject} {stage} {result["elapsed_seconds"]:.2f}s',flush=True)
 return result

def prepare():
 for subject in SUBJECTS:
  if (OUT/f'{subject}_quality').exists():raise FileExistsError(f'Preserve existing dataset:{OUT/subject}')
 prepared={}
 for subject in SUBJECTS:
  source=ROOT/'reconstructions'/f'{subject}_quality'; dataset=OUT/f'{subject}_quality'
  original=json.loads((source/'sfm_refine.json').read_text());config=dict(original)
  for key in ('baseline_dataset','boxes','guided_lock'):
   config[key]=os.path.relpath((source/original[key]).resolve(),dataset)
  config['matching_pairs']='matching_pairs.txt';config['guided_pairs']='guided_pairs.txt'
  baseline=(dataset/config['baseline_dataset']).resolve();boxes=(dataset/config['boxes']).resolve()
  assert baseline==ROOT/'reconstructions'/subject
  assert boxes==(source/original['boxes']).resolve()
  metadata,key_record,key,ignored=_select_checkpoint(source,original,json.loads((source/'colmap/sparse/sfm_inputs.json').read_text()))
  source_db=metadata.with_suffix('.db'); signature=key_record['signature']
  image_root=(source/'images').resolve()
  images=sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
  assert signature['images']==[[p.relative_to(image_root).as_posix(),p.stat().st_size,p.stat().st_mtime_ns] for p in images]
  baseline_files=[baseline/'colmap/database.db']+sorted(p for p in (baseline/'colmap/sparse').rglob('*') if p.is_file())
  assert signature['baseline']==[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in baseline_files]
  assert signature['boxes_sha256']==sha(boxes) and signature['pycolmap']==pycolmap.__version__
  assert signature['max_features']==config['max_features']==18000
  assert key==hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
  assert source_db.stat().st_size==key_record['database_size']
  cache=dataset/'colmap/quality_feature_cache';cache.mkdir(parents=True)
  (dataset/'images').symlink_to(os.path.relpath(image_root,dataset),target_is_directory=True)
  for filename in ('camera_groups.json','input_manifest.json','matching_pairs.txt'):
   shutil.copy2(source/filename,dataset/filename)
  (dataset/'guided_pairs.txt').write_text('')
  write(dataset/'sfm_refine.json',config)
  for path in (metadata,source_db):
   shutil.copy2(path,cache/path.name)
   assert sha(path)==sha(cache/path.name)
  assert sha(source/'matching_pairs.txt')==sha(dataset/'matching_pairs.txt')
  assert (dataset/'images').resolve()==image_root
  watched=[source/'sfm_refine.json',source/'matching_pairs.txt',source/'guided_pairs.txt',source_db,metadata,boxes,*baseline_files,
           *(p for p in (source/'colmap/sparse').rglob('*') if p.is_file()),
           ROOT/'assignment1/modules/colmap/api.py',ROOT/'work/quality-sfm/run_quality.py',ROOT/'work/full-capture/make_subject_preview.py']
  provenance={'experiment':'E6_guided_off','subject':subject,'prepared_utc':stamp(),'source_E3_dataset':str(source),'new_dataset':str(dataset),
   'algorithmic_change':'Only guided matching disabled by empty guided_pairs.txt; ordinary matching is recomputed from identical extracted features.',
   'unchanged':['native image files and order','feature bytes and checkpoint signature','baseline world poses','camera models and resolution scaling','feature crops/selection/options','ordinary pair list and CPU4 matching','fixed-pose triangulation parameters'],
   'image_count':len(images),'ordinary_pairs':sum(bool(x.strip()) and not x.lstrip().startswith('#') for x in (dataset/'matching_pairs.txt').read_text().splitlines()),
   'source_guided_pairs':sum(bool(x.strip()) and not x.lstrip().startswith('#') for x in (source/'guided_pairs.txt').read_text().splitlines()),'guided_pairs':0,
   'source_feature_checkpoint':str(source_db),'copied_feature_checkpoint':str(cache/source_db.name),'feature_checkpoint_key':key,
   'feature_checkpoint_sha256':sha(source_db),'feature_metadata_sha256':sha(metadata),'matching_pairs_sha256':sha(dataset/'matching_pairs.txt'),
   'guided_pairs_sha256':sha(dataset/'guided_pairs.txt'),'config':config,'resolved_baseline':str(baseline),'resolved_boxes':str(boxes),'resolved_images':str(image_root),
   'old_files_sha256':{str(p):sha(p) for p in dict.fromkeys(watched)}}
  write(dataset/'experiment_provenance.json',provenance);prepared[subject]=provenance
  print(f'PREPARED {subject}: {len(images)} images, {provenance["ordinary_pairs"]} ordinary pairs, zero guided pairs, byte-identical features.',flush=True)
 write(WORK/'preparation.json',prepared)
 return prepared

def main():
 started=time.monotonic();prepared=prepare();summary={'experiment':'E6_guided_off','started_utc':stamp(),'subjects':{}}
 write(WORK/'summary.json',summary)
 for subject in SUBJECTS:
  dataset=OUT/f'{subject}_quality';baseline=ROOT/'reconstructions'/subject
  stages=[]
  stages.append(run('reconstruction',subject,[ROOT/'work/quality-sfm/run_quality.py',f'experiments/E6_guided_off/{subject}_quality']))
  log=(WORK/f'{subject}_reconstruction.log').read_text()
  if 'reused the completed feature checkpoint' not in log or 'SfM quality: features ' in log or 'waiting for guided matching' in log:
   raise AssertionError('E6 did not exclusively reuse features and skip guided matching')
  report=json.loads((dataset/'reconstruction_report.json').read_text())
  print(f'MODEL READY {subject}: {report["registered_images"]} cameras, {report["points3D"]} points; {report["elapsed_seconds"]:.2f}s',flush=True)
  stages.append(run('preview',subject,[ROOT/'work/full-capture/make_subject_preview.py',dataset,'--boxes',prepared[subject]['resolved_boxes'],'--volume-fraction','.8' if subject=='light_shirt' else '.9']))
  stages.append(run('audit',subject,[ROOT/'work/quality-sfm/audit_model.py',baseline,dataset,'--output',WORK/f'{subject}_model_audit.json']))
  stages.append(run('cache',subject,[ROOT/'work/quality-sfm/verify_cached_results.py',dataset,dataset/'subject_preview','--output',WORK/f'{subject}_cache_verification.json']))
  for path,digest in prepared[subject]['old_files_sha256'].items():
   if sha(path)!=digest:raise AssertionError(f'Previously existing input changed:{path}')
  audit=json.loads((WORK/f'{subject}_model_audit.json').read_text());cache=json.loads((WORK/f'{subject}_cache_verification.json').read_text());preview=json.loads((dataset/'subject_preview/analysis.json').read_text())
  result={'registered_images':report['registered_images'],'full_points':report['points3D'],'preview_points':preview['retained_points'],'reconstruction_seconds':report['elapsed_seconds'],'stages':stages,
   'feature_checkpoint_reused':True,'guided_matching_called':False,'all_previous_input_hashes_unchanged':True,'audit_passed':audit['passed'],'cache_full_and_preview_passed':cache['all_cached'],
   'contributing_images':audit['image_point_support']['contributing_images'],'points_with_at_least_3_distinct_views':audit['tracks']['points_with_at_least_3_views'],
   'fair_quality_gate_full_points':audit['reprojection']['points_passing_distinct_views3_error3_angle1_5_all_observations_valid'],'mean_baseline_equivalent_point_error':audit['reprojection']['per_point_baseline_equivalent_pixels']['mean']}
  summary['subjects'][subject]=result;write(WORK/'summary.json',summary);write(dataset/'experiment_validation.json',result)
  print('SUBJECT COMPLETE '+subject+' '+json.dumps(result),flush=True)
 summary.update(complete=True,completed_utc=stamp(),total_wall_seconds=time.monotonic()-started);write(WORK/'summary.json',summary);write(WORK/'status.json',{'complete':True,'summary':str(WORK/'summary.json')})
 print('E6 COMPLETE '+json.dumps(summary),flush=True)
if __name__=='__main__':main()
