#!/usr/bin/env python3
"""Isolated SQLite staging tests; never touches the live E10 dataset/config."""
from pathlib import Path
from contextlib import closing
import fcntl,importlib.util,json,shutil,sqlite3,tempfile,unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('worker_change',HERE.parents[1]/'change_workers.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class WorkerChangeTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=HERE);self.work=Path(self.temp.name);self.dataset=self.work/'datasets/light_shirt_quality';self.dataset.mkdir(parents=True)
  self.baseline=self.work/'baseline';(self.baseline/'colmap/sparse/0').mkdir(parents=True);(self.baseline/'colmap/sparse/0/cameras.bin').write_bytes(b'model-not-used-by-migration')
  self.images=self.work/'images';self.images.mkdir()
  for name in ('a.jpg','b.jpg','c.jpg'):(self.images/name).write_bytes(name.encode())
  (self.dataset/'images').symlink_to(self.images,target_is_directory=True)
  self.boxes=self.work/'boxes.json';self.boxes.write_text('{}')
  self.config={'baseline_dataset':str(self.baseline),'boxes':str(self.boxes),'matching_pairs':'matching_pairs.txt','guided_pairs':'guided_pairs.txt','guided_lock':str(self.work/'guided.lock'),'max_features':18000,'resume_matching':True,'matching_threads':2,'matching_batch_size':128}
  (self.dataset/'sfm_refine.json').write_text(json.dumps(self.config));(self.work/'runner.lock').touch()
  for name in ('matching_pairs.txt','guided_pairs.txt'):(self.dataset/name).write_text('a.jpg b.jpg\na.jpg c.jpg\nb.jpg c.jpg\n')
  base=self.baseline/'colmap/database.db'
  with closing(sqlite3.connect(base)) as db:
   db.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT); CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER); CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER);')
   db.executemany('INSERT INTO images VALUES(?,?)',[(1,'a.jpg'),(2,'b.jpg'),(3,'c.jpg')]);db.commit()
  baseline_files=[base]+sorted(p for p in (self.baseline/'colmap/sparse').rglob('*') if p.is_file())
  extraction={'max_image_size':-1,'num_threads':1,'sift':{'first_octave':-1,'max_num_features':16384,'peak_threshold':.004,'estimate_affine_shape':True,'domain_size_pooling':True,'dsp_num_scales':10}}
  signature={'recipe_version':1,'images':[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(self.images.iterdir())],
   'pycolmap':m.pycolmap.__version__,'baseline':[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in baseline_files],
   'boxes_sha256':m.sha(self.boxes),'extraction':extraction,'crop_padding':32,'spatial_cells':[8,16],'max_features':18000}
  key=m.digest(signature);features=self.dataset/'colmap/quality_feature_cache';features.mkdir(parents=True);self.feature=features/(key+'.db');shutil.copy2(base,self.feature)
  (features/(key+'.json')).write_text(json.dumps({'signature':signature,'database_size':self.feature.stat().st_size}))
  self.content,self.names,self.pairs,_=m.expected_content(self.dataset,self.config)
  self.cache=self.dataset/'colmap/quality_matching_cache';self.cache.mkdir()
  self.source=self.make_checkpoint(self.content)
  old,_=m.changed_recipe(self.content,1,128);self.old_one=self.make_checkpoint(old)
 def tearDown(self):self.temp.cleanup()
 def make_checkpoint(self,content):
  key=m.digest(content);folder=self.cache/key;folder.mkdir();(self.cache/(key+'.lock')).touch()
  (folder/'manifest.json').write_text(json.dumps(dict(content,checkpoint_key=key)))
  shutil.copy2(self.feature,folder/'checkpoint.db')
  with closing(sqlite3.connect(folder/'checkpoint.db')) as db:
   first=min(self.pairs);db.execute('INSERT INTO matches VALUES(?,0)',(first,));db.execute('INSERT INTO two_view_geometries VALUES(?,0)',(first,));db.commit()
  (folder/'.working.db').write_bytes(b'incomplete-working-file-must-never-be-copied')
  return folder
 def migrate(self,workers=4,batch=None):return m.migrate(self.work,self.dataset,workers,batch)
 def current(self):return json.loads((self.dataset/'sfm_refine.json').read_text())
 def test_two_to_four_copies_stable_only_and_preserves_parent(self):
  source_hash=m.sha(self.source/'checkpoint.db');working_hash=m.sha(self.source/'.working.db');manifest_hash=m.sha(self.source/'manifest.json')
  result=self.migrate();target=Path(result['new_checkpoint']);record=json.loads(Path(result['record']).read_text())
  self.assertEqual(result['preserved_pairs'],1);self.assertEqual(result['remaining_pairs'],2)
  self.assertEqual(m.sha(target/'checkpoint.db'),source_hash);self.assertEqual(m.sha(self.source/'checkpoint.db'),source_hash)
  self.assertEqual(m.sha(self.source/'.working.db'),working_hash);self.assertEqual(m.sha(self.source/'manifest.json'),manifest_hash)
  self.assertNotEqual((target/'checkpoint.db').stat().st_ino,(self.source/'checkpoint.db').stat().st_ino)
  self.assertEqual(record['only_matching_option_changed'],{'num_threads':{'before':2,'after':4}})
  self.assertTrue(record['config_only_execution_changes']);self.assertEqual(self.current(),dict(self.config,matching_threads=4))
  self.assertFalse((target/'.working.db').exists())
 def test_four_to_two_batch129_uses_current_checkpoint_not_old_two(self):
  first=self.migrate();result=self.migrate(2,129);record=json.loads(Path(result['record']).read_text())
  self.assertEqual(record['source_checkpoint_dir'],first['new_checkpoint'])
  self.assertEqual(record['only_matching_option_changed'],{'num_threads':{'before':4,'after':2},'matching_batch_size':{'before':128,'after':129}})
  self.assertEqual(self.current()['matching_batch_size'],129)
  self.assertTrue(self.source.exists());self.assertTrue(self.old_one.exists());self.assertTrue(Path(first['new_checkpoint']).exists())
 def test_small_batch_updates_derived_block_only_when_changed(self):
  result=self.migrate(4,16);r=json.loads(Path(result['record']).read_text())
  self.assertEqual(r['only_matching_option_changed']['pairing_options.block_size'],{'before':32,'after':16})
  manifest=json.loads((Path(result['new_checkpoint'])/'manifest.json').read_text())
  self.assertEqual(manifest['recipe']['pairing_options']['block_size'],16)
 def test_existing_immutable_target_refused(self):
  self.migrate();(self.dataset/'sfm_refine.json').write_text(json.dumps(self.config))
  with self.assertRaises(FileExistsError):self.migrate()
  self.assertEqual(self.current(),self.config)
 def test_unpaired_rows_rejected(self):
  with closing(sqlite3.connect(self.source/'checkpoint.db')) as db:db.execute('DELETE FROM two_view_geometries');db.commit()
  with self.assertRaisesRegex(ValueError,'incompletely'):self.migrate()
  self.assertEqual(self.current(),self.config)
 def test_foreign_pair_rejected_even_when_paired(self):
  with closing(sqlite3.connect(self.source/'checkpoint.db')) as db:
   db.execute('INSERT INTO matches VALUES(123,0)');db.execute('INSERT INTO two_view_geometries VALUES(123,0)');db.commit()
  with self.assertRaisesRegex(ValueError,'foreign'):self.migrate()
 def test_wrong_image_ids_rejected(self):
  with closing(sqlite3.connect(self.source/'checkpoint.db')) as db:db.execute("UPDATE images SET name='wrong.jpg' WHERE image_id=1");db.commit()
  with self.assertRaisesRegex(ValueError,'image IDs'):self.migrate()
 def test_active_source_lock_blocks_without_config_mutation(self):
  with (self.cache/(self.source.name+'.lock')).open('r+') as lock:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):self.migrate()
  self.assertEqual(self.current(),self.config)
 def test_active_runner_lock_blocks(self):
  with (self.work/'runner.lock').open('r+') as lock:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):self.migrate()
 def test_pair_hash_change_invalidates_source_recipe(self):
  for name in ('matching_pairs.txt','guided_pairs.txt'):
   p=self.dataset/name;p.write_text(p.read_text()+'\n')
  with self.assertRaisesRegex(ValueError,'exactly one'):self.migrate()
 def test_nonempty_wal_rejected_and_working_file_ignored(self):
  (self.source/'checkpoint.db-wal').write_bytes(b'uncommitted')
  with self.assertRaisesRegex(ValueError,'WAL'):self.migrate()
 def test_unchanged_workers_and_batch_refuses_current_target(self):
  with self.assertRaises(FileExistsError):self.migrate(2)
 def test_cli_validation_helper_rejects_nonpositive_batch(self):
  with self.assertRaises(ValueError):self.migrate(4,0)

if __name__=='__main__':unittest.main(verbosity=2)
