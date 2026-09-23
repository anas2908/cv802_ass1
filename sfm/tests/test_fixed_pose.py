"""Safety/scientific invariants of the separate fixed-pose reuse command."""
import copy
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pycolmap

from sfm_engine import fixed_pose as engine


class FakeCamera:
    model = SimpleNamespace(value=2)
    width = height = 8
    params = np.array([4., 4., 4., 0.])

    def rescale(self, width, height):
        self.width, self.height = width, height


class FixedPoseTests(unittest.TestCase):
    def setUp(self):
        # The caller sets TMPDIR under DATA_ROOT; never place fixture images/DB in code.
        temporary_root = Path(os.environ['TMPDIR'])
        if not temporary_root.is_absolute() or not temporary_root.is_dir():
            raise RuntimeError('TMPDIR must name an existing data-root directory')
        self.tmp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "database.db"
        self.connection = sqlite3.connect(self.db)
        self.addCleanup(self.connection.close)
        self.connection.executescript("""
            CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER);
            CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,height INTEGER,params BLOB);
            CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);
        """)
        self.connection.execute("INSERT INTO cameras VALUES(1,2,8,8,?)", (FakeCamera.params.astype('<f8').tobytes(),))
        for image_id in (1, 2):
            name = f"{image_id}.png"
            pycolmap.Bitmap.from_array(np.zeros((8, 8, 3), dtype=np.uint8)).write(self.root / name)
            self.connection.execute("INSERT INTO images VALUES(?,?,1)", (image_id, name))
            points = np.array([[1, 1, 1, 0, 0, 1], [2, 2, 1, 0, 0, 1]], dtype='<f4')
            self.connection.execute("INSERT INTO keypoints VALUES(?,2,6,?)", (image_id, points.tobytes()))
            self.connection.execute("INSERT INTO descriptors VALUES(?,2,128,?)", (image_id, bytes(256)))
        for table in ('matches', 'two_view_geometries'):
            self.connection.execute(f"INSERT INTO {table} VALUES(?,2,2,?)", (engine.PAIR_BASE + 2, np.array([[0, 0], [1, 1]], dtype='<u4').tobytes()))
        self.connection.commit()
        self.model = SimpleNamespace(cameras={1: FakeCamera()}, images={i: SimpleNamespace(name=f'{i}.png', camera_id=1) for i in (1, 2)}, reg_image_ids=lambda: [1, 2])

    def test_valid_affine_features_and_matches_are_accounted_for(self):
        receipt = engine.validate_database(self.connection, self.model, self.root)
        self.assertEqual(receipt['features'], 4)
        self.assertEqual(receipt['nonempty_pairs']['two_view_geometries'], 1)

    def test_mismatched_camera_reference_rejected(self):
        self.connection.execute('UPDATE images SET camera_id=9 WHERE image_id=1')
        with self.assertRaisesRegex(ValueError, 'reference'):
            engine.validate_database(self.connection, self.model, self.root)

    def test_out_of_range_match_rejected(self):
        self.connection.execute('UPDATE two_view_geometries SET data=?', (np.array([[0, 99], [1, 1]], dtype='<u4').tobytes(),))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            engine.validate_database(self.connection, self.model, self.root)

    def test_scaled_intrinsics_mismatch_rejected(self):
        self.connection.execute('UPDATE cameras SET params=?', (np.array([5, 4, 4, 0], dtype='<f8').tobytes(),))
        with self.assertRaisesRegex(ValueError, 'intrinsics'):
            engine.validate_database(self.connection, self.model, self.root)

    def test_checkpoint_feature_tampering_changes_content_hash(self):
        before = engine.table_digest(self.connection, 'descriptors', 'image_id')
        self.connection.execute('UPDATE descriptors SET data=? WHERE image_id=1', (bytes([1]) * 256,))
        self.assertNotEqual(before, engine.table_digest(self.connection, 'descriptors', 'image_id'))

    def test_pending_wal_is_rejected_without_deletion(self):
        self.connection.close()
        wal = Path(str(self.db) + '-wal')
        wal.write_bytes(b'not checkpointed')
        with self.assertRaisesRegex(RuntimeError, 'pending or active'):
            engine.check_quiet_database(self.db)
        self.assertEqual(wal.read_bytes(), b'not checkpointed')

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux handle check')
    def test_open_writable_source_is_rejected(self):
        self.connection.close()
        with self.db.open('r+b'):
            with self.assertRaisesRegex(RuntimeError, 'writable handle'):
                engine.check_quiet_database(self.db)

    def test_readonly_inspection_leaves_source_bytes_identical(self):
        self.connection.close()
        before = engine.fingerprint(self.db)
        with engine.readonly_database(self.db) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM images').fetchone()[0], 2)
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute('DELETE FROM images')
        self.assertEqual(before, engine.fingerprint(self.db))

    def test_pose_change_is_detected_after_triangulation(self):
        expected = {'cameras': {'1': {'params': [1, 2, 3]}}, 'poses': {'1': {'matrix': [[1, 0, 0, 0]]}}}
        actual = copy.deepcopy(expected)
        self.assertTrue(engine.verify_fixed_calibration(expected, actual)['fixed_extrinsics'])
        actual['poses']['1']['matrix'][0][-1] = 1e-7
        with self.assertRaisesRegex(ValueError, 'changed fixed'):
            engine.verify_fixed_calibration(expected, actual)

    def test_recipe_explicitly_freezes_calibration_and_cpu_ba(self):
        options = engine.triangulation_options(2)
        self.assertTrue(options['fix_existing_frames'])
        self.assertTrue(options['mapper']['fix_existing_frames'])
        for name in ('ba_refine_focal_length', 'ba_refine_principal_point', 'ba_refine_extra_params', 'ba_refine_sensor_from_rig', 'ba_use_gpu'):
            self.assertFalse(options[name])
        # Validate actual installed API, not merely our dictionary spelling.
        pycolmap.IncrementalPipelineOptions(options)

    def test_legacy_experiment_id_and_working_database_rejected(self):
        fields = dict(image_dir=self.root, baseline_model=self.root, database=self.db, feature_database=self.db, feature_metadata=self.db)
        with patch.object(engine, 'SFM_DATA_ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'historical names'):
                engine.FixedPoseConfig('E10_replace', **fields).validate()
            bad = self.root / '.working.db'
            bad.touch()
            fields['database'] = bad
            with self.assertRaisesRegex(ValueError, '.working'):
                engine.FixedPoseConfig('E12_example', **fields).validate()


if __name__ == '__main__':
    unittest.main()
