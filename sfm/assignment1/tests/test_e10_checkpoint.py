"""Exercise E10 orchestration with real SQLite files and a tiny fake matcher.

No image matching or reconstruction workload is run. The real filesystem,
SQLite backup/transactions, atomic checkpoint replacement and method control
flow are tested while existing API fixtures supply tiny images/calibrations.
"""
import itertools
import json
import os
from pathlib import Path
import sqlite3
import types
import unittest
from unittest import mock

import numpy as np
import pycolmap as installed_pycolmap

import test_colmap_api as fixtures


class SQLiteFeatureDatabase:
    def __init__(self, path, events):
        self.connection = sqlite3.connect(path)
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.connection.commit()
        self.connection.close()

    def clear_keypoints(self):
        self.connection.execute("DELETE FROM keypoints")

    def clear_descriptors(self):
        self.connection.execute("DELETE FROM descriptors")

    def clear_matches(self):
        self.connection.execute("DELETE FROM matches")

    def clear_two_view_geometries(self):
        self.connection.execute("DELETE FROM two_view_geometries")

    def update_camera(self, camera):
        self.connection.execute("UPDATE cameras SET width=?,height=?,params=? WHERE camera_id=1",
            (camera.width, camera.height, camera.calibration_matrix().tobytes()))

    def read_image_with_name(self, name):
        row = self.connection.execute("SELECT image_id,name,camera_id FROM images WHERE name=?", (name,)).fetchone()
        return types.SimpleNamespace(image_id=row[0], name=row[1], camera_id=row[2])

    def write_keypoints(self, image_id, matrix):
        self.connection.execute("INSERT INTO keypoints VALUES(?,?,?,?)",
            (image_id, len(matrix), matrix.shape[1], np.asarray(matrix, dtype=np.float32).tobytes()))

    def write_descriptors(self, image_id, descriptors):
        self.connection.execute("INSERT INTO descriptors VALUES(?,?,?,?,?)",
            (image_id, 0, len(descriptors.data), descriptors.data.shape[1], descriptors.data.tobytes()))

    def delete_matches(self, first, second):
        self.connection.execute("DELETE FROM matches WHERE pair_id=?", (installed_pycolmap.image_pair_to_pair_id(first, second),))

    def delete_two_view_geometry(self, first, second):
        self.connection.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (installed_pycolmap.image_pair_to_pair_id(first, second),))


class E10CheckpointTest(unittest.TestCase):
    # Borrow setup/helpers without inheriting and rerunning all unrelated tests.
    setUp = fixtures.PycolmapAPITest.setUp
    _extract = fixtures.PycolmapAPITest._extract
    _map = fixtures.PycolmapAPITest._map
    _api = fixtures.PycolmapAPITest._api
    _run = fixtures.PycolmapAPITest._run
    _prepare_quality_case = fixtures.PycolmapAPITest._prepare_quality_case

    def prepare(self, **changes):
        from PIL import Image
        self._prepare_quality_case()
        names = ["one.jpg", "three.jpg", "two.jpg"]
        Image.new("RGB", (1280, 960), (80, 100, 120)).save(self.image_dir / "three.jpg")
        spec = self.baseline / "colmap/sparse/0/model.json"
        spec.write_text(json.dumps({"images": 3, "point_x": 1, "names": names}))
        self.quality_boxes["three.jpg"] = {"padded_bbox_xyxy": [100, 120, 220, 280]}
        (self.root / "boxes.json").write_text(json.dumps(self.quality_boxes))
        database = self.baseline / "colmap/database.db"
        database.unlink()
        with sqlite3.connect(database) as db:
            db.executescript("""
                CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER);
                CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT UNIQUE,camera_id INTEGER);
                CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);
                CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY,type INTEGER,rows INTEGER,cols INTEGER,data BLOB);
                CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);
                CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB,config INTEGER);
            """)
            db.execute("INSERT INTO cameras VALUES(1,0,640,480,?,0)", (np.eye(3).tobytes(),))
            db.executemany("INSERT INTO images VALUES(?,?,1)", enumerate(names, 1))
        text = "".join(f"{a} {b}\n" for a, b in itertools.combinations(names, 2))
        for filename in ("matching_pairs.txt", "guided_pairs.txt"):
            (self.root / filename).write_text(text)
        recipe_path = self.root / "sfm_refine.json"
        recipe = json.loads(recipe_path.read_text())
        recipe.update(resume_matching=True, matching_batch_size=1, matching_threads=1)
        recipe.update(changes)
        recipe_path.write_text(json.dumps(recipe))
        self.pycolmap.Database.open.side_effect = lambda path: SQLiteFeatureDatabase(path, self.db_events)
        self.pycolmap.FeatureMatchingOptions = installed_pycolmap.FeatureMatchingOptions
        self.pycolmap.TwoViewGeometryOptions = installed_pycolmap.TwoViewGeometryOptions
        self.pycolmap.ImportedPairingOptions = installed_pycolmap.ImportedPairingOptions
        self.pycolmap.image_pair_to_pair_id = installed_pycolmap.image_pair_to_pair_id
        self.pycolmap.pair_id_to_image_pair = installed_pycolmap.pair_id_to_image_pair
        self.pycolmap.match_image_pairs.side_effect = self.match
        self.calls = []
        self.fail_match_call = None
        self.zero_geometry_pair = ("one.jpg", "three.jpg")
        self.good_triangulate = self.pycolmap.triangulate_points.side_effect

    def match(self, **kwargs):
        options = kwargs["matching_options"]
        self.assertTrue(options["guided_matching"] if isinstance(options, dict) else options.guided_matching)
        pairing = kwargs["pairing_options"]
        pair_file = pairing["match_list_path"] if isinstance(pairing, dict) else pairing.match_list_path
        rows = [tuple(line.split()) for line in Path(pair_file).read_text().splitlines() if line.strip()]
        self.calls.append(rows)
        with sqlite3.connect(kwargs["database_path"]) as db:
            image_ids = dict(db.execute("SELECT name,image_id FROM images"))
            for first, second in rows:
                pair_id = installed_pycolmap.image_pair_to_pair_id(image_ids[first], image_ids[second])
                completed = db.execute("SELECT 1 FROM matches m JOIN two_view_geometries g USING(pair_id) WHERE pair_id=?", (pair_id,)).fetchone()
                self.assertIsNone(completed, "Completed checkpoint pairs must not be submitted again")
                db.execute("INSERT OR REPLACE INTO matches VALUES(?,20,2,?)", (pair_id, b"raw"))
                db.commit()
                if self.fail_match_call == len(self.calls):
                    raise RuntimeError("crash after raw match write")
                empty = tuple(sorted((first, second))) == self.zero_geometry_pair
                db.execute("INSERT OR REPLACE INTO two_view_geometries VALUES(?,?,?,?,?)",
                    (pair_id, 0 if empty else 20, 2, b"" if empty else b"verified", 0 if empty else 2))
                db.commit()

    def checkpoint(self):
        paths = list((self.root / "colmap/quality_matching_cache").glob("*/checkpoint.db"))
        self.assertEqual(len(paths), 1)
        return paths[0]

    def completed_count(self):
        with sqlite3.connect(self.checkpoint()) as db:
            return db.execute("SELECT COUNT(*) FROM matches JOIN two_view_geometries USING(pair_id)").fetchone()[0]

    def test_zero_geometry_is_complete_and_triangulation_failure_resumes_without_matching(self):
        self.prepare()
        self.pycolmap.triangulate_points.side_effect = RuntimeError("triangulation interrupted")
        self.assertIsInstance(self._run(self._api()), RuntimeError)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.completed_count(), 3)
        with sqlite3.connect(self.checkpoint()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows=0").fetchone()[0], 1)
        self.assertFalse((self.root / "colmap/sparse/sfm_inputs.json").exists())
        self.pycolmap.triangulate_points.side_effect = self.good_triangulate
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 3)
        self.assertTrue((self.root / "colmap/sparse/sfm_inputs.json").is_file())

    def test_partial_raw_write_is_discarded_but_prior_durable_batch_is_reused(self):
        self.prepare()
        self.fail_match_call = 2
        self.assertIsInstance(self._run(self._api()), RuntimeError)
        self.assertEqual(self.completed_count(), 1)
        self.fail_match_call = None
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[1], self.calls[2])
        self.assertNotEqual(self.calls[0], self.calls[2])
        self.assertEqual(self.completed_count(), 3)
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 3)

    def test_failed_checkpoint_replace_retries_only_uncommitted_batch(self):
        self.prepare()
        original = os.replace
        failed = []
        def fail_second_checkpoint(source, destination):
            if Path(destination).name == "checkpoint.db" and Path(destination).is_file() and not failed:
                failed.append(True)
                raise OSError("simulated checkpoint install failure")
            return original(source, destination)
        with mock.patch("os.replace", side_effect=fail_second_checkpoint):
            self.assertIsInstance(self._run(self._api()), OSError)
        self.assertEqual(self.completed_count(), 0)
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[0], self.calls[1])
        self.assertEqual(self.completed_count(), 3)

    def test_stale_progress_does_not_override_committed_database(self):
        self.prepare()
        original = os.replace
        failed = []
        def fail_progress_after_commit(source, destination):
            if Path(destination).name == "progress.json" and not failed:
                checkpoint = Path(destination).parent / "checkpoint.db"
                if checkpoint.is_file():
                    with sqlite3.connect(checkpoint) as db:
                        count = db.execute("SELECT COUNT(*) FROM matches JOIN two_view_geometries USING(pair_id)").fetchone()[0]
                    if count:
                        failed.append(True)
                        raise OSError("progress write interrupted after DB commit")
            return original(source, destination)
        with mock.patch("os.replace", side_effect=fail_progress_after_commit):
            self.assertIsInstance(self._run(self._api()), OSError)
        self.assertEqual(self.completed_count(), 1)
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.completed_count(), 3)

    def test_invalid_resume_options_are_rejected_before_matching(self):
        self.prepare()
        recipe_file = self.root / "sfm_refine.json"
        original = json.loads(recipe_file.read_text())
        for key, value in (("resume_matching", "yes"), ("matching_batch_size", 0),
            ("matching_batch_size", True), ("matching_threads", 0), ("matching_threads", True)):
            with self.subTest(key=key, value=value):
                recipe = dict(original); recipe[key] = value
                recipe_file.write_text(json.dumps(recipe))
                self.assertIsInstance(self._run(self._api()), ValueError)
        self.assertEqual(self.calls, [])
        self.pycolmap.triangulate_points.assert_not_called()

    def test_incomplete_exhaustive_or_guided_pair_set_is_rejected(self):
        self.prepare()
        ordinary = self.root / "matching_pairs.txt"
        guided = self.root / "guided_pairs.txt"
        complete = ordinary.read_text()
        ordinary.write_text("one.jpg two.jpg\n")
        guided.write_text("one.jpg two.jpg\n")
        self.assertIsInstance(self._run(self._api()), ValueError)
        ordinary.write_text(complete)
        self.assertIsInstance(self._run(self._api()), ValueError)
        self.assertEqual(self.calls, [])
        self.pycolmap.triangulate_points.assert_not_called()


if __name__ == "__main__":
    unittest.main()
