import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _PointCloud:
    def __init__(self):
        self.points = None
        self.colors = None


# The production application uses Open3D.  A tiny stand-in keeps this test
# focused on reconstruction orchestration and data conversion.
if "open3d" not in sys.modules:
    fake_open3d = types.ModuleType("open3d")
    fake_open3d.geometry = types.SimpleNamespace(PointCloud=_PointCloud)
    fake_open3d.utility = types.SimpleNamespace(
        Vector3dVector=lambda values: np.asarray(values, dtype=np.float64)
    )
    sys.modules["open3d"] = fake_open3d

from modules.colmap.api import (
    ColmapAPI,
    _contains_sparse_model,
    _gpu_arguments,
    _load_largest_sparse_model,
    _replace_cached_reconstruction,
)



class _Reconstruction:
    """Only the public PyCOLMAP fields consumed by the SfM function."""

    def __init__(self, directory):
        spec = json.loads((Path(directory) / "model.json").read_text())
        self.cameras = {
            1: types.SimpleNamespace(
                width=640,
                height=480,
                calibration_matrix=lambda: np.array(
                    [[500, 0, 320], [0, 510, 240], [0, 0, 1]], dtype=float
                ),
            )
        }
        self.images = {}
        for number in range(1, spec["images"] + 1):
            pose = types.SimpleNamespace(
                rotation=types.SimpleNamespace(matrix=lambda: np.eye(3)),
                translation=np.array([number - 1, 0, 0], dtype=float),
            )
            self.images[number] = types.SimpleNamespace(
                name=f"image {number:02d}.jpg",
                camera_id=1,
                cam_from_world=lambda pose=pose: pose,
            )
        self.points3D = {
            7: types.SimpleNamespace(
                xyz=np.array([spec["point_x"], 2, 3], dtype=float),
                color=np.array([255, 128, 0], dtype=np.uint8),
            )
        } if spec.get("valid", True) else {}

    def num_reg_images(self):
        return len(self.images)

    def num_points3D(self):
        return len(self.points3D)

    def reg_image_ids(self):
        return list(self.images)


class _QualityCamera:
    def __init__(self, width=640, height=480):
        self.camera_id = 1
        self.width, self.height = width, height
        self.has_prior_focal_length = False

    def rescale(self, width, height):
        self.width, self.height = width, height

    def calibration_matrix(self):
        return np.array([
            [500 * self.width / 640, 0, self.width / 2],
            [0, 510 * self.height / 480, self.height / 2],
            [0, 0, 1],
        ], dtype=float)


class _QualityReconstruction(_Reconstruction):
    def __init__(self, directory):
        super().__init__(directory)
        spec = json.loads((Path(directory) / "model.json").read_text())
        self.cameras = {1: _QualityCamera(*spec.get("size", [640, 480]))}
        for image_id, name in enumerate(spec["names"], start=1):
            self.images[image_id].name = name
            self.images[image_id].image_id = image_id

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for stem in ("cameras", "images", "points3D"):
            (directory / f"{stem}.bin").touch()
        camera = self.cameras[1]
        (directory / "model.json").write_text(json.dumps({
            "images": len(self.images),
            "names": [image.name for image in self.images.values()],
            "point_x": float(self.points3D[7].xyz[0]),
            "size": [camera.width, camera.height],
        }))


class _QualityDatabase:
    """Persist minimal DB state so real copies/renames exercise cache safety."""

    def __init__(self, path, events):
        self.path = Path(path)
        self.state = json.loads(self.path.read_text())
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        self.path.write_text(json.dumps(self.state))

    def clear_keypoints(self):
        self.state["keypoints"] = {}

    def clear_descriptors(self):
        self.state["descriptors"] = {}

    def clear_matches(self):
        self.events.append(("clear_matches",))

    def clear_two_view_geometries(self):
        self.events.append(("clear_two_view_geometries",))

    def update_camera(self, camera):
        self.state["calibration"] = {
            "size": [camera.width, camera.height],
            "K": camera.calibration_matrix().tolist(),
            "has_prior_focal_length": camera.has_prior_focal_length,
        }

    def read_image_with_name(self, name):
        return types.SimpleNamespace(
            name=name, image_id=self.state["images"][name], camera_id=1
        )

    def write_keypoints(self, image_id, keypoints):
        self.state["keypoints"][str(image_id)] = np.asarray(keypoints).tolist()

    def write_descriptors(self, image_id, descriptors):
        self.state["descriptors"][str(image_id)] = descriptors.data.tolist()

    def delete_matches(self, image_id1, image_id2):
        self.events.append(("delete_matches", image_id1, image_id2))

    def delete_two_view_geometry(self, image_id1, image_id2):
        self.events.append(("delete_two_view_geometry", image_id1, image_id2))


class PycolmapAPITest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        for number in (1, 2):
            (self.image_dir / f"image {number:02d}.jpg").touch()
        self.components = [{"images": 2, "point_x": 1}]
        self.pycolmap = types.ModuleType("pycolmap")
        self.pycolmap.__version__ = "4.2.0"
        self.pycolmap.CameraMode = types.SimpleNamespace(
            AUTO="auto", PER_FOLDER="per_folder"
        )
        self.pycolmap.Device = types.SimpleNamespace(cpu="cpu")
        self.pycolmap.Reconstruction = _Reconstruction
        self.pycolmap.extract_features = mock.Mock(side_effect=self._extract)
        self.pycolmap.match_exhaustive = mock.Mock()
        self.pycolmap.match_sequential = mock.Mock()
        self.pycolmap.match_vocabtree = mock.Mock()
        self.pycolmap.incremental_mapping = mock.Mock(side_effect=self._map)
        module_patch = mock.patch.dict(sys.modules, {"pycolmap": self.pycolmap})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def _extract(self, **kwargs):
        Path(kwargs["database_path"]).write_text("new database")

    def _map(self, **kwargs):
        models = {}
        for number, spec in enumerate(self.components):
            directory = Path(kwargs["output_path"]) / str(number)
            directory.mkdir()
            for stem in ("cameras", "images", "points3D"):
                (directory / f"{stem}.bin").touch()
            (directory / "model.json").write_text(json.dumps(spec))
            models[number] = _Reconstruction(directory)
        return models

    def _api(self, matcher="exhaustive_matcher"):
        api = ColmapAPI(0, "OPENCV", matcher)
        api.data_path = str(self.root)
        return api

    def _run(self, api, recompute=False):
        api.estimate_cameras(recompute)
        api._thread.join(timeout=5)
        self.assertFalse(api._thread.is_alive(), "SfM worker did not finish")
        return api.estimate_error

    def _cache_bytes(self):
        return {
            path.relative_to(self.root): path.read_bytes()
            for path in (self.root / "colmap").rglob("*") if path.is_file()
        }

    def _prepare_quality_case(self):
        from PIL import Image

        for path in self.image_dir.iterdir():
            path.unlink()
        for name in ("one.jpg", "two.jpg"):
            Image.new("RGB", (1280, 960), color=(80, 100, 120)).save(
                self.image_dir / name
            )
        self.baseline = self.root / "baseline"
        baseline_sparse = self.baseline / "colmap" / "sparse" / "0"
        baseline_sparse.mkdir(parents=True)
        (baseline_sparse / "model.json").write_text(json.dumps({
            "images": 2, "point_x": 1, "names": ["one.jpg", "two.jpg"],
        }))
        for stem in ("cameras", "images", "points3D"):
            (baseline_sparse / f"{stem}.bin").touch()
        (self.baseline / "colmap" / "database.db").write_text(json.dumps({
            "images": {"one.jpg": 1, "two.jpg": 2},
            "frames": ["original frames"],
            "rigs": ["original rigs"],
            "keypoints": {"old": []}, "descriptors": {"old": []},
        }))
        self.quality_boxes = {
            name: {"padded_bbox_xyxy": [100, 120, 220, 280]}
            for name in ("one.jpg", "two.jpg")
        }
        (self.root / "boxes.json").write_text(json.dumps(self.quality_boxes))
        for filename in ("matching_pairs.txt", "guided_pairs.txt"):
            (self.root / filename).write_text("one.jpg two.jpg\n")
        (self.root / "sfm_refine.json").write_text(json.dumps({
            "baseline_dataset": "baseline",
            "boxes": "boxes.json",
            "matching_pairs": "matching_pairs.txt",
            "guided_pairs": "guided_pairs.txt",
            "guided_lock": "guided.lock",
            "max_features": 2,
        }))

        self.db_events = []
        self.pycolmap.Reconstruction = _QualityReconstruction
        self.pycolmap.Database = types.SimpleNamespace(open=mock.Mock(
            side_effect=lambda path: _QualityDatabase(path, self.db_events)
        ))
        self.pycolmap.FeatureExtractionOptions = mock.Mock(
            side_effect=lambda options: options
        )
        self.feature_rows = np.array([
            [2, 3, 1, 0.1, 0.2, 2],
            [-1000, 25, 3, 0.3, 0.4, 4],  # Invalid point must drop its descriptor too.
            [35, 45, 5, 0.5, 0.6, 6],
            [55, 65, 7, 0.7, 0.8, 8],
        ], dtype=np.float32)
        keypoints = [types.SimpleNamespace(**dict(zip(
            ("x", "y", "a11", "a12", "a21", "a22"), row
        ))) for row in self.feature_rows]
        descriptors = types.SimpleNamespace(
            type="SIFT", data=np.repeat(np.arange(4, dtype=np.uint8)[:, None], 128, axis=1)
        )
        self.extractor = types.SimpleNamespace(
            extract_from_uint8_array=mock.Mock(return_value=(keypoints, descriptors))
        )
        self.pycolmap.FeatureExtractor = types.SimpleNamespace(
            create=mock.Mock(return_value=self.extractor)
        )
        self.pycolmap.FeatureDescriptors = mock.Mock(
            side_effect=lambda descriptor_type, data: types.SimpleNamespace(
                type=descriptor_type, data=np.asarray(data)
            )
        )
        self.pycolmap.match_image_pairs = mock.Mock()

        def triangulate(reconstruction, **kwargs):
            reconstruction.points3D[7].xyz[0] = 9
            reconstruction.write(kwargs["output_path"])
            return reconstruction

        self.pycolmap.triangulate_points = mock.Mock(side_effect=triangulate)

    def _result_bytes(self):
        paths = [self.root / "colmap" / "database.db"]
        paths.extend((self.root / "colmap" / "sparse").rglob("*"))
        return {path.relative_to(self.root): path.read_bytes()
                for path in paths if path.is_file()}

    def test_quality_crops_preserve_affine_features_calibration_and_fixed_poses(self):
        self._prepare_quality_case()
        baseline_before = {path.relative_to(self.baseline): path.read_bytes()
                           for path in self.baseline.rglob("*") if path.is_file()}
        api = self._api()
        with mock.patch("os.cpu_count", return_value=16):
            self.assertIsNone(self._run(api))

        self.pycolmap.extract_features.assert_not_called()
        self.pycolmap.incremental_mapping.assert_not_called()
        database = json.loads(Path(api.database_path).read_text())
        self.assertEqual(database["images"], {"one.jpg": 1, "two.jpg": 2})
        self.assertEqual(database["frames"], ["original frames"])
        self.assertEqual(database["rigs"], ["original rigs"])
        self.assertEqual(database["calibration"]["size"], [1280, 960])
        self.assertTrue(database["calibration"]["has_prior_focal_length"])
        np.testing.assert_allclose(database["calibration"]["K"],
                                   [[1000, 0, 640], [0, 1020, 480], [0, 0, 1]])
        for image_id in ("1", "2"):
            keypoints = np.asarray(database["keypoints"][image_id])
            descriptors = np.asarray(database["descriptors"][image_id])
            self.assertEqual(keypoints.shape, (2, 6))
            self.assertEqual(descriptors.shape, (2, 128))
            expected = self.feature_rows[descriptors[:, 0].astype(int)].copy()
            expected[:, :2] += [68, 88]  # Native crop origin, including padding.
            np.testing.assert_allclose(keypoints, expected)
        for call in self.extractor.extract_from_uint8_array.call_args_list:
            pixels = call.args[0]
            self.assertEqual(pixels.shape, (224, 184))
            self.assertEqual(pixels.dtype, np.uint8)

        ordinary, guided = self.pycolmap.match_image_pairs.call_args_list
        self.assertFalse(ordinary.kwargs["matching_options"]["guided_matching"])
        self.assertEqual(ordinary.kwargs["matching_options"]["num_threads"], 4)
        self.assertTrue(guided.kwargs["matching_options"]["guided_matching"])
        self.assertEqual(guided.kwargs["matching_options"]["num_threads"], 1)
        self.assertIn(("delete_two_view_geometry", 1, 2), self.db_events)
        self.assertFalse(any(event[0] == "delete_matches" for event in self.db_events))
        triangulation = self.pycolmap.triangulate_points.call_args.kwargs
        self.assertTrue(triangulation["clear_points"])
        self.assertFalse(triangulation["refine_intrinsics"])
        self.assertEqual(Path(triangulation["output_path"]).name, "0")
        self.assertEqual(api._cameras["two.jpg"]["intrinsic"]["fx"], 1000)
        np.testing.assert_allclose(api._cameras["two.jpg"]["extrinsic"][1], [1, 0, 0])
        self.assertEqual(baseline_before, {
            path.relative_to(self.baseline): path.read_bytes()
            for path in self.baseline.rglob("*") if path.is_file()
        })

    def test_quality_cache_tracks_recipe_and_boxes_but_ignores_gui_defaults(self):
        self._prepare_quality_case()
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 2)
        fresh = self._api("sequential_matcher")
        fresh.camera_model = "SIMPLE_RADIAL"
        self.assertIsNone(self._run(fresh))
        self.assertEqual(self.pycolmap.match_image_pairs.call_count, 2)

        # Pair changes require a new result, but the expensive crop features
        # have identical inputs and must be reusable.
        (self.root / "matching_pairs.txt").write_text("two.jpg one.jpg\n")
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(self.pycolmap.match_image_pairs.call_count, 4)
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 2)

        self.quality_boxes["one.jpg"]["padded_bbox_xyxy"][0] = 110
        (self.root / "boxes.json").write_text(json.dumps(self.quality_boxes))
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(self.pycolmap.match_image_pairs.call_count, 6)
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 4)
        self.assertEqual(self.pycolmap.triangulate_points.call_count, 3)

    def test_quality_matching_failure_preserves_result_and_reuses_new_features(self):
        self._prepare_quality_case()
        api = self._api()
        self.assertIsNone(self._run(api))
        previous = self._result_bytes()
        self.quality_boxes["one.jpg"]["padded_bbox_xyxy"][0] = 110
        (self.root / "boxes.json").write_text(json.dumps(self.quality_boxes))
        self.pycolmap.match_image_pairs.side_effect = RuntimeError("matching failed")
        self.assertIsInstance(self._run(api, recompute=True), RuntimeError)
        self.assertEqual(self._result_bytes(), previous)
        np.testing.assert_allclose(api.pcd.points, [[9, 2, 3]])
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 4)

        self.pycolmap.match_image_pairs.side_effect = None
        self.assertIsNone(self._run(api, recompute=True))
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 4)
        self.assertEqual(self.pycolmap.triangulate_points.call_count, 2)
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(self.extractor.extract_from_uint8_array.call_count, 4)

    def test_cpu_pipeline_converts_results_and_reuses_cache(self):
        api = self._api()
        with mock.patch("os.cpu_count", return_value=16):
            self.assertIsNone(self._run(api))
        self.assertEqual(api.num_cameras, 2)
        np.testing.assert_allclose(api.pcd.points, [[1, 2, 3]])
        np.testing.assert_allclose(api.pcd.colors, [[1, 128 / 255, 0]])
        camera = api._cameras["image 02.jpg"]
        np.testing.assert_allclose(camera["extrinsic"][0], np.eye(3))
        np.testing.assert_allclose(camera["extrinsic"][1], [1, 0, 0])
        self.assertEqual(camera["intrinsic"]["fy"], 510)
        self.assertEqual(api.activate_camera_name, "image 01.jpg")
        extraction = self.pycolmap.extract_features.call_args.kwargs
        self.assertEqual(extraction["device"], "cpu")
        self.assertEqual(extraction["camera_mode"], "auto")
        self.assertEqual(extraction["image_names"], ["image 01.jpg", "image 02.jpg"])
        self.assertEqual(extraction["extraction_options"]["max_image_size"], 3200)
        self.assertEqual(extraction["extraction_options"]["num_threads"], 4)
        self.assertEqual(
            extraction["extraction_options"]["sift"]["max_num_features"], 16384
        )
        self.assertEqual(extraction["extraction_options"]["sift"]["first_octave"], 0)
        matching = self.pycolmap.match_exhaustive.call_args.kwargs
        self.assertEqual(matching["device"], "cpu")
        self.assertEqual(matching["matching_options"]["num_threads"], 4)
        # CPU guided matching allocates dense descriptor-pair matrices. Keep
        # ordinary geometric verification enabled without that extra pass.
        self.assertFalse(matching["matching_options"]["guided_matching"])
        self.assertFalse(
            matching["matching_options"].get("skip_geometric_verification", False)
        )
        mapping = self.pycolmap.incremental_mapping.call_args.kwargs
        self.assertEqual(mapping["options"]["num_threads"], 4)
        self.assertTrue(mapping["options"]["extract_colors"])

        fresh = self._api()
        self.assertIsNone(self._run(fresh))
        np.testing.assert_allclose(fresh.pcd.points, api.pcd.points)
        self.assertEqual(self.pycolmap.extract_features.call_count, 1)
        self.assertEqual(self.pycolmap.incremental_mapping.call_count, 1)

    def test_camera_group_marker_changes_intrinsics_grouping_and_invalidates_cache(self):
        # The same image tree can use EXIF grouping or explicit capture folders.
        # Switching this policy must rebuild the camera database even when none
        # of the image bytes or timestamps changed.
        for number, folder_name in ((1, "phone_photos"), (2, "video_clip")):
            folder = self.image_dir / folder_name
            folder.mkdir()
            name = f"image {number:02d}.jpg"
            (self.image_dir / name).rename(folder / name)

        self.assertIsNone(self._run(self._api()))
        self.assertEqual(
            self.pycolmap.extract_features.call_args.kwargs["camera_mode"], "auto"
        )

        marker = self.root / "camera_groups.json"
        marker.write_text(json.dumps({"groups": ["phone_photos", "video_clip"]}))
        grouped = self._api()
        self.assertIsNone(self._run(grouped))
        extraction = self.pycolmap.extract_features.call_args.kwargs
        self.assertEqual(extraction["camera_mode"], "per_folder")
        self.assertEqual(
            extraction["image_names"],
            ["phone_photos/image 01.jpg", "video_clip/image 02.jpg"],
        )
        self.assertEqual(self.pycolmap.extract_features.call_count, 2)
        signature = json.loads(
            (Path(grouped.sparse_dir) / "sfm_inputs.json").read_text()
        )
        self.assertEqual(signature["camera_mode"], "PER_FOLDER")

        self.assertIsNone(self._run(self._api()))
        self.assertEqual(self.pycolmap.extract_features.call_count, 2)
        marker.unlink()
        self.assertIsNone(self._run(self._api()))
        self.assertEqual(self.pycolmap.extract_features.call_count, 3)
        self.assertEqual(
            self.pycolmap.extract_features.call_args.kwargs["camera_mode"], "auto"
        )

    def test_changed_images_rebuild_but_settings_apply_only_on_explicit_recompute(self):
        api = self._api()
        self.assertIsNone(self._run(api))
        (self.image_dir / "image 01.jpg").write_bytes(b"new photo contents")
        self.assertIsNone(self._run(api))
        self.assertEqual(self.pycolmap.extract_features.call_count, 2)
        api.camera_model = "SIMPLE_RADIAL"
        self.assertIsNone(self._run(api))
        self.assertEqual(self.pycolmap.extract_features.call_count, 2)
        api.matcher = "sequential_matcher"
        self.assertIsNone(self._run(api))
        self.assertEqual(self.pycolmap.extract_features.call_count, 2)
        self.pycolmap.match_sequential.assert_not_called()

        self.assertIsNone(self._run(api, recompute=True))
        self.assertEqual(self.pycolmap.extract_features.call_count, 3)
        self.assertEqual(
            self.pycolmap.extract_features.call_args.kwargs["reader_options"]["camera_model"],
            "SIMPLE_RADIAL",
        )
        self.assertEqual(
            self.pycolmap.match_sequential.call_args.kwargs["pairing_options"],
            {"overlap": 10, "loop_detection": False},
        )

        # Opening this saved result from a fresh GUI must not recompute it
        # using the GUI's default OPENCV/exhaustive settings or a newer library.
        self.pycolmap.__version__ = "4.3.0"
        fresh = self._api()
        self.assertIsNone(self._run(fresh))
        np.testing.assert_allclose(fresh.pcd.points, api.pcd.points)
        self.assertEqual(self.pycolmap.extract_features.call_count, 3)
        self.assertEqual(self.pycolmap.incremental_mapping.call_count, 3)
        self.assertEqual(self.pycolmap.match_exhaustive.call_count, 2)
        self.assertEqual(self.pycolmap.match_sequential.call_count, 1)

    def test_failed_recompute_preserves_cache_and_previous_result(self):
        api = self._api()
        self.assertIsNone(self._run(api))
        previous = self._cache_bytes()
        self.pycolmap.extract_features.side_effect = RuntimeError("extraction failed")
        self.assertIsInstance(self._run(api, recompute=True), RuntimeError)
        self.assertEqual(self._cache_bytes(), previous)
        np.testing.assert_allclose(api.pcd.points, [[1, 2, 3]])
        self.assertIsNone(self._run(self._api()))
        self.pycolmap.extract_features.side_effect = self._extract
        self.components[0]["valid"] = False
        self.assertIsInstance(self._run(api, recompute=True), RuntimeError)
        self.assertEqual(self._cache_bytes(), previous)
        self.assertIsNone(self._run(self._api()))

    def test_invalid_first_run_is_not_cached_and_can_be_retried(self):
        api = self._api()
        self.components[0]["valid"] = False
        self.assertIsInstance(self._run(api), RuntimeError)
        self.assertFalse(Path(api.database_path).exists())
        self.assertFalse(Path(api.sparse_dir).exists())
        self.components[0]["valid"] = True
        self.assertIsNone(self._run(api))
        self.assertTrue(Path(api.database_path).exists())
        self.assertIsNone(api.estimate_error)

    def test_largest_component_is_selected_on_compute_and_cache_load(self):
        (self.image_dir / "image 03.jpg").touch()
        self.components = [
            {"images": 2, "point_x": 1},
            {"images": 3, "point_x": 9},
        ]
        for api in (self._api(), self._api()):
            self.assertIsNone(self._run(api))
            self.assertEqual(api.num_cameras, 3)
            np.testing.assert_allclose(api.pcd.points, [[9, 2, 3]])
        self.assertEqual(self.pycolmap.incremental_mapping.call_count, 1)

    def test_cache_install_failure_restores_previous_cache(self):
        api = self._api()
        self.assertIsNone(self._run(api))
        previous = self._cache_bytes()
        original_rename = Path.rename

        def fail_installing_sparse(source, destination):
            if source.name == "sparse" and source.parent.name.startswith("sfm_run_"):
                raise OSError("simulated installation failure")
            return original_rename(source, destination)

        self.components[0]["point_x"] = 9
        with mock.patch.object(Path, "rename", new=fail_installing_sparse):
            self.assertIsInstance(self._run(api, recompute=True), OSError)
        self.assertEqual(self._cache_bytes(), previous)
        cached = self._api()
        self.assertIsNone(self._run(cached))
        np.testing.assert_allclose(cached.pcd.points, [[1, 2, 3]])


class ColmapHelpersTest(unittest.TestCase):
    def test_image_discovery_supports_nested_raster_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "camera_a"
            nested.mkdir()
            (root / "one.jpg").touch()
            (nested / "two.tif").touch()
            (nested / "not_an_image.svg").touch()

            images = ColmapAPI._list_images_in_folder(str(root))

            self.assertEqual(
                [Path(path).relative_to(root) for path in images],
                [Path("camera_a/two.tif"), Path("one.jpg")],
            )

    def test_legacy_gpu_flags_and_mixed_model_detection(self):
        self.assertEqual(
            _gpu_arguments(
                "--SiftExtraction.use_gpu --SiftExtraction.gpu_index",
                ("FeatureExtraction", "SiftExtraction"),
                2,
            ),
            [
                "--SiftExtraction.use_gpu",
                "1",
                "--SiftExtraction.gpu_index",
                "2",
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "cameras.bin").touch()
            (model_dir / "images.txt").touch()
            (model_dir / "points3D.txt").touch()
            self.assertFalse(_contains_sparse_model(model_dir))

    def test_selects_component_with_the_most_registered_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smaller = root / "0"
            larger = root / "1"

            for model in (smaller, larger):
                model.mkdir()
                (model / "cameras.txt").write_text(
                    "1 PINHOLE 640 480 500 500 320 240\n",
                    encoding="utf-8",
                )
                (model / "points3D.txt").write_text(
                    "1 0 0 1 1 2 3 0.1 1 0\n",
                    encoding="utf-8",
                )

            (smaller / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 one.jpg\n0 0 1\n",
                encoding="utf-8",
            )
            (larger / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 one.jpg\n0 0 1\n"
                "2 1 0 0 0 1 0 0 1 two.jpg\n0 0 1\n",
                encoding="utf-8",
            )

            selected, _, images, _ = _load_largest_sparse_model(
                [str(smaller), str(larger)]
            )

            self.assertEqual(Path(selected), larger)
            self.assertEqual(len(images), 2)

    def test_cache_install_failure_restores_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "colmap"
            staged = cache / "staged"
            old_sparse = cache / "sparse"
            staged_sparse = staged / "sparse"
            old_sparse.mkdir(parents=True)
            staged_sparse.mkdir(parents=True)

            old_database = cache / "database.db"
            staged_database = staged / "database.db"
            old_database.write_text("old database", encoding="utf-8")
            staged_database.write_text("new database", encoding="utf-8")
            (old_sparse / "marker").write_text("old sparse", encoding="utf-8")
            (staged_sparse / "marker").write_text(
                "new sparse",
                encoding="utf-8",
            )

            from modules.colmap import api as api_module

            original_move = api_module.shutil.move

            def fail_when_installing_sparse(source, destination, *args, **kwargs):
                if Path(source) == staged_sparse:
                    raise OSError("simulated cache installation failure")
                return original_move(source, destination, *args, **kwargs)

            with mock.patch.object(
                api_module.shutil,
                "move",
                side_effect=fail_when_installing_sparse,
            ):
                with self.assertRaises(OSError):
                    _replace_cached_reconstruction(
                        staged_database=str(staged_database),
                        staged_sparse_dir=str(staged_sparse),
                        database_path=str(old_database),
                        sparse_dir=str(old_sparse),
                    )

            self.assertEqual(
                old_database.read_text(encoding="utf-8"),
                "old database",
            )
            self.assertEqual(
                (old_sparse / "marker").read_text(encoding="utf-8"),
                "old sparse",
            )


if __name__ == "__main__":
    unittest.main()
