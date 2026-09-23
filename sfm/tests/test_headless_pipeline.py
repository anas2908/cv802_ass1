import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from sfm_engine import cli, pipeline


class FakeDevice:
    cpu = "CPU"
    cuda = "CUDA"


class FakePycolmap:
    Device = FakeDevice
    has_cuda = True


class _FakeCamera:
    width = 1920
    height = 1080
    params = [1200.0, 960.0, 540.0, 0.01]

    @staticmethod
    def verify_params():
        return True


class _FakeRotation:
    @staticmethod
    def matrix():
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class _FakePose:
    translation = [0.0, 0.0, 0.0]
    rotation = _FakeRotation()


class _FakeImage:
    def __init__(self, name):
        self.name = name
        self.camera_id = 1

    @staticmethod
    def cam_from_world():
        return _FakePose()


class _FakePoint:
    xyz = [1.0, 2.0, 3.0]
    color = [10, 20, 30]
    error = 0.25


class _FakeReconstruction:
    def __init__(self, owner, names=("a.jpg", "b.jpg")):
        self.owner = owner
        self.cameras = {1: _FakeCamera()}
        self.images = {
            index: _FakeImage(name) for index, name in enumerate(names, start=1)
        }
        self.points3D = {1: _FakePoint()}

    @staticmethod
    def is_valid():
        return True

    def reg_image_ids(self):
        return list(self.images)

    @staticmethod
    def compute_mean_reprojection_error():
        return 0.5

    @staticmethod
    def compute_mean_track_length():
        return 2.0

    def write(self, path):
        self.owner.install_model(Path(path), self)

    def export_PLY(self, path):
        Path(path).write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
            "1 2 3 10 20 30\n",
            encoding="ascii",
        )


class RecordingPycolmap(types.ModuleType):
    """Small standard-library-only contract fake for PyCOLMAP 4.2."""

    def __init__(self):
        super().__init__("pycolmap")
        self.__version__ = "4.2.0"
        self.has_cuda = False
        self.Device = types.SimpleNamespace(cpu="CPU", cuda="CUDA")
        self.CameraMode = types.SimpleNamespace(
            AUTO="AUTO", PER_FOLDER="PER_FOLDER", SINGLE="SINGLE"
        )
        self.events = []
        self.models = {}
        self.match_failures = 0
        self.mutate_during_extract = None
        self.Reconstruction = self.load_model

    def get_num_cuda_devices(self):
        return 0

    def set_random_seed(self, seed):
        self.random_seed = seed

    def install_model(self, path, reconstruction):
        path.mkdir(parents=True, exist_ok=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (path / name).write_bytes(name.encode("ascii"))
        self.models[str(path.resolve())] = reconstruction

    def load_model(self, path):
        resolved = Path(path).resolve()
        known = self.models.get(str(resolved))
        if known is not None:
            return known
        # The production pipeline atomically renames a validated staging tree;
        # a real Reconstruction reloads its bytes at the new path. Mirror that
        # file-backed behaviour instead of tying the fake to a pathname.
        if all(
            (resolved / name).is_file()
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        ):
            known = _FakeReconstruction(self)
            self.models[str(resolved)] = known
            return known
        raise KeyError(str(resolved))

    def extract_features(self, **kwargs):
        self.events.append(("extract", kwargs))
        Path(kwargs["database_path"]).write_bytes(b"sqlite")
        if self.mutate_during_extract is not None:
            self.mutate_during_extract.write_bytes(b"changed-during-run")

    def match_exhaustive(self, **kwargs):
        self.events.append(("exhaustive", kwargs))
        if self.match_failures:
            self.match_failures -= 1
            raise RuntimeError("injected matching failure")

    def match_sequential(self, **kwargs):
        self.events.append(("sequential", kwargs))

    def incremental_mapping(self, **kwargs):
        self.events.append(("mapping", kwargs))
        reconstruction = _FakeReconstruction(self)
        self.install_model(Path(kwargs["output_path"]) / "0", reconstruction)
        return {0: reconstruction}


class HeadlessPipelineTests(unittest.TestCase):
    def _case(self, temporary, *, matcher="exhaustive"):
        data_root = Path(temporary) / "data"
        image_root = data_root / "sfm" / "inputs" / "light" / "images"
        image_root.mkdir(parents=True)
        (image_root / "a.jpg").write_bytes(b"image-a")
        (image_root / "b.jpg").write_bytes(b"image-b")
        config = pipeline.SfMConfig(
            experiment_name="test-run",
            image_dir=image_root,
            matcher=matcher,
            device="cpu",
            num_threads=3,
        )
        return data_root, image_root, config

    def test_require_under_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "must be below"):
                pipeline.require_under(Path(temporary) / "elsewhere", root, "test")

    def test_discover_images_is_recursive_sorted_and_ignores_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            image_root = data_root / "sfm" / "inputs" / "scene" / "images"
            (image_root / "b").mkdir(parents=True)
            (image_root / "z.JPG").write_bytes(b"z")
            (image_root / "b" / "a.png").write_bytes(b"a")
            (image_root / "notes.txt").write_text("not an image")
            (image_root / "link.jpg").symlink_to(image_root / "z.JPG")
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"):
                found = pipeline.discover_images(image_root)
            self.assertEqual(
                [path.relative_to(image_root).as_posix() for path in found],
                ["b/a.png", "z.JPG"],
            )

    def test_manifest_hashes_file_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "a.jpg"
            image.write_bytes(b"pixels")
            record = pipeline.image_manifest(root, [image])[0]
            self.assertEqual(record["path"], "a.jpg")
            self.assertEqual(record["bytes"], 6)
            self.assertEqual(
                record["sha256"],
                "6ec9c2b0eb14010746c8bce8939303b382344b296206612eb8a907a37b2b2f37",
            )

    def test_cuda_selection_fails_closed(self):
        fake = FakePycolmap()
        fake.has_cuda = False
        with self.assertRaisesRegex(RuntimeError, "has_cuda=False"):
            pipeline.resolve_device(fake, "cuda")

    def test_cuda_selection_uses_reported_cuda(self):
        name, value = pipeline.resolve_device(FakePycolmap(), "cuda")
        self.assertEqual((name, value), ("cuda", "CUDA"))

    def test_cuda_identity_requires_one_device_and_records_hardware(self):
        fake = FakePycolmap()
        fake.get_num_cuda_devices = lambda: 1
        completed = types.SimpleNamespace(
            returncode=0,
            stdout="0, GPU-abc, NVIDIA A100-SXM4-40GB, 570.1, 40960\n",
            stderr="",
        )
        runner = mock.Mock(return_value=completed)
        with mock.patch.object(pipeline, "DATA_ROOT", Path("/tmp/test-data")), mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "3"}, clear=False
        ):
            identity = pipeline.cuda_runtime_identity(fake, runner=runner)
        self.assertEqual(identity["pycolmap_visible_cuda_devices"], 1)
        self.assertEqual(identity["nvidia_smi_inventory"][0]["uuid"], "GPU-abc")
        self.assertEqual(identity["nvidia_smi_inventory"][0]["memory_mib"], 40960)
        self.assertEqual(identity["selected_gpu"]["name"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual(
            runner.call_args.args[0],
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
        )

        fake.get_num_cuda_devices = lambda: 2
        with self.assertRaisesRegex(RuntimeError, "exactly one visible GPU"):
            pipeline.cuda_runtime_identity(fake, runner=runner)

    def test_full_fake_pipeline_matches_pycolmap_42_contract_and_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                first = pipeline.run_reconstruction(config)
                second = pipeline.run_reconstruction(config)

            self.assertTrue(first["complete"])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(
                [name for name, _kwargs in fake.events],
                ["extract", "exhaustive", "mapping"],
            )
            extraction = fake.events[0][1]
            self.assertEqual(extraction["camera_mode"], "AUTO")
            self.assertEqual(extraction["reader_options"], {"camera_model": "SIMPLE_RADIAL"})
            self.assertEqual(extraction["extraction_options"]["num_threads"], 3)
            self.assertEqual(extraction["extraction_options"]["sift"]["first_octave"], 0)
            self.assertEqual(extraction["device"], "CPU")
            matching = fake.events[1][1]
            self.assertFalse(matching["matching_options"]["guided_matching"])
            mapping = fake.events[2][1]
            self.assertEqual(
                mapping["options"],
                {"num_threads": 3, "extract_colors": True, "random_seed": 0},
            )
            self.assertEqual(fake.random_seed, 0)
            self.assertEqual(first["metrics"]["registered_image_names"], ["a.jpg", "b.jpg"])
            self.assertEqual(first["ply"]["vertices"], 1)

    def test_sequential_options_match_pycolmap_42_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary, matcher="sequential")
            fake = RecordingPycolmap()
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                pipeline.run_reconstruction(config)
            call = next(kwargs for name, kwargs in fake.events if name == "sequential")
            self.assertEqual(call["pairing_options"], {"overlap": 10, "loop_detection": False})

    def test_interrupted_run_uses_fresh_attempt_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            fake.match_failures = 1
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                with self.assertRaisesRegex(RuntimeError, "injected matching failure"):
                    pipeline.run_reconstruction(config)
                result = pipeline.run_reconstruction(config, resume=True)

            attempts = data_root / "sfm" / "experiments" / "test-run" / "attempts"
            first_state = json.loads((attempts / "0001" / "status.json").read_text())
            second_state = json.loads((attempts / "0002" / "status.json").read_text())
            self.assertEqual(first_state["stage"], "failed")
            self.assertEqual(second_state["stage"], "complete")
            databases = [
                kwargs["database_path"]
                for name, kwargs in fake.events
                if name == "extract"
            ]
            self.assertEqual(len(databases), 2)
            self.assertNotEqual(databases[0], databases[1])
            self.assertEqual(result["attempt_id"], "0002")

    def test_no_resume_rejects_incomplete_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            fake.match_failures = 1
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                with self.assertRaises(RuntimeError):
                    pipeline.run_reconstruction(config)
                with self.assertRaisesRegex(FileExistsError, "requires --resume"):
                    pipeline.run_reconstruction(config, resume=False)
            self.assertEqual(sum(name == "extract" for name, _kwargs in fake.events), 1)

    def test_output_without_receipt_is_recovered_without_recomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            receipt = data_root / "sfm" / "experiments" / "test-run" / "receipt.json"
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                pipeline.run_reconstruction(config)
                receipt.unlink()
                recovered = pipeline.run_reconstruction(config)
            self.assertTrue(recovered["recovered_after_interruption"])
            self.assertEqual(sum(name == "extract" for name, _kwargs in fake.events), 1)

    def test_cache_rejects_tampered_receipt_and_missing_ply(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            experiment = data_root / "sfm" / "experiments" / "test-run"
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                pipeline.run_reconstruction(config)
                receipt_path = experiment / "receipt.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["metrics"]["points3D"] = 999
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(RuntimeError, "receipt and saved artifacts disagree"):
                    pipeline.run_reconstruction(config)
                receipt["metrics"]["points3D"] = 1
                receipt_path.write_text(json.dumps(receipt))
                camera_file = experiment / "outputs" / "colmap" / "sparse" / "0" / "cameras.bin"
                original_camera = camera_file.read_bytes()
                camera_file.write_bytes(original_camera + b"tampered")
                with self.assertRaisesRegex(RuntimeError, "receipt and saved artifacts disagree"):
                    pipeline.run_reconstruction(config)
                camera_file.write_bytes(original_camera)
                (experiment / "outputs" / "sparse_colored.ply").unlink()
                with self.assertRaisesRegex(RuntimeError, "missing or empty coloured PLY"):
                    pipeline.run_reconstruction(config)

    def test_input_change_during_run_never_publishes_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            fake.mutate_during_extract = image_root / "a.jpg"
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                with self.assertRaisesRegex(RuntimeError, "input images changed"):
                    pipeline.run_reconstruction(config)
            experiment = data_root / "sfm" / "experiments" / "test-run"
            self.assertFalse((experiment / "outputs").exists())
            self.assertFalse((experiment / "receipt.json").exists())

    def test_experiment_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            sfm_root = Path(temporary) / "sfm"
            experiment = sfm_root / "experiments" / "experiment"
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", sfm_root):
                with pipeline._experiment_lock(experiment):
                    with self.assertRaisesRegex(RuntimeError, "another process"):
                        with pipeline._experiment_lock(experiment):
                            self.fail("second lock unexpectedly succeeded")

    def test_symlinked_output_escape_is_rejected_before_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            outside = Path(temporary) / "outside"
            outside.mkdir()
            fake = RecordingPycolmap()
            fake.match_failures = 1
            experiment = data_root / "sfm" / "experiments" / "test-run"
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                with self.assertRaises(RuntimeError):
                    pipeline.run_reconstruction(config)
                (experiment / "outputs").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "must be below"):
                    pipeline.run_reconstruction(config)
            self.assertEqual(sum(name == "extract" for name, _kwargs in fake.events), 1)

    def test_geometric_validation_rejects_internal_and_camera_errors(self):
        fake = RecordingPycolmap()
        reconstruction = _FakeReconstruction(fake)
        reconstruction.is_valid = lambda: False
        with self.assertRaisesRegex(ValueError, "internally invalid"):
            pipeline.reconstruction_metrics(reconstruction)

        reconstruction = _FakeReconstruction(fake)
        reconstruction.cameras = {
            1: types.SimpleNamespace(
                width=1920,
                height=1080,
                params=[float("nan"), 960.0, 540.0],
                verify_params=lambda: True,
            )
        }
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pipeline.reconstruction_metrics(reconstruction)

        reconstruction = _FakeReconstruction(fake)
        reconstruction.images[1].camera_id = 999
        with self.assertRaisesRegex(ValueError, "missing camera"):
            pipeline.reconstruction_metrics(reconstruction)

    def test_experiment_name_cannot_create_nested_or_hidden_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, image_root, _config = self._case(temporary)
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"):
                for name in ("../escape", "nested/name", r"nested\name", ".hidden"):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        pipeline.SfMConfig(name, image_root, device="cpu").validate()

    def test_cli_defaults_to_portable_auto_device(self):
        arguments = cli.parser().parse_args(
            ["run", "--experiment", "portable", "--images", "/tmp/images"]
        )
        self.assertEqual(arguments.device, "auto")
        self.assertEqual(arguments.random_seed, 0)

    def test_wrong_pycolmap_version_fails_before_any_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, _image_root, config = self._case(temporary)
            fake = RecordingPycolmap()
            fake.__version__ = "4.3.0"
            with mock.patch.object(pipeline, "SFM_DATA_ROOT", data_root / "sfm"), mock.patch.dict(
                sys.modules, {"pycolmap": fake}
            ):
                with self.assertRaisesRegex(RuntimeError, "validated for pycolmap==4.2.0"):
                    pipeline.run_reconstruction(config)
            self.assertEqual(fake.events, [])


if __name__ == "__main__":
    unittest.main()
