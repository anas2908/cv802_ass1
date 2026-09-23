from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest

from cv802_mvs.pycolmap_backend import build_parser, main, run_stage


TEST_TEMP = Path("/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests")


class _Options:
    pass


class _FakePycolmap:
    __version__ = "4.2.0"
    has_cuda = True
    UndistortCameraOptions = _Options
    PatchMatchOptions = _Options
    StereoFusionOptions = _Options

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def undistort_images(self, *args, **kwargs) -> None:
        self.calls.append(("undistort", args, kwargs))

    def patch_match_stereo(self, *args, **kwargs) -> None:
        self.calls.append(("patch_match", args, kwargs))

    def stereo_fusion(self, *args, **kwargs) -> None:
        self.calls.append(("fusion", args, kwargs))

    def poisson_meshing(self, *args, **kwargs) -> None:
        self.calls.append(("poisson", args, kwargs))


class PycolmapBackendTest(unittest.TestCase):
    def test_probe_requires_pinned_cuda_dense_api(self) -> None:
        fake = _FakePycolmap()
        result = run_stage(build_parser().parse_args(["probe"]), fake)
        self.assertEqual(result["pycolmap_version"], "4.2.0")
        self.assertTrue(result["pycolmap_has_cuda"])
        self.assertTrue(result["dense_api"])

    def test_patch_match_translates_every_configured_option(self) -> None:
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            fake = _FakePycolmap()
            argv = [
                "patch_match_stereo",
                "--workspace_path", directory,
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.gpu_index", "0",
                "--PatchMatchStereo.geom_consistency", "1",
                "--PatchMatchStereo.filter", "1",
                "--PatchMatchStereo.cache_size", "16",
                "--PatchMatchStereo.max_image_size", "1600",
                "--PatchMatchStereo.num_iterations", "5",
                "--PatchMatchStereo.num_samples", "15",
                "--PatchMatchStereo.window_radius", "5",
                "--PatchMatchStereo.window_step", "1",
                "--PatchMatchStereo.filter_min_num_consistent", "2",
                "--PatchMatchStereo.num_threads", "16",
            ]
            result = run_stage(build_parser().parse_args(argv), fake)
            self.assertEqual(result["stage"], "patch_match_stereo")
            options = fake.calls[0][2]["options"]
            self.assertEqual(options.gpu_index, "0")
            self.assertTrue(options.geom_consistency)
            self.assertEqual(options.cache_size, 16.0)
            self.assertEqual(options.num_threads, 16)

    def test_wrong_version_is_rejected(self) -> None:
        fake = _FakePycolmap()
        fake.__version__ = "4.1.0"
        self.assertEqual(main(["probe"], pycolmap_module=fake), 2)


if __name__ == "__main__":
    unittest.main()
