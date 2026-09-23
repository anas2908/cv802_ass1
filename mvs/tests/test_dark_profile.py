"""Prevent accidental body-only masking or quality relabelling of the dark pilot."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cv802_mvs.config import MVSConfig
from cv802_mvs.paths import PathPolicy
from cv802_mvs.runner import MVSRunner
from test_paths_config_commands import raw_config


class DarkPilotProfileTests(unittest.TestCase):
    def test_profile_is_separate_raw_all_view_resource_pilot(self):
        path = Path(__file__).resolve().parents[1] / "configs/dark_e3_1024_raw_pycolmap.json"
        config = MVSConfig.load(path, PathPolicy.production())
        self.assertEqual(config.experiment, "dark_e3_colmap_mvs_1024_raw_v1")
        self.assertEqual(config.masking_mode, "none")
        self.assertIsNone(config.masks_relative)
        self.assertEqual(config.max_image_size, 1024)
        self.assertEqual(config.source_images_per_view, 5)
        self.assertEqual(config.patch_num_iterations, 3)
        self.assertEqual(config.num_threads, 4)
        self.assertTrue(config.geom_consistency)
        self.assertTrue(config.require_slurm_job)
        self.assertEqual(config.fusion_min_num_pixels, 5)
        self.assertEqual(config.colmap_backend, "pycolmap")

    def test_unmasked_run_creates_undistorter_parent_without_mask_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = PathPolicy.for_tests(root)
            config = MVSConfig.parse(raw_config(root), policy)
            self.assertEqual(config.masking_mode, "none")
            runner = MVSRunner(config, policy)
            (runner.paths.root / "inputs").mkdir(parents=True)
            self.assertFalse(runner.paths.work.exists())

            def stop_before_colmap(command, **kwargs):
                self.assertEqual(command.name, "undistort")
                self.assertTrue(runner.paths.workspace.parent.is_dir())
                raise RuntimeError("verified precondition; no COLMAP execution")

            with patch("cv802_mvs.runner.validate_inputs", return_value={}), \
                 patch("cv802_mvs.runner.validate_runtime", return_value=({}, {})), \
                 patch.object(runner, "_run_mask_stage", side_effect=AssertionError("unmasked")), \
                 patch.object(runner, "_run_command_stage", side_effect=stop_before_colmap):
                with self.assertRaisesRegex(RuntimeError, "verified precondition"):
                    runner.run()


if __name__ == "__main__":
    unittest.main()
