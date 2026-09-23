from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cv802_mvs.commands import build_commands, set_auto_source_count
from cv802_mvs.config import MVSConfig
from cv802_mvs.errors import ConfigurationError, StoragePolicyError
from cv802_mvs.paths import PathPolicy


TEST_TEMP = Path("/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests")


def raw_config(data_root: Path, *, backend: str = "native") -> dict:
    colmap = {"backend": backend, "binary": "colmap", "gpu_index": "0"}
    if backend == "singularity":
        colmap.update(
            {
                "singularity_binary": "/usr/bin/singularity",
                "image": "dependencies/colmap.sif",
            }
        )
    elif backend == "pycolmap":
        colmap["python"] = "envs/mvs-engine/bin/python"
    return {
        "schema_version": 1,
        "data_root": str(data_root / "mvs"),
        "experiment": "pilot",
        "inputs": {"images": "inputs/images", "sparse_model": "inputs/sparse"},
        "masking": {"mode": "none"},
        "colmap": colmap,
        "max_image_size": 1600,
        "patch_match": {"geom_consistency": True, "source_images_per_view": 10},
        "fusion": {},
        "meshing": "none",
        "execution": {"num_threads": 8, "require_slurm_job": True},
    }


class PathsConfigCommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        self.root = Path(self.temporary.name)
        self.policy = PathPolicy.for_tests(self.root)
        self.policy.method_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rejects_path_escape_and_bad_experiment(self) -> None:
        with self.assertRaises(StoragePolicyError):
            self.policy.require_method(self.root / "sfm", "wrong method")
        with self.assertRaises(StoragePolicyError):
            self.policy.experiment("../escape")

    def test_config_rejects_wrong_storage_root(self) -> None:
        raw = raw_config(self.root)
        raw["data_root"] = "/home/example/heavy-data"
        with self.assertRaises(ConfigurationError):
            MVSConfig.parse(raw, self.policy)

    def test_numeric_boolean_and_single_gpu_in_commands(self) -> None:
        config = MVSConfig.parse(raw_config(self.root), self.policy)
        commands = {command.name: command for command in build_commands(config, config.paths(self.policy))}
        patch = commands["patch_match"].argv
        option = patch.index("--PatchMatchStereo.geom_consistency")
        self.assertEqual(patch[option + 1], "1")
        gpu = patch.index("--PatchMatchStereo.gpu_index")
        self.assertEqual(patch[gpu + 1], "0")
        self.assertNotIn("true", patch)

    def test_singularity_command_uses_nv_and_only_mvs_bind(self) -> None:
        config = MVSConfig.parse(raw_config(self.root, backend="singularity"), self.policy)
        command = build_commands(config, config.paths(self.policy))[0]
        self.assertEqual(command.argv[0], "/usr/bin/singularity")
        self.assertIn("--nv", command.argv)
        bind = command.argv[command.argv.index("--bind") + 1]
        self.assertEqual(bind, f"{self.policy.method_root}:{self.policy.method_root}")
        self.assertEqual(command.subcommand, "image_undistorter")

    def test_pycolmap_command_uses_data_root_python_and_source_worker(self) -> None:
        config = MVSConfig.parse(raw_config(self.root, backend="pycolmap"), self.policy)
        command = build_commands(config, config.paths(self.policy))[0]
        self.assertEqual(
            command.argv[0], str(self.policy.method_root / "envs/mvs-engine/bin/python")
        )
        self.assertEqual(command.argv[1], "-B")
        self.assertTrue(command.argv[2].endswith("mvs/run_pycolmap_stage.py"))
        self.assertEqual(command.subcommand, "image_undistorter")
        self.assertIn("--num_patch_match_src_images", command.argv)
        self.assertIn("--num_threads", command.argv)

    def test_patch_match_source_rewrite(self) -> None:
        path = self.root / "patch-match.cfg"
        path.write_text("a.jpg\n__auto__, 20\nb.jpg\na.jpg\n", encoding="utf-8")
        self.assertEqual(set_auto_source_count(path, 7), 2)
        self.assertEqual(
            path.read_text(encoding="utf-8"),
            "a.jpg\n__auto__, 7\nb.jpg\n__auto__, 7\n",
        )


if __name__ == "__main__":
    unittest.main()
