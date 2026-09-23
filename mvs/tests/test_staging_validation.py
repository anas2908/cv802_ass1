from __future__ import annotations

from pathlib import Path
import hashlib
import json
import struct
import tempfile
import unittest

from cv802_mvs.paths import PathPolicy
from cv802_mvs.staging import stage_inputs
from cv802_mvs.validation import validate_colored_ply


TEST_TEMP = Path("/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests")


class StagingValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        self.root = Path(self.temporary.name)
        self.policy = PathPolicy.for_tests(self.root)
        self.policy.method_root.mkdir(parents=True)
        source = self.root / "sfm"
        actual_images = source / "actual-images"
        actual_images.mkdir(parents=True)
        (actual_images / "a.jpg").write_bytes(b"first image")
        (actual_images / "b.jpg").write_bytes(b"second image")
        (source / "images").symlink_to(actual_images, target_is_directory=True)
        model = source / "model"
        model.mkdir()
        (model / "cameras.txt").write_text(
            "1 PINHOLE 10 20 8 8 5 10\n", encoding="utf-8"
        )
        (model / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 a.jpg\n\n"
            "2 1 0 0 0 1 0 0 1 b.jpg\n\n",
            encoding="utf-8",
        )
        (model / "points3D.txt").write_text("# empty is valid for staging audit\n", encoding="utf-8")
        self.images = source / "images"
        self.model = model

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stage_resolves_symlink_and_resume_verifies_hashes(self) -> None:
        manifest = stage_inputs(
            policy=self.policy,
            experiment="pilot",
            images_source=self.images,
            model_source=self.model,
        )
        destination = self.policy.experiment("pilot") / "inputs" / "images" / "a.jpg"
        self.assertTrue(destination.is_file())
        self.assertFalse(destination.is_symlink())
        self.assertEqual(manifest["registered_image_count"], 2)
        resumed = stage_inputs(
            policy=self.policy,
            experiment="pilot",
            images_source=self.images,
            model_source=self.model,
            resume=True,
        )
        self.assertEqual(resumed["files_digest"], manifest["files_digest"])

    def test_validates_all_vertices_in_colored_ascii_ply(self) -> None:
        ply = self.policy.method_root / "cloud.ply"
        ply.write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n0 1 2 10 20 30\n3 4 5 40 50 60\n",
            encoding="ascii",
        )
        report = validate_colored_ply(ply)
        self.assertEqual(report["vertex_count"], 2)
        self.assertEqual(report["bounds_max_xyz"], [3.0, 4.0, 5.0])

    def test_stages_and_rebases_mask_bundle(self) -> None:
        source = self.root / "masked-source"
        images = source / "images"
        masks = source / "masks"
        model = source / "model"
        images.mkdir(parents=True)
        masks.mkdir()
        model.mkdir()
        # Dimension audit needs only the complete PNG signature and IHDR fields.
        png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 10, 20)
        for name in ("a", "b"):
            (images / f"{name}.png").write_bytes(png)
            (masks / f"{name}.png").write_bytes(png + name.encode())
        (model / "cameras.txt").write_text(
            "1 PINHOLE 10 20 8 8 5 10\n", encoding="utf-8"
        )
        (model / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 a.png\n\n"
            "2 1 0 0 0 1 0 0 1 b.png\n\n",
            encoding="utf-8",
        )
        (model / "points3D.txt").write_text("# none\n", encoding="utf-8")
        records = {}
        for name in ("a", "b"):
            path = masks / f"{name}.png"
            records[f"{name}.png"] = {
                "usable": True,
                "status": "valid",
                "mask_relative_path": f"masks/{name}.png",
                "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = source / "mask_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "complete": True,
                    "method": {
                        "value_semantics": "high means foreground",
                        "pixel_coordinates": "distorted native pixels",
                    },
                    "images": records,
                }
            ),
            encoding="utf-8",
        )
        stage_inputs(
            policy=self.policy,
            experiment="masked",
            images_source=images,
            model_source=model,
            masks_source=masks,
            mask_manifest_source=manifest,
        )
        staged = json.loads(
            (self.policy.experiment("masked") / "inputs" / "mask_manifest.json").read_text()
        )
        self.assertTrue(staged["ready_for_dense_input_masking"])
        self.assertEqual(set(staged["images"]), {"a.png", "b.png"})


if __name__ == "__main__":
    unittest.main()
