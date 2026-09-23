from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests.helpers import write_fake_colmap_model
from vggsfm_engine.colmap import convert_points3d_to_ply


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clean_light_with_masks.py"
SPEC = importlib.util.spec_from_file_location("clean_light_with_masks", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class CleanupRulesTests(unittest.TestCase):
    def test_mask_polarity_threshold_is_128_inclusive(self) -> None:
        mask = np.asarray([[0, 127, 128, 255]], dtype=np.uint8)
        actual = module.foreground_values(
            mask, np.asarray([0, 0, 0, 0]), np.asarray([0, 1, 2, 3])
        )
        np.testing.assert_array_equal(actual, [False, False, True, True])

    def test_support_rule_requires_minimum_views_and_ceiling_agreement(self) -> None:
        foreground = np.asarray([2, 5, 5, 9, 8], dtype=np.uint16)
        usable = np.asarray([2, 5, 6, 10, 10], dtype=np.uint16)
        selected = module.select_by_support(
            foreground, usable, minimum_usable=6, agreement_threshold=0.9
        )
        np.testing.assert_array_equal(selected, [False, False, False, True, False])

    def test_projection_uses_positive_z_original_pixels_and_mask(self) -> None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction()
        reconstruction.add_camera(pycolmap.Camera.create(
            1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 50.0, 100, 100
        ))
        reconstruction.add_image(pycolmap.Image(
            name="official.jpg", keypoints=np.empty((0, 2)),
            cam_from_world=pycolmap.Rigid3d(), camera_id=1, id=1,
        ))
        reconstruction.register_image(1)
        xyz = np.asarray([
            [0.0, 0.0, 5.0],       # center: usable foreground
            [20.0, 0.0, 5.0],      # projected out of bounds
            [0.0, 0.0, -5.0],      # behind camera
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[50, 50] = 255
            Image.fromarray(mask).save(root / "source.png")
            foreground, usable, rows = module.projection_votes(
                reconstruction, np.asarray([1, 2, 3]), xyz,
                {"official.jpg": "source.jpg"}, root,
            )
        np.testing.assert_array_equal(usable, [1, 0, 0])
        np.testing.assert_array_equal(foreground, [1, 0, 0])
        self.assertEqual(rows[0]["positive_z_count"], 2)
        self.assertEqual(rows[0]["usable_in_bounds_count"], 1)

    def test_subset_is_exact_source_ply_record_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            write_fake_colmap_model(model)
            source = root / "source.ply"
            convert_points3d_to_ply(model / "points3D.bin", source)
            output = root / "subset.ply"
            metrics = module.write_subset_ply(
                model / "points3D.bin", source, {10}, output, expected_count=2
            )
            self.assertEqual(metrics["point_count"], 1)
            self.assertEqual(metrics["tracked_point_count"], 1)
            self.assertEqual(metrics["trackless_point_count"], 0)
            self.assertTrue(metrics["source_ply_records_copied_byte_identically"])
            body = output.read_bytes().split(b"end_header\n", 1)[1]
            source_body = source.read_bytes().split(b"end_header\n", 1)[1]
            self.assertEqual(body, source_body[:15])


if __name__ == "__main__":
    unittest.main()
