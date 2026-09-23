import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.colmap.model_io import (
    camera_to_pinhole_intrinsics,
    qvec_to_rotation_matrix,
    read_text_model,
)


class ModelIOTest(unittest.TestCase):
    def test_reads_text_model_and_preserves_image_names(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "cameras.txt").write_text(
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
                "1 SIMPLE_PINHOLE 640 480 500 320 240\n",
                encoding="utf-8",
            )
            (model_dir / "images.txt").write_text(
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                "3 1 0 0 0 1 2 3 1 image with spaces.jpg\n"
                "10.0 20.0 -1\n",
                encoding="utf-8",
            )
            (model_dir / "points3D.txt").write_text(
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                "9 1.5 2.5 3.5 10 20 30 0.25 3 0\n",
                encoding="utf-8",
            )

            cameras, images, points3d = read_text_model(model_dir)

            self.assertEqual(images[3]["name"], "image with spaces.jpg")
            np.testing.assert_allclose(images[3]["tvec"], [1, 2, 3])
            np.testing.assert_allclose(points3d[9]["xyz"], [1.5, 2.5, 3.5])
            np.testing.assert_array_equal(points3d[9]["rgb"], [10, 20, 30])
            self.assertEqual(cameras[1]["model"], "SIMPLE_PINHOLE")

    def test_converts_camera_models_to_pinhole_fields(self):
        simple = {
            "model": "SIMPLE_RADIAL",
            "width": 640,
            "height": 480,
            "params": np.array([500, 320, 240, 0.01]),
        }
        opencv = {
            "model": "OPENCV",
            "width": 1920,
            "height": 1080,
            "params": np.array([1000, 1010, 960, 540, 0, 0, 0, 0]),
        }

        self.assertEqual(
            camera_to_pinhole_intrinsics(simple),
            {
                "width": 640,
                "height": 480,
                "fx": 500.0,
                "fy": 500.0,
                "cx": 320.0,
                "cy": 240.0,
            },
        )
        self.assertEqual(camera_to_pinhole_intrinsics(opencv)["fy"], 1010.0)

    def test_converts_colmap_quaternion_to_rotation(self):
        half_angle = np.pi / 4
        qvec = [np.cos(half_angle), 0, 0, np.sin(half_angle)]

        rotation = qvec_to_rotation_matrix(qvec)

        np.testing.assert_allclose(
            rotation,
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
