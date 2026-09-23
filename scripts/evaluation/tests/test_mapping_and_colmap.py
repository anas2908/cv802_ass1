from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from cv802_evaluation.colmap_source import CameraPose, load_binary_camera_poses
from cv802_evaluation.matching import (
    MappingError,
    load_official_name_map,
    match_camera_centers,
)


def pose(image_id: int, name: str, center: tuple[float, float, float]) -> CameraPose:
    return CameraPose(
        image_id=image_id,
        name=name,
        camera_id=1,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=tuple(-value for value in center),
        center=np.asarray(center, dtype=np.float64),
    )


class MappingTests(unittest.TestCase):
    def test_request_map_restores_nested_names_and_matches_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "official_image_name_map": [
                            {"official": "hash-a__same.jpg", "source": "images/cam-a/same.jpg"},
                            {"official": "hash-b__same.jpg", "source": "images/cam-b/same.jpg"},
                            {"official": "hash-c__other.jpg", "source": "images/video/other.jpg"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mapping, _ = load_official_name_map(request)

        e10 = (
            pose(1, "cam-a/same.jpg", (0.0, 0.0, 0.0)),
            pose(2, "cam-b/same.jpg", (1.0, 0.0, 0.0)),
            pose(3, "video/other.jpg", (0.0, 1.0, 0.0)),
            pose(4, "e10-only.jpg", (0.0, 0.0, 1.0)),
        )
        vgg = (
            pose(10, "hash-a__same.jpg", (2.0, 0.0, 0.0)),
            pose(11, "hash-b__same.jpg", (3.0, 0.0, 0.0)),
            pose(12, "hash-c__other.jpg", (2.0, 1.0, 0.0)),
        )
        matches = match_camera_centers(e10, vgg, mapping)

        self.assertEqual(len(matches.names), 3)
        self.assertEqual(matches.coverage()["fraction_of_vggsfm_registered"], 1.0)
        self.assertEqual(matches.e10_images_missing_from_vggsfm, ("e10-only.jpg",))
        self.assertEqual(matches.names[0][1], "cam-a/same.jpg")
        self.assertEqual(matches.names[1][1], "cam-b/same.jpg")

    def test_registered_vgg_name_missing_from_receipt_fails(self) -> None:
        e10 = tuple(pose(i, f"cam/{i}.jpg", (float(i), float(i % 2), 0.0)) for i in range(3))
        vgg = tuple(pose(i, f"official-{i}.jpg", (float(i), float(i % 2), 0.0)) for i in range(3))
        with self.assertRaises(MappingError):
            match_camera_centers(e10, vgg, {"official-0.jpg": "cam/0.jpg"})


class BinaryModelAdapterTests(unittest.TestCase):
    def test_actual_colmap_binary_pose_records_produce_camera_centers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            with (model / "cameras.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 1))
                stream.write(struct.pack("<IiQQ", 1, 1, 640, 480))
                stream.write(struct.pack("<4d", 500.0, 500.0, 320.0, 240.0))
            centers = [(1.0, 2.0, 3.0), (-1.0, 0.5, 4.0), (0.0, -2.0, 1.0)]
            with (model / "images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", len(centers)))
                for index, center in enumerate(centers, 1):
                    stream.write(
                        struct.pack(
                            "<I7dI",
                            index,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            -center[0],
                            -center[1],
                            -center[2],
                            1,
                        )
                    )
                    stream.write(f"cam/{index}.jpg".encode("utf-8") + b"\0")
                    stream.write(struct.pack("<Q", 0))
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))

            parsed = load_binary_camera_poses(model)

        self.assertEqual(len(parsed), 3)
        for record, expected in zip(parsed, centers):
            np.testing.assert_allclose(record.center, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
