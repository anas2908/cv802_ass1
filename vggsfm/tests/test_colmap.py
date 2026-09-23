from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path

from vggsfm_engine.colmap import (
    camera_count,
    convert_points3d_to_ply,
    discover_model,
    registered_image_count,
    validate_binary_model,
)
from vggsfm_engine.errors import OutputValidationError

from tests.helpers import write_fake_colmap_model


class ColmapConversionTests(unittest.TestCase):
    def test_discovers_direct_sparse_and_converts_rgb_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "scene"
            model = scene / "sparse"
            write_fake_colmap_model(model)

            self.assertEqual(discover_model(scene), model)
            self.assertEqual(camera_count(model / "cameras.bin"), 1)
            self.assertEqual(registered_image_count(model / "images.bin"), 3)
            validation = validate_binary_model(model)
            self.assertEqual(validation["point_count"], 2)
            self.assertEqual(
                validation["image_names"],
                ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg"],
            )
            self.assertEqual(validation["image_name_to_id"]["frame_001.jpg"], 2)
            ply = Path(temporary) / "cloud.ply"
            metrics = convert_points3d_to_ply(model / "points3D.bin", ply)
            self.assertEqual(metrics["point_count"], 2)
            self.assertEqual(metrics["tracked_point_count"], 2)
            self.assertEqual(metrics["trackless_point_count"], 0)
            self.assertAlmostEqual(metrics["tracked_mean_reprojection_error_px"], 0.5)
            self.assertTrue(metrics["has_rgb"])

            payload = ply.read_bytes()
            marker = b"end_header\n"
            header, vertices = payload.split(marker, 1)
            self.assertIn(b"element vertex 2", header)
            self.assertEqual(len(vertices), 2 * struct.calcsize("<fffBBB"))
            first = struct.unpack("<fffBBB", vertices[: struct.calcsize("<fffBBB")])
            self.assertAlmostEqual(first[0], 1.25)
            self.assertAlmostEqual(first[1], -2.5)
            self.assertEqual(first[3:], (255, 2, 3))

    def test_discovers_zero_subdirectory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "scene"
            model = scene / "sparse" / "0"
            write_fake_colmap_model(model)
            self.assertEqual(discover_model(scene), model)

    def test_accepts_legitimate_zero_camera_image_and_point_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model, zero_based_ids=True)

            validation = validate_binary_model(model)
            self.assertEqual(validation["cameras"][0]["camera_id"], 0)
            self.assertEqual(validation["images"][0]["image_id"], 0)
            self.assertEqual(validation["images"][0]["camera_id"], 0)
            self.assertEqual(validation["image_name_to_id"]["frame_000.jpg"], 0)
            self.assertEqual(validation["point_count"], 2)
            metrics = convert_points3d_to_ply(
                model / "points3D.bin", Path(temporary) / "zero-id-cloud.ply"
            )
            self.assertEqual(metrics["point_count"], 2)

    def test_rejects_negative_and_reserved_ids_without_rejecting_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model)

            cameras = bytearray(model.joinpath("cameras.bin").read_bytes())
            struct.pack_into("<i", cameras, 8, -1)
            model.joinpath("cameras.bin").write_bytes(cameras)
            with self.assertRaisesRegex(OutputValidationError, "negative ID -1"):
                validate_binary_model(model)

            write_fake_colmap_model(model)
            points_path = model / "points3D.bin"
            points = bytearray(points_path.read_bytes())
            struct.pack_into("<Q", points, 8, (1 << 64) - 1)
            points_path.write_bytes(points)
            with self.assertRaisesRegex(OutputValidationError, "reserved invalid ID"):
                validate_binary_model(model)

            write_fake_colmap_model(model)
            points = bytearray(points_path.read_bytes())
            first_track_offset = 8 + struct.calcsize("<QdddBBBd") + 8
            struct.pack_into("<I", points, first_track_offset, (1 << 32) - 1)
            points_path.write_bytes(points)
            with self.assertRaisesRegex(
                OutputValidationError, "reserved invalid image ID"
            ):
                validate_binary_model(model)

    def test_rejects_invalid_camera_geometry_and_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model)
            model.joinpath("cameras.bin").write_bytes(
                struct.pack("<QiiQQddd", 1, 1, 0, 640, 480, -1.0, 320.0, 240.0)
            )
            with self.assertRaisesRegex(OutputValidationError, "focal length"):
                validate_binary_model(model)

            model.joinpath("cameras.bin").write_bytes(
                struct.pack(
                    "<QiiQQddd", 1, 1, 0, 640, 480, 500.0, math.nan, 240.0
                )
            )
            with self.assertRaisesRegex(OutputValidationError, "non-finite intrinsic"):
                validate_binary_model(model)

            model.joinpath("cameras.bin").write_bytes(
                struct.pack("<QiiQQddd", 1, 1, 0, 640, 480, 500.0, 320.0, 240.0)
                + b"x"
            )
            with self.assertRaisesRegex(OutputValidationError, "trailing bytes"):
                validate_binary_model(model)

    def test_rejects_non_normalized_pose_and_missing_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model, image_count=1)
            with model.joinpath("images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 1))
                stream.write(
                    struct.pack(
                        "<idddddddi", 1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1
                    )
                )
                stream.write(b"frame.jpg\x00" + struct.pack("<Q", 0))
            with self.assertRaisesRegex(OutputValidationError, "quaternion norm"):
                validate_binary_model(model)

            with model.joinpath("images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 1))
                stream.write(
                    struct.pack(
                        "<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99
                    )
                )
                stream.write(b"frame.jpg\x00" + struct.pack("<Q", 0))
            with self.assertRaisesRegex(OutputValidationError, "missing camera ID 99"):
                validate_binary_model(model)

    def test_rejects_duplicate_image_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model)
            with model.joinpath("images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 2))
                for image_id in (1, 2):
                    stream.write(
                        struct.pack(
                            "<idddddddi",
                            image_id,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1,
                        )
                    )
                    stream.write(b"same.jpg\x00" + struct.pack("<Q", 0))
            with self.assertRaisesRegex(OutputValidationError, "duplicate image name"):
                validate_binary_model(model)

    def test_rejects_nonreciprocal_point_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model)
            payload = bytearray(model.joinpath("points3D.bin").read_bytes())
            # First track entry is (image=1, point2D=0); changing its point2D
            # index makes it disagree with the corresponding image record.
            first_track_offset = 8 + struct.calcsize("<QdddBBBd") + 8
            struct.pack_into("<II", payload, first_track_offset, 1, 1)
            model.joinpath("points3D.bin").write_bytes(payload)
            with self.assertRaisesRegex(OutputValidationError, "track disagrees"):
                validate_binary_model(model)

    def test_rejects_declared_count_beyond_file_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model)
            model.joinpath("images.bin").write_bytes(struct.pack("<Q", 10_000))
            with self.assertRaisesRegex(OutputValidationError, "cannot fit"):
                registered_image_count(model / "images.bin")

    def test_trackless_extra_is_valid_and_excluded_from_quality_means(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model)
            points_path = model / "points3D.bin"
            payload = bytearray(points_path.read_bytes())
            struct.pack_into("<Q", payload, 0, 3)
            payload.extend(
                struct.pack(
                    "<QdddBBBdQ",
                    12,
                    7.0,
                    8.0,
                    9.0,
                    8,
                    9,
                    10,
                    -1.0,
                    0,
                )
            )
            points_path.write_bytes(payload)

            validation = validate_binary_model(model)
            self.assertEqual(validation["point_count"], 3)
            metrics = convert_points3d_to_ply(
                points_path, Path(temporary) / "cloud.ply"
            )
            self.assertEqual(metrics["tracked_point_count"], 2)
            self.assertEqual(metrics["trackless_point_count"], 1)
            self.assertAlmostEqual(metrics["tracked_mean_reprojection_error_px"], 0.5)
            self.assertAlmostEqual(
                metrics[
                    "legacy_raw_mean_reprojection_error_px_including_negative_sentinels"
                ],
                0.0,
            )
            self.assertEqual(metrics["tracked_reprojection_error_sample_count"], 2)
            self.assertEqual(metrics["unavailable_reprojection_error_count_all_points"], 1)

    def test_negative_reprojection_error_sentinel_is_unavailable_not_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            write_fake_colmap_model(model, point_errors=(-1.0, -1.0))

            # Negative finite upstream sentinels remain structurally valid.
            validation = validate_binary_model(model)
            self.assertEqual(validation["point_count"], 2)
            metrics = convert_points3d_to_ply(
                model / "points3D.bin", Path(temporary) / "sentinel-cloud.ply"
            )
            self.assertIsNone(metrics["mean_reprojection_error_px"])
            self.assertIsNone(metrics["tracked_mean_reprojection_error_px"])
            self.assertEqual(metrics["tracked_reprojection_error_sample_count"], 0)
            self.assertEqual(metrics["tracked_reprojection_error_unavailable_count"], 2)
            self.assertEqual(metrics["unavailable_reprojection_error_count_all_points"], 2)
            self.assertEqual(
                metrics[
                    "legacy_raw_mean_reprojection_error_px_including_negative_sentinels"
                ],
                -1.0,
            )
            self.assertTrue(
                metrics["negative_reprojection_errors_treated_as_unavailable"]
            )


if __name__ == "__main__":
    unittest.main()
