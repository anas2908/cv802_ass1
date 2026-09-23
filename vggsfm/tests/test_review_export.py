"""Tests for the CPU-only, derived VGGSfM COLMAP review export."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_export.py"
SPEC = importlib.util.spec_from_file_location("review_export", SCRIPT)
review_export = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review_export)


class ExactCoordinateTests(unittest.TestCase):
    def test_portrait_and_landscape_use_upstream_float32_formula(self) -> None:
        portrait = review_export.exact_crop_correction(3024, 4032, 1024)
        self.assertEqual(portrait["left"], -504)
        self.assertEqual(portrait["top"], 0)
        self.assertEqual(portrait["top_left_abs"], [128.0, 0.0])
        self.assertEqual(portrait["resize_ratio"], 3.9375)
        np.testing.assert_array_equal(
            review_export.correct_xy([128.0, 0.0], portrait), [0.0, 0.0]
        )
        np.testing.assert_array_equal(
            review_export.correct_xy([896.0, 1024.0], portrait), [3024.0, 4032.0]
        )

        landscape = review_export.exact_crop_correction(4032, 3024, 1024)
        self.assertEqual(landscape["top_left_abs"], [0.0, 128.0])
        np.testing.assert_array_equal(
            review_export.correct_xy([0.0, 128.0], landscape), [0.0, 0.0]
        )

    def test_odd_dimensions_match_independent_float32_reference(self) -> None:
        width, height, size = 1081, 1921, 1024
        actual = review_export.exact_crop_correction(width, height, size)
        length = max(width, height)
        offset = np.abs(np.asarray(
            np.asarray([(width - length) // 2, (height - length) // 2])
            / length * size,
            dtype=np.float32,
        ))
        ratio = float(np.float32(np.float32(length) / np.float32(size)))
        self.assertEqual(actual["top_left_abs"], offset.astype(float).tolist())
        self.assertEqual(actual["resize_ratio"], ratio)

    def test_residual_statistics_fail_closed_on_empty_or_nonfinite(self) -> None:
        with self.assertRaisesRegex(Exception, "No tracked observations"):
            review_export.residual_statistics([])
        with self.assertRaisesRegex(Exception, "finite and nonnegative"):
            review_export.residual_statistics([1.0, float("nan")])
        summary = review_export.residual_statistics([3.0, 4.0])
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["rmse_px"], (12.5) ** 0.5)


class EvidenceTests(unittest.TestCase):
    def test_resolved_config_requires_explicit_false_and_rejects_true_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "official.log"
            log.write_text("header\nshift_point2d_to_original_res: false\n", encoding="utf-8")
            evidence = review_export._resolved_setting_evidence(log, ["img_size=1024"])
            self.assertIs(evidence["shift_point2d_to_original_res"], False)
            self.assertEqual(evidence["resolved_config_line_numbers"], [2])
            self.assertTrue(Path(evidence["official_log"]["path"]).is_absolute())
            with self.assertRaisesRegex(Exception, "explicitly enabled"):
                review_export._resolved_setting_evidence(
                    log, ["shift_point2d_to_original_res=true"]
                )
            log.write_text("shift_point2d_to_original_res: true\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "does not explicitly resolve"):
                review_export._resolved_setting_evidence(log, [])

    def test_request_fingerprint_and_safe_identifier_fail_closed(self) -> None:
        descriptor = {
            "schema_version": 1,
            "method": "official_vggsfm_v2",
            "dataset": "fixture",
        }
        request = {
            **descriptor,
            "created_at": "ignored-for-fingerprint",
            "request_fingerprint": review_export.object_sha256(descriptor),
        }
        self.assertTrue(review_export._request_fingerprint_is_valid(request))
        request["dataset"] = "mutated"
        self.assertFalse(review_export._request_fingerprint_is_valid(request))
        with self.assertRaises(Exception):
            review_export.derive("../escape")

    def test_file_record_is_absolute_and_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.bin"
            path.write_bytes(b"first")
            first = review_export.absolute_file_record(path)
            self.assertEqual(set(first), {"path", "bytes", "sha256"})
            self.assertTrue(Path(first["path"]).is_absolute())
            path.write_bytes(b"other")
            self.assertNotEqual(first, review_export.absolute_file_record(path))


class PyCOLMAPFixtureTests(unittest.TestCase):
    def test_padding_track_is_audited_and_never_clamped(self) -> None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction()
        reconstruction.add_camera(pycolmap.Camera.create(
            1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0, 200, 100
        ))
        reconstruction.add_image(pycolmap.Image(
            name="padding.jpg", keypoints=np.asarray([[50.0, 0.0]]),
            cam_from_world=pycolmap.Rigid3d(), camera_id=1, id=1,
        ))
        reconstruction.register_image(1)
        point_id = reconstruction.add_point3D(
            np.asarray([0.0, 0.0, 5.0]), pycolmap.Track(),
            np.asarray([1, 2, 3], dtype=np.uint8),
        )
        reconstruction.add_observation(point_id, pycolmap.TrackElement(1, 0))
        correction = review_export.exact_crop_correction(200, 100, 100)
        metrics = review_export.apply_coordinate_correction(
            reconstruction, {"padding.jpg": correction}
        )
        np.testing.assert_array_equal(
            reconstruction.images[1].points2D[0].xy, [100.0, -50.0]
        )
        self.assertEqual(metrics["linked_point2d_count"], 1)
        self.assertEqual(metrics["linked_in_camera_bounds_count"], 0)
        self.assertEqual(metrics["linked_outside_camera_bounds_count"], 1)
        self.assertTrue(metrics["linked_bounds_count_sum_matches_total"])
        self.assertEqual(metrics["minimum_linked_border_margin_px"], -50.0)
        self.assertEqual(metrics["policy"], "preserve_exact_formula_and_tracks_no_clamp")

    def test_real_reprojection_audit_and_semantic_invariants(self) -> None:
        import pycolmap

        source = pycolmap.Reconstruction()
        source.add_camera(pycolmap.Camera.create(
            1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0, 200, 100
        ))
        source.add_image(pycolmap.Image(
            name="fixture.jpg",
            keypoints=np.asarray([[50.0, 50.0], [25.0, 25.0]], dtype=np.float64),
            cam_from_world=pycolmap.Rigid3d(), camera_id=1, id=1,
        ))
        source.register_image(1)
        tracked_id = source.add_point3D(
            np.asarray([0.0, 0.0, 5.0]), pycolmap.Track(),
            np.asarray([1, 2, 3], dtype=np.uint8),
        )
        source.add_observation(tracked_id, pycolmap.TrackElement(1, 0))
        source.add_point3D(
            np.asarray([1.0, 1.0, 5.0]), pycolmap.Track(),
            np.asarray([4, 5, 6], dtype=np.uint8),
        )

        correction = review_export.exact_crop_correction(200, 100, 100)
        correction.update({"official_image": "fixture.jpg"})
        corrections = {"fixture.jpg": correction}
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            source_dir.mkdir()
            source.write(str(source_dir))
            original = pycolmap.Reconstruction(str(source_dir))
            derived = pycolmap.Reconstruction(str(source_dir))

            pre, _ = review_export.actual_reprojection_errors(original)
            correction_metrics = review_export.apply_coordinate_correction(
                derived, corrections
            )
            post, means = review_export.actual_reprojection_errors(derived)
            review_export.set_reprojection_errors(derived, means)
            derived_dir = Path(temporary) / "derived"
            derived_dir.mkdir()
            derived.write(str(derived_dir))
            reloaded = pycolmap.Reconstruction(str(derived_dir))
            invariants = review_export.semantic_invariants(
                original, reloaded, corrections, means
            )

        self.assertAlmostEqual(pre["mean_px"], 50.0)
        self.assertAlmostEqual(post["mean_px"], 0.0)
        self.assertEqual(correction_metrics["linked_point2d_count"], 1)
        self.assertTrue(correction_metrics["all_linked_finite"])
        self.assertEqual(correction_metrics["policy"], "preserve_exact_formula_and_tracks_no_clamp")
        self.assertTrue(invariants["point_ids_xyz_rgb_tracks_unchanged"])
        self.assertTrue(invariants["only_point2d_xy_and_point_error_fields_permitted_to_differ"])


if __name__ == "__main__":
    unittest.main()
