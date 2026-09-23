from __future__ import annotations

import math
import unittest

import numpy as np

from cv802_evaluation.alignment import (
    robust_similarity_alignment,
    umeyama_similarity,
)


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class UmeyamaTests(unittest.TestCase):
    def test_recovers_known_proper_similarity(self) -> None:
        source = np.random.default_rng(7).normal(size=(30, 3))
        rotation = rotation_z(0.61)
        scale = 2.75
        translation = np.asarray([4.0, -3.0, 1.25])
        target = scale * (source @ rotation.T) + translation

        estimate = umeyama_similarity(source, target)

        np.testing.assert_allclose(estimate.rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(estimate.translation, translation, atol=1e-10)
        self.assertAlmostEqual(estimate.scale, scale, places=10)
        self.assertAlmostEqual(float(np.linalg.det(estimate.rotation)), 1.0, places=10)
        self.assertFalse(estimate.reflection_correction_applied)

    def test_reflection_is_detected_but_never_returned_as_rotation(self) -> None:
        source = np.random.default_rng(11).normal(size=(20, 3))
        reflection = np.diag([-1.0, 1.0, 1.0])
        target = source @ reflection.T + np.asarray([1.0, 2.0, 3.0])

        estimate = umeyama_similarity(source, target)
        residuals = np.linalg.norm(estimate.apply(source) - target, axis=1)

        self.assertTrue(estimate.reflection_correction_applied)
        self.assertAlmostEqual(float(np.linalg.det(estimate.rotation)), 1.0, places=10)
        self.assertGreater(float(np.sqrt(np.mean(residuals**2))), 0.1)

    def test_robust_fit_ignores_large_outliers_and_reports_all_errors(self) -> None:
        source = np.random.default_rng(19).normal(size=(24, 3))
        rotation = rotation_z(-0.37)
        target = 1.4 * (source @ rotation.T) + np.asarray([-2.0, 0.5, 5.0])
        target[-4:] += np.asarray([20.0, -15.0, 8.0])

        estimate = robust_similarity_alignment(
            source,
            target,
            threshold_ratio=0.05,
            min_inlier_ratio=0.60,
            min_inliers=6,
            max_trials=500,
        )

        self.assertEqual(int(np.count_nonzero(estimate.inlier_mask)), 20)
        self.assertTrue(estimate.robust_for_point_cloud)
        np.testing.assert_allclose(estimate.transform.rotation, rotation, atol=1e-9)
        self.assertAlmostEqual(estimate.transform.scale, 1.4, places=9)
        self.assertGreater(
            float(estimate.all_metrics["max_e10_units"]),
            float(estimate.inlier_metrics["max_e10_units"]),
        )


if __name__ == "__main__":
    unittest.main()

