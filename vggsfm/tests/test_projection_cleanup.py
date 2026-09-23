from __future__ import annotations

import unittest

import numpy as np

from vggsfm_engine.projection_cleanup import (
    corridor_hits,
    project_original_pixels,
    separated_view_confirmation,
)


class IdentityPose:
    def __mul__(self, xyz: np.ndarray) -> np.ndarray:
        return xyz


class FakeImage:
    cam_from_world = IdentityPose()


class FakeCamera:
    width = 100
    height = 80

    @staticmethod
    def img_from_cam(xy: np.ndarray) -> np.ndarray:
        return xy * np.array([50.0, 40.0]) + np.array([50.0, 40.0])


class ProjectionCleanupTests(unittest.TestCase):
    def test_projection_requires_positive_z_finite_and_original_bounds(self) -> None:
        xyz = np.array([
            [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [np.nan, 0.0, 1.0],
        ])
        pixels, usable = project_original_pixels(FakeImage(), FakeCamera(), xyz)
        np.testing.assert_allclose(pixels[0], [50.0, 40.0])
        self.assertEqual(usable.tolist(), [True, False, False, False])

    def test_corridors_union_groups_and_respect_reviewed_width(self) -> None:
        pixels = np.array([[50.0, 40.0], [55.0, 40.0], [80.0, 40.0]])
        groups = {
            "A": {
                "half_width_fraction_of_image_width": 0.01,
                "polylines": [[[0.5, 0.2], [0.5, 0.8]]],
            },
            "B": {
                "half_width_fraction_of_image_width": 0.01,
                "polylines": [[[0.8, 0.2], [0.8, 0.8]]],
            },
        }
        hit = corridor_hits(
            pixels, np.ones(3, dtype=bool), width=100, height=80, groups=groups
        )
        self.assertEqual(hit.tolist(), [True, False, True])

    def test_confirmation_needs_three_pairwise_separated_hit_rays(self) -> None:
        xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        centers = np.array([
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
            [-1.01, 0.0, 0.0],
        ])
        hits = np.array([
            # Point 0 has a well-separated triple. Point 1 also has three
            # positive corridor hits, but views 0 and 3 are almost collinear;
            # it must be rejected specifically by the angular gate.
            [True, True], [True, True], [True, False], [False, True],
        ])
        confirmed = separated_view_confirmation(xyz, centers, hits)
        self.assertEqual(confirmed.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
