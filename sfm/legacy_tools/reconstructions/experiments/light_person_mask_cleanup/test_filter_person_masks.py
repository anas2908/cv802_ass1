"""Synthetic helper tests; never load real masks or reconstruction files."""

import unittest

import numpy as np

from filter_person_masks import choose_points, sample_foreground


class SampleForegroundTest(unittest.TestCase):
    def test_pixel_coordinates_are_floored_and_use_xy_order(self):
        foreground = np.zeros((5, 7), dtype=bool)
        foreground[2, 3] = True
        uv = np.array([[3.99, 2.99], [2.99, 2.99], [2.99, 3.99]])

        result = sample_foreground(foreground, uv, tolerance=0)

        np.testing.assert_array_equal(result, [True, False, False])
        self.assertEqual(result.dtype, np.dtype(bool))
        self.assertEqual(result.shape, (3,))

    def test_tolerance_is_a_disk_in_native_pixels(self):
        foreground = np.zeros((9, 9), dtype=bool)
        foreground[4, 4] = True
        uv = np.array([[3.1, 4.1], [3.1, 3.1], [2.1, 4.1], [4.1, 4.1]])

        np.testing.assert_array_equal(
            sample_foreground(foreground, uv, tolerance=0.99),
            [False, False, False, True],
        )
        np.testing.assert_array_equal(
            sample_foreground(foreground, uv, tolerance=1),
            [True, False, False, True],
        )
        np.testing.assert_array_equal(
            sample_foreground(foreground, uv, tolerance=1.5),
            [True, True, False, True],
        )
        np.testing.assert_array_equal(
            sample_foreground(foreground, uv, tolerance=2),
            [True, True, True, True],
        )

    def test_border_offsets_are_ignored_without_wrapping_or_clamping(self):
        foreground = np.zeros((4, 5), dtype=bool)
        foreground[0, 0] = True
        foreground[3, 2] = True
        uv = np.array([
            [0.2, 0.8],  # Valid corner foreground still counts.
            [1.2, 0.8],  # One valid offset reaches that corner.
            [4.2, 0.8],  # Positive out-of-bounds x must not wrap to x=0.
            [2.2, 0.8],  # Negative out-of-bounds y must not wrap to y=3.
            [4.2, 3.8],  # Bottom/right offsets are also outside the image.
        ])

        np.testing.assert_array_equal(
            sample_foreground(foreground, uv, tolerance=1),
            [True, True, False, False, False],
        )


class ChoosePointsTest(unittest.TestCase):
    def test_perfect_ratio_still_requires_absolute_support(self):
        result = choose_points(
            foreground_counts=np.array([1, 3, 4, 4, 0]),
            in_frame_counts=np.array([1, 3, 4, 5, 0]),
            distinct_views=np.array([3, 3, 3, 3, 3]),
            reliable_count=20,
            consensus=0.9,
        )

        # Twenty reliable masks require at least four positive views. A 1/1
        # ratio cannot pass; zero valid projection votes cannot pass either.
        np.testing.assert_array_equal(result, [False, False, True, False, False])
        self.assertEqual(result.dtype, np.dtype(bool))

    def test_required_support_is_rounded_up(self):
        result = choose_points(
            foreground_counts=np.array([4, 5]),
            in_frame_counts=np.array([4, 5]),
            distinct_views=np.array([3, 3]),
            reliable_count=21,
            consensus=0.9,
            min_positive_view_fraction=0.2,
        )

        np.testing.assert_array_equal(result, [False, True])

    def test_repeated_track_observations_do_not_replace_distinct_views(self):
        result = choose_points(
            foreground_counts=np.array([3, 3, 3]),
            in_frame_counts=np.array([3, 3, 3]),
            distinct_views=np.array([2, 3, 4]),
            reliable_count=10,
            consensus=0.97,
            min_track_views=3,
        )

        # The first point could have three raw observations, but only two
        # distinct observing images; it must fail the track-support gate.
        np.testing.assert_array_equal(result, [False, True, True])

    def test_ninety_seven_percent_selection_is_a_subset_of_ninety_percent(self):
        inputs = {
            "foreground_counts": np.array([97, 96, 90, 89, 20, 19, 0]),
            "in_frame_counts": np.array([100, 100, 100, 100, 20, 19, 0]),
            "distinct_views": np.full(7, 3),
            "reliable_count": 100,
        }
        permissive = choose_points(**inputs, consensus=0.90)
        strict = choose_points(**inputs, consensus=0.97)

        np.testing.assert_array_equal(permissive, [True, True, True, False, True, False, False])
        np.testing.assert_array_equal(strict, [True, False, False, False, True, False, False])
        self.assertTrue(np.all(~strict | permissive))
        self.assertTrue(np.any(permissive & ~strict))


if __name__ == "__main__":
    unittest.main()
