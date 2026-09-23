from __future__ import annotations

import unittest

import numpy as np

import clean_sparse


class SparseCleanupTests(unittest.TestCase):
    def test_combines_quality_and_spatial_gates(self) -> None:
        xyz = np.asarray(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [-0.1, 0, 0],
                [0, 0.1, 0],
                [0, -0.1, 0],
                [100, 0, 0],
            ],
            dtype=float,
        )
        errors = np.asarray([1, 1, 1, 4, 1, 1], dtype=float)
        tracks = np.asarray([4, 4, 2, 4, 4, 4], dtype=int)
        keep, audit = clean_sparse.quality_mask(xyz, errors, tracks)
        self.assertEqual(keep.tolist(), [True, True, False, False, True, False])
        self.assertEqual(audit["within_reprojection_limit"], 5)

    def test_rejects_incompatible_array_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible shapes"):
            clean_sparse.quality_mask(
                np.zeros((3, 2)), np.zeros(3), np.ones(3, dtype=int)
            )


if __name__ == "__main__":
    unittest.main()
