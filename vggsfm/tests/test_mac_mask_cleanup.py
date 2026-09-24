from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clean_with_mac_masks import safe_relative, select_by_support, source_rows


class MacMaskCleanupTests(unittest.TestCase):
    def test_90_percent_consensus_needs_six_usable_views(self) -> None:
        foreground = np.array([5, 5, 6, 9, 9], dtype=np.uint16)
        usable = np.array([5, 6, 6, 10, 9], dtype=np.uint16)
        self.assertEqual(select_by_support(foreground, usable).tolist(),
                         [False, False, True, True, True])

    def test_rejects_unsafe_mask_paths_and_duplicate_rows(self) -> None:
        with self.assertRaises(ValueError):
            safe_relative("../other.png")
        with self.assertRaises(ValueError):
            source_rows({"status": "complete", "mask_count": 2,
                         "rows": [{"image_relative": "same"}, {"image_relative": "same"}]})


if __name__ == "__main__":
    unittest.main()
