from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from reviewed_assets import BUNDLE, install_mvs_reference


class ReviewedAssetsTests(unittest.TestCase):
    def test_bundled_calibrations_install_without_legacy_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset, expected_images in (("light_shirt", 125), ("dark_shirt", 290)):
                reference = install_mvs_reference(dataset, root, lambda _line: None)
                self.assertTrue((reference / "sparse" / "cameras.bin").is_file())
                self.assertTrue((reference / "sparse" / "images.bin").is_file())
                self.assertTrue((reference / "sparse" / "points3D.bin").is_file())
                provenance = json.loads((reference / "input_provenance.json").read_text())
                self.assertEqual(len(provenance["files"]), expected_images)
                self.assertEqual(install_mvs_reference(dataset, root, lambda _line: None), reference)
                self.assertEqual((reference / "masks").is_dir(), dataset == "light_shirt")

    def test_bundled_masks_cover_each_included_dataset(self) -> None:
        for dataset, count in (("light_shirt", 125), ("dark_shirt", 290)):
            folder = BUNDLE / "mac_masks" / dataset
            receipt = json.loads((folder / "receipt.json").read_text())
            self.assertEqual(len(receipt["rows"]), count)
            images = Path(__file__).resolve().parents[2] / "datasets" / dataset / "images"
            self.assertEqual(
                {row["image_relative"] for row in receipt["rows"]},
                {photo.relative_to(images).as_posix() for photo in images.rglob("*.jpg")},
            )


if __name__ == "__main__":
    unittest.main()
