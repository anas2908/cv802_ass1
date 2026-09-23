import tempfile
import unittest
from pathlib import Path

import mac_reconstruction_ui as ui


class MacReconstructionUITest(unittest.TestCase):
    def test_discovers_only_image_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "new_subject" / "images"
            images.mkdir(parents=True)
            (images / "one.jpg").write_bytes(b"one")
            (images / "two.PNG").write_bytes(b"two")
            (root / "empty" / "images").mkdir(parents=True)
            self.assertEqual(list(ui.discover_datasets(root)), ["new_subject"])

    def test_dark_e10_is_rejected(self):
        datasets = {"dark_shirt": Path("/tmp/dark")}
        with self.assertRaisesRegex(ValueError, "light_shirt"):
            ui.validate_selection("dark_shirt", "E10", datasets)

    def test_light_e10_dependency_plan(self):
        datasets = {"light_shirt": Path("/tmp/light")}
        old = ui.data_root
        ui.data_root = lambda: Path("/tmp/cv802")
        try:
            plan = ui.experiment_plan("light_shirt", "E10", datasets)
        finally:
            ui.data_root = old
        self.assertEqual(plan["prerequisites"], ["E1", "E2", "E3"])


if __name__ == "__main__":
    unittest.main()
