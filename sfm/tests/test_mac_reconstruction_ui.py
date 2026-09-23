import json
import os
import tempfile
import unittest
from pathlib import Path

import mac_recipe_runner as runner
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

    def test_browser_ui_contains_dataset_and_experiment_selectors(self):
        old = ui.discover_datasets
        ui.discover_datasets = lambda: {"light_shirt": Path("/tmp/light")}
        try:
            page = ui.render_web_ui(ui.ReconstructionState()).decode("utf-8")
        finally:
            ui.discover_datasets = old
        self.assertIn('name="dataset"', page)
        self.assertIn('value="light_shirt"', page)
        self.assertIn('name="experiment"', page)
        self.assertIn('value="E10"', page)

    def test_recipe_result_paths_match_the_published_e1_e10_catalog(self):
        catalog = json.loads(ui.HISTORICAL_CONFIG.read_text())
        subject_names = {"light": "light_shirt", "dark": "black_shirt_crutches"}
        root = Path("/tmp/workspace")
        for experiment in catalog["experiments"]:
            for subject, record in experiment["subjects"].items():
                if not record["available"]:
                    continue
                actual = runner.expected_result(
                    root, subject_names[subject], experiment["id"]
                )
                expected = root / "reconstructions" / record["relative_ply"]
                self.assertEqual(actual, expected)

    def test_completed_result_enables_progress_and_view_button(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud = root / "outputs/result.ply"
            cloud.parent.mkdir()
            cloud.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n0 0 0 255 0 0\n"
            )
            receipts = root / "sfm/mac-e1-e10-workspace/receipts"
            receipts.mkdir(parents=True)
            (receipts / "light_shirt_E1.json").write_text(json.dumps({
                "dataset": "light_shirt", "experiment": "E1",
                "result_ply": str(cloud),
            }))
            old = os.environ.get("CV802_DATA_ROOT")
            os.environ["CV802_DATA_ROOT"] = str(root)
            try:
                state = ui.ReconstructionState()
                state.restore_latest()
                page = ui.render_web_ui(state).decode("utf-8")
                payload = ui.result_catalog(state)
            finally:
                if old is None:
                    os.environ.pop("CV802_DATA_ROOT", None)
                else:
                    os.environ["CV802_DATA_ROOT"] = old
            self.assertIn("View reconstructed result", page)
            self.assertIn("1 coloured points", page)
            self.assertIn('aria-valuenow="100"', page)
            self.assertEqual(state.snapshot()["label"], "Previous result")
            self.assertEqual(payload["experiments"][0]["subjects"]["light"]["point_count"], 1)


if __name__ == "__main__":
    unittest.main()
