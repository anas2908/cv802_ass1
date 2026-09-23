import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mac_recipe_runner as runner
import mac_reconstruction_ui as ui


class MacReconstructionUITest(unittest.TestCase):
    def test_restores_augmented_boxes_when_retry_replaced_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "boxes.json"
            backup = root / "light_shirt_boxes.json"
            destination.write_bytes(b"original detector boxes")
            backup.write_bytes(b"augmented E3 boxes")
            expected = runner.file_sha256(backup)
            self.assertTrue(runner.restore_file_for_hash(destination, backup, expected))
            self.assertEqual(destination.read_bytes(), backup.read_bytes())
            self.assertFalse(runner.restore_file_for_hash(destination, backup, expected))

    def test_reuses_verified_vocabulary_pair_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "light_shirt"
            folder.mkdir()
            outputs = {
                "matching_pairs.txt": b"a.jpg b.jpg\n",
                "guided_pairs.txt": b"a.jpg b.jpg\n",
                "retrieved_neighbors.json": b"{}\n",
            }
            for name, content in outputs.items():
                (folder / name).write_bytes(content)
            provenance = {
                "subject": "light_shirt", "ordinary_pairs": 1, "guided_pairs": 1,
                "output_sha256": {
                    name: runner.file_sha256(folder / name) for name in outputs
                },
            }
            (folder / "retrieval_provenance.json").write_text(json.dumps(provenance))
            self.assertTrue(runner.validate_vocab_pairs(folder, "light_shirt"))

    def test_repairs_prepared_e10_before_matching_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "light_shirt_quality"
            (dataset / "colmap/quality_feature_cache").mkdir(parents=True)
            (dataset / "sfm_refine.json").write_text(json.dumps({
                "matching_pairs": "matching_pairs.txt",
                "guided_pairs": "guided_pairs.txt",
            }))
            (dataset / "experiment_provenance.json").write_text(json.dumps({
                "experiment": "E10_quality_exhaustive_guided",
                "subject": "light_shirt",
            }))
            names = [f"image_{number:03}.jpg" for number in range(125)]
            pairs = "".join(
                f"{first} {second}\n"
                for index, first in enumerate(names) for second in names[index + 1:]
            )
            (dataset / "matching_pairs.txt").write_text(pairs)
            (dataset / "guided_pairs.txt").write_text(pairs)
            self.assertTrue(runner.ensure_e10_resume_config(dataset, "light_shirt"))
            config = json.loads((dataset / "sfm_refine.json").read_text())
            self.assertIs(config["resume_matching"], True)
            self.assertEqual(config["matching_batch_size"], 128)
            self.assertEqual(config["matching_threads"], 1)
            self.assertFalse(runner.ensure_e10_resume_config(dataset, "light_shirt"))

    def test_e4_and_e5_create_only_the_selected_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality = root / "quality"
            masks = root / "masks"
            masks.mkdir()
            with (mock.patch.object(runner, "ensure_e3", return_value=quality),
                  mock.patch.object(runner, "ensure_mask_prerequisites",
                                    return_value=(masks, None)),
                  mock.patch.object(runner, "run") as run):
                runner.ensure_e4_or_e5(root, "light_shirt", root / "baseline", "E4")
                e4_command = [str(value) for value in run.call_args.args[0]]
                self.assertIn("mask_consensus_90:.90", e4_command)
                self.assertNotIn("mask_consensus_97:.97", e4_command)
                run.reset_mock()
                runner.ensure_e4_or_e5(root, "light_shirt", root / "baseline", "E5")
                e5_command = [str(value) for value in run.call_args.args[0]]
                self.assertIn("mask_consensus_97:.97", e5_command)
                self.assertNotIn("mask_consensus_90:.90", e5_command)

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
