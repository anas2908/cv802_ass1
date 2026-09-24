from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import sys

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ciai_gpu_ui
import ciai_pipeline


def write_ply(path: Path, points: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{index} 0 0 255 0 0" for index in range(points))
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {points}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" + rows + "\n",
        encoding="ascii",
    )


class CIAIPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_pipeline_root = ciai_pipeline.CODE_ROOT

    def tearDown(self) -> None:
        ciai_pipeline.CODE_ROOT = self.old_pipeline_root
        self.temporary.cleanup()

    def make_dataset(self, name: str = "scene") -> Path:
        code = self.root / "code"
        images = code / "datasets" / name / "images"
        images.mkdir(parents=True)
        files = []
        for index, payload in enumerate((b"first", b"second")):
            relative = Path("camera") / f"{index:04d}.jpg"
            path = images / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            files.append(
                {
                    "name": (Path(name) / "images" / relative).as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (code / "datasets" / "images_manifest.json").write_text(
            json.dumps({"datasets": {name: {"files": files}}}), encoding="utf-8"
        )
        ciai_pipeline.CODE_ROOT = code
        return code

    def test_stage_images_is_verified_and_resumable(self) -> None:
        self.make_dataset()
        destination = self.root / "data" / "sfm" / "inputs" / "scene" / "images"
        messages: list[str] = []
        ciai_pipeline.stage_images("scene", destination, messages.append)
        self.assertEqual((destination / "camera" / "0000.jpg").read_bytes(), b"first")
        ciai_pipeline.stage_images("scene", destination, messages.append)
        self.assertIn("Reused 2 checksum-verified images", messages[-1])
        (destination / "camera" / "0000.jpg").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            ciai_pipeline.stage_images("scene", destination, messages.append)

    def test_dynamic_mvs_config_keeps_mutable_file_under_data_root(self) -> None:
        self.make_dataset()
        source_template = self.old_pipeline_root / "mvs" / "configs" / "ciai_template.json"
        template = self.root / "code" / "mvs" / "configs"
        template.mkdir(parents=True)
        (template / "ciai_template.json").write_bytes(source_template.read_bytes())
        data = self.root / "data"
        path = ciai_pipeline._write_mvs_config(data, "ciai-scene-mvs1600-v1")
        payload = json.loads(path.read_text())
        self.assertEqual(payload["experiment"], "ciai-scene-mvs1600-v1")
        self.assertEqual(payload["data_root"], "${CV802_DATA_ROOT}/mvs")
        self.assertTrue(path.is_relative_to(data))

    def test_reviewed_light_and_dark_mvs_recipes_are_not_interchanged(self) -> None:
        self.make_dataset()
        source_template = self.old_pipeline_root / "mvs" / "configs" / "ciai_template.json"
        template = self.root / "code" / "mvs" / "configs"
        template.mkdir(parents=True)
        (template / "ciai_template.json").write_bytes(source_template.read_bytes())
        data = self.root / "data"

        light_recipe = ciai_pipeline.MVS_REFERENCE_RECIPES["light_shirt"]
        light = json.loads(
            ciai_pipeline._write_mvs_config(
                data, str(light_recipe["experiment"]), light_recipe
            ).read_text()
        )
        self.assertEqual(light["experiment"], "light_e10_colmap_mvs_1600")
        self.assertEqual(light["max_image_size"], 1600)
        self.assertEqual(light["masking"]["mode"], "black_background")
        self.assertEqual(light["patch_match"]["num_iterations"], 5)

        dark_recipe = ciai_pipeline.MVS_REFERENCE_RECIPES["dark_shirt"]
        dark = json.loads(
            ciai_pipeline._write_mvs_config(
                data, str(dark_recipe["experiment"]), dark_recipe
            ).read_text()
        )
        self.assertEqual(dark["experiment"], "dark_e3_colmap_mvs_1024_raw_v1")
        self.assertEqual(dark["max_image_size"], 1024)
        self.assertEqual(dark["masking"]["mode"], "none")
        self.assertEqual(dark["patch_match"]["num_iterations"], 3)

    def test_reference_mvs_inputs_honors_explicit_data_root(self) -> None:
        reference = self.root / "reviewed"
        inputs = (
            reference
            / "mvs"
            / "experiments"
            / "light_e10_colmap_mvs_1600"
            / "inputs"
        )
        (inputs / "images").mkdir(parents=True)
        (inputs / "masks").mkdir()
        (inputs / "mask_manifest.json").write_text("{}")
        sparse = inputs / "sparse"
        sparse.mkdir()
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse / name).write_bytes(b"model")
        old = os.environ.get("CV802_REFERENCE_DATA_ROOT")
        os.environ["CV802_REFERENCE_DATA_ROOT"] = str(reference)
        try:
            found = ciai_pipeline._reference_mvs_inputs(
                "light_shirt", self.root / "new-data"
            )
        finally:
            if old is None:
                os.environ.pop("CV802_REFERENCE_DATA_ROOT", None)
            else:
                os.environ["CV802_REFERENCE_DATA_ROOT"] = old
        self.assertEqual(found, inputs)

    def test_reference_calibration_is_verified_and_imported_under_active_root(self) -> None:
        self.make_dataset("light_shirt")
        reference = self.root / "legacy" / "mvs" / "experiments" / "light_e10_colmap_mvs_1600"
        inputs = reference / "inputs"
        sparse = inputs / "sparse"
        masks = inputs / "masks" / "camera"
        sparse.mkdir(parents=True)
        masks.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse / name).write_bytes(name.encode())
        (masks / "0000.png").write_bytes(b"mask0")
        (masks / "0001.png").write_bytes(b"mask1")
        (inputs / "mask_manifest.json").write_text("{}")
        records = []
        for index, payload in enumerate((b"first", b"second")):
            records.append(
                {
                    "role": "registered_image",
                    "destination_relative": f"inputs/images/camera/{index:04d}.jpg",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifests = reference / "manifests"
        manifests.mkdir()
        (manifests / "input_provenance.json").write_text(json.dumps({"files": records}))

        active = self.root / "active"
        images, model, imported_masks, manifest = ciai_pipeline._prepare_reference_mvs_sources(
            "light_shirt", inputs, active, lambda _message: None
        )
        self.assertTrue(images.is_relative_to(active))
        self.assertTrue(model.is_relative_to(active))
        self.assertEqual((model / "cameras.bin").read_bytes(), b"cameras.bin")
        self.assertEqual((imported_masks / "camera" / "0001.png").read_bytes(), b"mask1")
        self.assertEqual(manifest.read_text(), "{}")

    def test_ply_point_count_and_result_discovery(self) -> None:
        self.make_dataset()
        # The UI imported the function object, which reads ciai_pipeline.CODE_ROOT.
        data = self.root / "data"
        cloud = data / "mvs" / "experiments" / "ciai-scene-mvs1600-v1" / "outputs" / "fused.ply"
        write_ply(cloud, 3)
        self.assertEqual(ciai_gpu_ui.ply_point_count(cloud), 3)
        found = ciai_gpu_ui.discover_results(data)
        self.assertEqual(found["MVS-scene"]["point_count"], 3)

    def test_vggsfm_cleanup_is_a_separate_result_and_ui_choice(self) -> None:
        self.make_dataset()
        data = self.root / "data"
        raw = data / "vggsfm" / "outputs" / "ciai-scene-vggsfm-v1" / "point_cloud.ply"
        cleaned = data / "vggsfm" / "outputs" / "ciai-scene-vggsfm-v1-geometry-clean-v1" / "point_cloud.ply"
        write_ply(raw, 3)
        write_ply(cleaned, 2)
        found = ciai_gpu_ui.discover_results(data)
        self.assertEqual(found["VGGSfM-scene"]["point_count"], 3)
        self.assertEqual(found["VGGSfM-geometric-cleanup-scene"]["point_count"], 2)
        self.assertIn('value="vggsfm_cleanup"', ciai_gpu_ui.main_page(["scene"]).decode())

    def test_vggsfm_mac_mask_cleanup_is_separate_from_geometric_cleanup(self) -> None:
        self.make_dataset("light_shirt")
        data = self.root / "data"
        cloud = data / "vggsfm" / "outputs" / "ciai-light_shirt-vggsfm-v1-mac-mask-clean-v1" / "point_cloud.ply"
        write_ply(cloud, 2)
        found = ciai_gpu_ui.discover_results(data)
        self.assertEqual(found["VGGSfM-Mac-mask-cleanup-light_shirt"]["point_count"], 2)
        self.assertIn('value="vggsfm_mask_cleanup"', ciai_gpu_ui.main_page(["light_shirt"]).decode())

    def test_generated_ui_keeps_javascript_newline_escape(self) -> None:
        page = ciai_gpu_ui.main_page(["scene"]).decode("utf-8")
        self.assertIn("s.logs.join('\\n')", page)
        self.assertNotIn("s.logs.join('\n')", page)


if __name__ == "__main__":
    unittest.main()
