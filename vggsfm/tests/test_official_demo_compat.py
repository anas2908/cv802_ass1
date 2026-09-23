from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from vggsfm_engine.engine import COMPATIBILITY_RUNNER, VGGSfMEngine
from vggsfm_engine.profile import InferenceProfile

from tests.test_engine_fake_runner import fixture_layout


def load_adapter():
    spec = importlib.util.spec_from_file_location("official_demo_compat", COMPATIBILITY_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompatibilityAdapterTests(unittest.TestCase):
    def test_resizes_only_invalid_mask_and_retains_keypoint_mapping(self) -> None:
        from lightglue.utils import Extractor

        original_extract = Extractor.extract
        self.addCleanup(setattr, Extractor, "extract", original_extract)
        adapter = load_adapter()
        adapter.install_invalid_mask_compatibility()

        seen: dict[str, torch.Tensor | tuple[int, ...]] = {}

        class Dummy:
            preprocess_conf = {"resize": 1024, "side": "long"}

            def forward(self, data, invalid_mask=None):
                seen["image_shape"] = tuple(data["image"].shape)
                seen["mask"] = invalid_mask.clone()
                return {"keypoints": torch.tensor([[[0.0, 0.0], [511.5, 255.5]]])}

        image = torch.zeros((1, 3, 320, 640))
        invalid = torch.zeros((1, 320, 640), dtype=torch.bool)
        invalid[:, :160] = True
        result = Extractor.extract(Dummy(), image, invalid_mask=invalid)

        self.assertEqual(seen["image_shape"], (1, 3, 512, 1024))
        resized = seen["mask"]
        assert isinstance(resized, torch.Tensor)
        self.assertEqual(tuple(resized.shape), (1, 512, 1024))
        self.assertEqual(resized.dtype, torch.bool)
        self.assertTrue(torch.all(resized[:, :256]))
        self.assertFalse(torch.any(resized[:, 256:]))
        expected = torch.tensor([[[-0.1875, -0.1875], [319.5, 159.5]]])
        self.assertTrue(torch.allclose(result["keypoints"], expected))
        self.assertEqual(tuple(result["image_size"].shape), (1, 2))
        self.assertTrue(torch.equal(result["image_size"], torch.tensor([[640.0, 320.0]])))

    def test_non1024_request_binds_wrapper_and_upstream_function_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = fixture_layout(Path(temporary))
            source = layout.downloads / "source" / "lightglue" / "lightglue" / "utils.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "class Extractor:\n"
                "    def extract(self, image, invalid_mask=None):\n"
                "        return image\n",
                encoding="utf-8",
            )
            engine = VGGSfMEngine(layout, verify_source=False)
            profile = InferenceProfile(
                name="compat-test", query_frame_num=3, img_size=640,
                extra_point_pixel_interval=-1, concatenate_extra_points=False,
            )
            request = engine._prepare_request("light_shirt", "compat-run", profile)
            record = request.descriptor["compatibility_adapter"]
            self.assertEqual(record["wrapper_source"]["path"], str(COMPATIBILITY_RUNNER))
            self.assertEqual(len(record["wrapper_source"]["sha256"]), 64)
            self.assertEqual(record["upstream_extractor"]["qualified_name"], "Extractor.extract")
            self.assertEqual(len(record["upstream_extractor"]["function_sha256"]), 64)
            command = engine._command(layout.root / "scene", profile)
            self.assertEqual(
                command[1:3],
                (str(COMPATIBILITY_RUNNER), str(layout.official_root / "demo.py")),
            )

    def test_1024_profile_invokes_official_demo_without_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = fixture_layout(Path(temporary))
            engine = VGGSfMEngine(layout, verify_source=False)
            profile = InferenceProfile(
                name="official-default", query_frame_num=3,
                extra_point_pixel_interval=-1, concatenate_extra_points=False,
            )
            request = engine._prepare_request("light_shirt", "official-run", profile)
            self.assertNotIn("compatibility_adapter", request.descriptor)
            command = engine._command(layout.root / "scene", profile)
            self.assertEqual(command[1], str(layout.official_root / "demo.py"))
            self.assertNotIn(str(COMPATIBILITY_RUNNER), command)


if __name__ == "__main__":
    unittest.main()
