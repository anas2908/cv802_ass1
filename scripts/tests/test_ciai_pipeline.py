from __future__ import annotations

import hashlib
import json
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

    def test_ply_point_count_and_result_discovery(self) -> None:
        self.make_dataset()
        # The UI imported the function object, which reads ciai_pipeline.CODE_ROOT.
        data = self.root / "data"
        cloud = data / "mvs" / "experiments" / "ciai-scene-mvs1600-v1" / "outputs" / "fused.ply"
        write_ply(cloud, 3)
        self.assertEqual(ciai_gpu_ui.ply_point_count(cloud), 3)
        found = ciai_gpu_ui.discover_results(data)
        self.assertEqual(found["MVS-scene"]["point_count"], 3)


if __name__ == "__main__":
    unittest.main()
