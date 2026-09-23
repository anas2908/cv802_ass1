from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from vggsfm_engine.engine import ExecutionOutcome, VGGSfMEngine
from vggsfm_engine.errors import ExecutionError, IncompleteRunError, OutputValidationError
from vggsfm_engine.paths import Layout
from vggsfm_engine.profile import InferenceProfile

from tests.helpers import write_fake_colmap_model


class FakeExecutor:
    def __init__(
        self,
        *,
        fail_first_inference: bool = False,
        inject_unexpected_image_name: bool = False,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail_first_inference = fail_first_inference
        self.inject_unexpected_image_name = inject_unexpected_image_name
        self.inference_calls = 0

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        label: str,
    ) -> ExecutionOutcome:
        del cwd
        self.calls.append(tuple(command))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"fake: {label}\n")
        if label == "CUDA preflight":
            return ExecutionOutcome(0, 0.01)
        self.inference_calls += 1
        if self.fail_first_inference and self.inference_calls == 1:
            return ExecutionOutcome(9, 0.02)
        scene_arg = next(item for item in command if item.startswith("SCENE_DIR="))
        scene = Path(scene_arg.split("=", 1)[1])
        image_names = sorted(path.name for path in (scene / "images").iterdir())
        if self.inject_unexpected_image_name:
            image_names[-1] = "not-an-input.jpg"
        write_fake_colmap_model(
            scene / "sparse",
            image_names=image_names,
        )
        checkpoint = (
            Path(env["HF_HUB_CACHE"])
            / "models--facebook--VGGSfM"
            / "snapshots"
            / "fake-revision"
            / "vggsfm_v2_0_0.bin"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fake-checkpoint")
        return ExecutionOutcome(0, 0.03)


def fixture_layout(root: Path) -> Layout:
    layout = Layout(root / "vggsfm", enforce_production_root=False)
    images = layout.inputs / "light_shirt" / "images"
    images.mkdir(parents=True)
    for index in range(3):
        (images / f"frame_{index:03d}.jpg").write_bytes(
            b"test-image-not-decoded-" + bytes([index])
        )
    layout.official_root.mkdir(parents=True)
    (layout.official_root / "demo.py").write_text("# fake\n", encoding="utf-8")
    layout.python.parent.mkdir(parents=True)
    layout.python.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(layout.python, 0o755)
    return layout


def sparse_profile() -> InferenceProfile:
    return InferenceProfile(
        name="unit-test",
        query_frame_num=3,
        extra_point_pixel_interval=-1,
        concatenate_extra_points=False,
    )


class EngineFakeRunnerTests(unittest.TestCase):
    def test_nested_inputs_get_unique_flat_official_names_with_recorded_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = fixture_layout(Path(temporary))
            images = layout.inputs / "light_shirt" / "images"
            (images / "frame_000.jpg").unlink()
            first = images / "camera_a" / "frame_000.jpg"
            second = images / "camera_b" / "frame_000.jpg"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"nested-a")
            second.write_bytes(b"nested-b")
            engine = VGGSfMEngine(layout, executor=FakeExecutor(), verify_source=False)

            plan = engine.plan("light_shirt", "nested-run", sparse_profile())
            self.assertEqual(plan.image_count, 4)
            request = engine._prepare_request("light_shirt", "nested-run", sparse_profile())
            self.assertEqual(len(request.official_image_names), 4)
            self.assertEqual(len(set(request.official_image_names)), 4)
            self.assertEqual(
                request.official_image_names,
                tuple(sorted(request.official_image_names)),
            )
            self.assertEqual(
                [path.name for path in request.images],
                ["frame_000.jpg", "frame_000.jpg", "frame_001.jpg", "frame_002.jpg"],
            )
            self.assertTrue(request.official_image_names[0].startswith("000000__"))
            mapped = request.descriptor["official_image_name_map"]
            self.assertIn("images/camera_a/frame_000.jpg", {item["source"] for item in mapped})
            self.assertIn("images/camera_b/frame_000.jpg", {item["source"] for item in mapped})

    def test_fake_official_runner_publish_and_idempotent_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = fixture_layout(Path(temporary))
            fake = FakeExecutor()
            engine = VGGSfMEngine(layout, executor=fake, verify_source=False)
            result = engine.run("light_shirt", "unit-run", sparse_profile())
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.point_count, 2)
            self.assertEqual(result.registered_image_count, 3)
            self.assertTrue((result.output_root / "point_cloud.ply").is_file())
            self.assertTrue(
                (result.output_root / "colmap" / "sparse" / "0" / "points3D.bin").is_file()
            )
            manifest = json.loads(
                (result.output_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["model_artifacts"]), 1)
            self.assertEqual(manifest["model_artifacts"][0]["bytes"], 15)
            self.assertEqual(fake.inference_calls, 1)

            cached = engine.run("light_shirt", "unit-run", sparse_profile())
            self.assertEqual(cached.status, "cached")
            self.assertEqual(fake.inference_calls, 1)

    def test_partial_attempt_needs_explicit_resume_and_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = fixture_layout(Path(temporary))
            fake = FakeExecutor(fail_first_inference=True)
            engine = VGGSfMEngine(layout, executor=fake, verify_source=False)
            with self.assertRaises(ExecutionError):
                engine.run("light_shirt", "retry-run", sparse_profile())
            first = layout.experiment_root("retry-run") / "attempts" / "0001"
            self.assertTrue(first.is_dir())
            with self.assertRaises(IncompleteRunError):
                engine.run("light_shirt", "retry-run", sparse_profile())

            result = engine.run(
                "light_shirt", "retry-run", sparse_profile(), resume=True
            )
            self.assertEqual(result.status, "complete")
            self.assertTrue(first.is_dir())
            self.assertTrue(
                (layout.experiment_root("retry-run") / "attempts" / "0002").is_dir()
            )

    def test_publish_rejects_registered_name_absent_from_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = fixture_layout(Path(temporary))
            engine = VGGSfMEngine(
                layout,
                executor=FakeExecutor(inject_unexpected_image_name=True),
                verify_source=False,
            )
            with self.assertRaisesRegex(OutputValidationError, "absent from the input"):
                engine.run("light_shirt", "bad-name-run", sparse_profile())
            self.assertFalse(layout.output_root("bad-name-run").exists())
            receipt = json.loads(
                (
                    layout.experiment_root("bad-name-run")
                    / "attempts"
                    / "0001"
                    / "receipt.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "failed")


if __name__ == "__main__":
    unittest.main()
