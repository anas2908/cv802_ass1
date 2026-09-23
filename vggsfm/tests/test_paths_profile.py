from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from vggsfm_engine.errors import ConfigurationError, StoragePolicyError
from vggsfm_engine.paths import Layout, require_within
from vggsfm_engine.profile import InferenceProfile


class PathAndProfileTests(unittest.TestCase):
    def test_path_policy_resolves_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            link = root / "escape"
            os.symlink(Path(outside), link)
            with self.assertRaises(StoragePolicyError):
                require_within(link, root, label="test", must_exist=True)

    def test_production_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(StoragePolicyError):
                Layout(Path(temporary))

    def test_profile_rejects_unbounded_hydra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            profile.write_text('{"name":"bad","load_gt":true}', encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                InferenceProfile.from_json(profile)

    def test_sparse_profile_is_valid(self) -> None:
        profile = InferenceProfile(
            name="sparse",
            extra_point_pixel_interval=-1,
            concatenate_extra_points=False,
        )
        profile.validate(image_count=6)
        self.assertNotIn("load_gt", {item.name for item in fields(InferenceProfile)})
        self.assertIn("extra_by_neighbor=16", profile.hydra_overrides())

    def test_extra_point_neighbor_window_must_support_triangulation(self) -> None:
        with self.assertRaises(ConfigurationError):
            InferenceProfile(extra_point_neighbor_frames=3).validate(image_count=6)


if __name__ == "__main__":
    unittest.main()
