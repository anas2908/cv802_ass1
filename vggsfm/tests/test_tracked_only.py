from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from vggsfm_engine.colmap import Point3D


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_tracked_only.py"
SPEC = importlib.util.spec_from_file_location("export_tracked_only", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def point(point_id: int, track_length: int) -> Point3D:
    return Point3D(
        point_id=point_id, x=1.25 * point_id, y=-2.5, z=3.75,
        red=point_id, green=20, blue=30, reprojection_error=-1.0,
        track_length=track_length,
        track=tuple((index, index) for index in range(track_length)),
    )


class TrackedOnlyTests(unittest.TestCase):
    def test_selects_only_positive_tracks_and_preserves_binary_xyz_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "tracked.ply"
            metrics = module.write_tracked_ply(
                [point(1, 2), point(2, 0), point(3, 1)], destination
            )
            self.assertEqual(metrics["point_count"], 2)
            self.assertTrue(metrics["point_ids_unique"])
            self.assertTrue(metrics["xyz_rgb_float32_records_verified"])
            payload = destination.read_bytes()
            body = payload.split(b"end_header\n", 1)[1]
            self.assertEqual(body, module.point_record(point(1, 2)) + module.point_record(point(3, 1)))

    def test_point_record_refuses_trackless_point(self) -> None:
        with self.assertRaisesRegex(ValueError, "trackless"):
            module.point_record(point(2, 0))


if __name__ == "__main__":
    unittest.main()
