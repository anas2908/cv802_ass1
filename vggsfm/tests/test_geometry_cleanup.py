from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clean_point_cloud import clean, read_coloured_ply


class GeometryCleanupTests(unittest.TestCase):
    def test_separate_cloud_removes_isolated_points_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw.ply", root / "derived" / "point_cloud.ply"
            cluster = np.random.default_rng(42).normal(0, 0.01, (100, 3))
            xyz = np.vstack((cluster, [[10, 10, 10], [-10, -10, -10], [20, 0, 0]]))
            records = np.zeros(len(xyz), dtype=np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]))
            for axis in "xyz":
                records[axis] = xyz[:, "xyz".index(axis)]
            records["red"] = 123
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(records)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode()
            raw.write_bytes(header + records.tobytes())
            original = raw.read_bytes()
            receipt = clean(raw, output)
            self.assertEqual(raw.read_bytes(), original)
            self.assertGreaterEqual(receipt["cleaned_points"], 95)
            self.assertLess(receipt["cleaned_points"], len(records))
            self.assertEqual(read_coloured_ply(output)[1].shape[0], receipt["cleaned_points"])
            self.assertEqual(clean(raw, output), receipt)
            self.assertEqual(json.loads((output.parent / "cleanup_receipt.json").read_text()), receipt)
            with self.assertRaises(FileExistsError):
                clean(raw, output, mad_scale=3.0)


if __name__ == "__main__":
    unittest.main()
