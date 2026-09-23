"""The dense reviewer reads new geometry, with malformed vertices rejected."""

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "dark_crutch_review", Path(__file__).resolve().parents[1] / "scripts/review_dark_crutches.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class DarkGeometryReviewTests(unittest.TestCase):
    def make_cloud(self, root, x=1.0):
        path = Path(root) / "cloud.ply"
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        path.write_bytes(header.encode() + struct.pack("<fffBBBfffBBB", x, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60))
        return path

    def test_reads_every_new_xyz_row_in_serialized_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            xyz = review.read_dense_xyz(self.make_cloud(temporary))
            self.assertEqual(xyz.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_nonfinite_vertices_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                review.read_dense_xyz(self.make_cloud(temporary, float("nan")))

    def test_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_cloud(temporary)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                review.read_dense_xyz(path)


if __name__ == "__main__":
    unittest.main()
