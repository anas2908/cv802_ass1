import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("dark_cleanup", Path(__file__).resolve().parents[1] / "scripts/clean_dark_person_crutches.py")
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class DarkCleanupTests(unittest.TestCase):
    def test_crutches_bypass_body_mask(self):
        body, kept = cleanup.select_union([0, 9, 8, 1], [10, 10, 10, 1], [1, 0, 0, 0])
        self.assertEqual(body.tolist(), [False, True, False, False])
        self.assertEqual(kept.tolist(), [True, True, False, False])

    def test_invalid_votes_fail(self):
        with self.assertRaises(ValueError):
            cleanup.select_union([11], [10], [0])

    def test_binary_vertex_subset_preserves_normals_and_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.ply", Path(folder) / "selected.ply"
            dtype = np.dtype([(n, "<f4") for n in ("x", "y", "z", "nx", "ny", "nz")] + [(n, "u1") for n in ("red", "green", "blue")])
            rows = np.zeros(3, dtype=dtype)
            rows["x"] = [1.25, 4.5, 7.75]
            rows["red"] = [2, 90, 211]
            rows["nz"] = [-1, 0, 1]
            header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
            header += "".join("property float " + n + "\n" for n in ("x", "y", "z", "nx", "ny", "nz"))
            header += "".join("property uchar " + n + "\n" for n in ("red", "green", "blue")) + "end_header\n"
            source.write_bytes(header.encode() + rows.tobytes())
            old = source.read_bytes()
            self.assertEqual(cleanup.write_exact_subset(source, [True, False, True], target), 2)
            self.assertEqual(cleanup.read_vertices(target)[1].tobytes(), rows[[0, 2]].tobytes())
            self.assertEqual(source.read_bytes(), old)
            with self.assertRaises(FileExistsError):
                cleanup.write_exact_subset(source, [True, False, True], target)
