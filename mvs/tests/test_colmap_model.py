from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from cv802_mvs.colmap_model import load_model_records


TEST_TEMP = Path("/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests")


class ColmapModelTest(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reads_text_model_with_nested_names(self) -> None:
        (self.root / "cameras.txt").write_text(
            "# cameras\n1 PINHOLE 10 20 8 8 5 10\n", encoding="utf-8"
        )
        (self.root / "images.txt").write_text(
            "# images\n"
            "1 1 0 0 0 0 0 0 1 folder/a.jpg\n\n"
            "2 1 0 0 0 1 0 0 1 folder/b.jpg\n\n",
            encoding="utf-8",
        )
        (self.root / "points3D.txt").write_text("# points\n", encoding="utf-8")
        model = load_model_records(self.root)
        self.assertEqual(model.format, "text")
        self.assertEqual([image.name for image in model.images], ["folder/a.jpg", "folder/b.jpg"])

    def test_reads_binary_model_and_skips_points(self) -> None:
        with (self.root / "cameras.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 1))
            stream.write(struct.pack("<IiQQ4d", 1, 1, 10, 20, 8.0, 8.0, 5.0, 10.0))
        with (self.root / "images.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 2))
            for image_id, name in ((1, "a.jpg"), (2, "b.jpg")):
                stream.write(struct.pack("<I7dI", image_id, 1, 0, 0, 0, image_id, 0, 0, 1))
                stream.write(name.encode() + b"\0")
                stream.write(struct.pack("<Q", 1))
                stream.write(struct.pack("<ddQ", 1.0, 2.0, 999))
        (self.root / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        model = load_model_records(self.root)
        self.assertEqual(model.format, "binary")
        self.assertEqual(len(model.images), 2)
        self.assertEqual(model.cameras[0].parameters, (8.0, 8.0, 5.0, 10.0))


if __name__ == "__main__":
    unittest.main()

