from __future__ import annotations

import struct
from pathlib import Path


def write_fake_colmap_model(
    root: Path,
    *,
    image_count: int = 3,
    image_names: list[str] | None = None,
    zero_based_ids: bool = False,
    point_errors: tuple[float, float] = (0.25, 0.75),
) -> None:
    if image_names is not None:
        image_count = len(image_names)
    camera_id = 0 if zero_based_ids else 1
    first_point_id = 0 if zero_based_ids else 10
    second_point_id = first_point_id + 1
    root.mkdir(parents=True, exist_ok=True)
    (root / "cameras.bin").write_bytes(
        struct.pack(
            "<QiiQQddd", 1, camera_id, 0, 640, 480, 500.0, 320.0, 240.0
        )
    )
    with (root / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", image_count))
        for index in range(image_count):
            stream.write(
                struct.pack(
                    "<idddddddi",
                    index if zero_based_ids else index + 1,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(index),
                    0.0,
                    0.0,
                    camera_id,
                )
            )
            name = image_names[index] if image_names is not None else f"frame_{index:03d}.jpg"
            stream.write(name.encode("utf-8") + b"\x00")
            # Keep the synthetic image observations reciprocal with the point
            # tracks below, as a real COLMAP binary model must be.
            if index < 2:
                stream.write(struct.pack("<Q", 2))
                stream.write(struct.pack("<ddQ", 10.0, 20.0, first_point_id))
                stream.write(
                    struct.pack(
                        "<ddQ",
                        30.0,
                        40.0,
                        second_point_id if index == 0 else (1 << 64) - 1,
                    )
                )
            else:
                stream.write(struct.pack("<Q", 0))
    points = [
        (
            first_point_id,
            1.25,
            -2.5,
            3.75,
            255,
            2,
            3,
            point_errors[0],
            [(0, 0), (1, 0)] if zero_based_ids else [(1, 0), (2, 0)],
        ),
        (
            second_point_id,
            -4.0,
            5.5,
            6.0,
            4,
            250,
            6,
            point_errors[1],
            [(0, 1)] if zero_based_ids else [(1, 1)],
        ),
    ]
    with (root / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, x, y, z, r, g, b, error, track in points:
            stream.write(struct.pack("<QdddBBBd", point_id, x, y, z, r, g, b, error))
            stream.write(struct.pack("<Q", len(track)))
            for image_id, point2d_index in track:
                stream.write(struct.pack("<II", image_id, point2d_index))
