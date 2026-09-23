#!/usr/bin/env python3
"""CPU-only orthographic RGB previews of published PLY point clouds.

Outputs are qualitative projections in each model's own coordinate frame.
Independent methods must be aligned before their coordinates are compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cv802_evaluation.io_utils import DATA_ROOT, atomic_write_json, evaluation_directory, file_record, require_within


TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_sample(path: Path, limit: int = 300000) -> tuple[np.ndarray, np.ndarray, int]:
    """Read regular scalar vertex records and sample deterministically by index."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        fields = []
        count = None
        format_name = None
        active = None
        for _ in range(256):
            line = stream.readline().decode("ascii").strip()
            words = line.split()
            if line == "end_header":
                break
            if not words:
                continue
            if words[0] == "format":
                format_name = words[1]
            elif words[0] == "element":
                active = words[1]
                if active == "vertex":
                    count = int(words[2])
            elif words[0] == "property" and active == "vertex":
                if len(words) != 3 or words[1] not in TYPES:
                    raise ValueError("Only scalar vertex properties are supported")
                fields.append((words[2], TYPES[words[1]]))
        else:
            raise ValueError("Missing PLY end_header")
        if count is None or count <= 0:
            raise ValueError("Empty PLY")
        if not {"x", "y", "z", "red", "green", "blue"}.issubset(dict(fields)):
            raise ValueError("XYZ/RGB properties required")
        stride = max(1, (count + limit - 1) // limit)
        if format_name in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if format_name == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + kind) for name, kind in fields])
            if path.stat().st_size - stream.tell() < count * dtype.itemsize:
                raise ValueError("Truncated PLY vertices")
            vertices = np.memmap(path, dtype=dtype, mode="r", offset=stream.tell(), shape=(count,))[::stride]
            xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
            rgb = np.column_stack([vertices[name] for name in ("red", "green", "blue")]).astype(np.uint8)
        elif format_name == "ascii":
            names = [name for name, _ in fields]
            positions = [names.index(name) for name in ("x", "y", "z", "red", "green", "blue")]
            rows = []
            for index in range(count):
                values = stream.readline().split()
                if len(values) != len(fields):
                    raise ValueError("Malformed PLY vertex")
                if index % stride == 0:
                    rows.append([float(values[column]) for column in positions])
            values = np.asarray(rows)
            xyz, rgb = values[:, :3], values[:, 3:].astype(np.uint8)
        else:
            raise ValueError(f"Unsupported PLY format: {format_name}")
        if not np.isfinite(xyz).all():
            raise ValueError("Non-finite point coordinate")
        return xyz, rgb, count


def render(path: Path, output: Path, label: str, limit: int) -> dict:
    xyz, rgb, original_count = read_sample(path, limit)
    width, height, margin = 600, 620, 25
    sheet = Image.new("RGB", (width * 3, height + 70), (18, 23, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), label, fill="white")
    draw.text((20, 32), f"{original_count:,} published points; {len(xyz):,} displayed | model coordinates; no alignment implied", fill=(190, 200, 215))
    axes = ((0, 1, 2, "XY"), (0, 2, 1, "XZ"), (1, 2, 0, "YZ"))
    for index, (horizontal, vertical, depth, name) in enumerate(axes):
        plane = xyz[:, [horizontal, vertical]]
        lower, upper = plane.min(axis=0), plane.max(axis=0)
        span = np.maximum(upper - lower, 1e-12)
        scale = min((width - 2 * margin) / span[0], (height - 2 * margin - 30) / span[1])
        pixels = np.rint((plane - (lower + upper) / 2) * scale).astype(int)
        pixels[:, 0] += width // 2
        pixels[:, 1] = height // 2 - pixels[:, 1]
        canvas = np.full((height, width, 3), (24, 30, 41), dtype=np.uint8)
        order = np.argsort(xyz[:, depth], kind="stable")
        # Far-to-near overwrite is a one-pixel orthographic depth rendering.
        canvas[pixels[order, 1], pixels[order, 0]] = rgb[order]
        panel = Image.fromarray(canvas)
        ImageDraw.Draw(panel).text((20, 12), f"{name} projection | full coordinate bounds", fill="white")
        sheet.paste(panel, (index * width, 70))
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation ID; preserved preview exists: {output}")
    sheet.save(output)
    return {
        "source": file_record(path), "output": file_record(output),
        "original_points": original_count, "displayed_points": len(xyz),
        "sampling": "deterministic uniform index stride", "coordinate_transform": "none",
        "clipping": "none; each projection fits full sampled coordinate bounds",
        "limitation": "Qualitative view only; point count and appearance do not establish geometric accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-points", type=int, default=300000)
    args = parser.parse_args()
    if not 1 <= args.max_points <= 1000000:
        parser.error("max-points must be in [1, 1000000]")
    source = require_within(args.ply, DATA_ROOT, label="published PLY")
    folder = evaluation_directory(args.evaluation_id)
    report = render(source, folder / "preview.png", args.label, args.max_points)
    atomic_write_json(folder / "preview_receipt.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
