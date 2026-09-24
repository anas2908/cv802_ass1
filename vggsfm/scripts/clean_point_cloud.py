#!/usr/bin/env python3
"""Non-destructive, CPU-only statistical cleanup of a VGGSfM coloured PLY.

The filter removes points whose mean distance to nearby points is unusually
large. It does not use person masks, infer visibility, or change camera poses.
The raw cloud is never overwritten. This deliberately modest cleanup is
portable to a new user's dataset; mask-based historical reviews are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from scipy.spatial import cKDTree


SCALAR_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8",
}


def read_coloured_ply(path: Path) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Read a scalar-property binary PLY while retaining every vertex byte."""
    payload = path.read_bytes()
    marker = re.search(rb"(?m)^end_header\r?\n", payload[:65536])
    if marker is None:
        raise ValueError("PLY has no complete header")
    header, body = payload[:marker.end()], payload[marker.end():]
    lines = header.decode("ascii").splitlines()
    if not lines or lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
        raise ValueError("Cleanup requires a binary little-endian PLY")
    element = None
    count = None
    fields: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"]:
            if len(parts) != 3:
                raise ValueError("Invalid PLY element declaration")
            element = parts[1]
            if element == "vertex":
                if count is not None:
                    raise ValueError("Duplicate PLY vertex element")
                count = int(parts[2])
            elif int(parts[2]) != 0:
                raise ValueError("PLY contains a non-vertex element")
        elif parts[:1] == ["property"] and element == "vertex":
            if len(parts) != 3 or parts[1] not in SCALAR_TYPES:
                raise ValueError("Only scalar vertex properties are supported")
            fields.append((parts[2], SCALAR_TYPES[parts[1]]))
    if count is None or count < 3 or len({name for name, _ in fields}) != len(fields):
        raise ValueError("Invalid PLY vertex count or properties")
    if not {"x", "y", "z", "red", "green", "blue"}.issubset(name for name, _ in fields):
        raise ValueError("PLY must contain XYZ and RGB vertex properties")
    dtype = np.dtype(fields, align=False)
    if len(body) != count * dtype.itemsize:
        raise ValueError("PLY body length does not match its vertex declaration")
    records = np.frombuffer(body, dtype=dtype)
    xyz = np.column_stack([records[axis] for axis in "xyz"]).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("PLY contains non-finite XYZ")
    return header, records, xyz


def select_inliers(xyz: np.ndarray, *, neighbors: int = 16, mad_scale: float = 4.0) -> tuple[np.ndarray, dict]:
    """Keep points below median + scale * robust-MAD of mean KNN distance."""
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
        raise ValueError("Need at least three XYZ points")
    if neighbors < 2 or not np.isfinite(mad_scale) or mad_scale <= 0:
        raise ValueError("neighbors and mad_scale must be positive")
    k = min(neighbors, len(xyz) - 1)
    distances, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=4)
    means = distances[:, 1:].mean(axis=1)
    median = float(np.median(means))
    mad = float(np.median(np.abs(means - median)))
    robust_sigma = 1.4826 * mad
    # When many points coincide, the MAD can be zero. Keep exact-neighbour
    # points and use an ordinary spread only if there is a nonzero spread.
    if robust_sigma == 0.0:
        robust_sigma = float(np.std(means))
    threshold = median + mad_scale * robust_sigma
    selected = means <= threshold
    if selected.sum() < 3 or selected.mean() < 0.25:
        raise ValueError("Cleanup would remove over 75% of points; review the raw cloud")
    return selected, {
        "neighbors": k,
        "mad_scale": mad_scale,
        "median_mean_neighbor_distance": median,
        "robust_sigma": robust_sigma,
        "distance_threshold": threshold,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(source: Path, destination: Path, *, neighbors: int = 16, mad_scale: float = 4.0) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("Cleanup destination must differ from the raw PLY")
    raw_hash = sha256(source)
    receipt_path = destination.with_name("cleanup_receipt.json")
    if destination.exists() or receipt_path.exists():
        if not destination.is_file() or not receipt_path.is_file():
            raise FileExistsError("Incomplete previous cleanup; preserved for review")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("raw_sha256") != raw_hash
                or receipt.get("neighbors_requested") != neighbors
                or receipt.get("mad_scale") != mad_scale
                or receipt.get("cleaned_sha256") != sha256(destination)):
            raise FileExistsError("Existing cleanup has different inputs or settings; preserved")
        return receipt
    header, records, xyz = read_coloured_ply(source)
    selected, settings = select_inliers(xyz, neighbors=neighbors, mad_scale=mad_scale)
    new_header, changes = re.subn(
        rb"(?m)^element vertex [0-9]+(?=\r?$)",
        b"element vertex " + str(int(selected.sum())).encode(), header,
    )
    if changes != 1:
        raise ValueError("Could not update PLY vertex count")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".clean-", suffix=".ply", delete=False) as stream:
        staged = Path(stream.name)
        stream.write(new_header)
        stream.write(records[selected].tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        cleaned_hash = sha256(staged)
        receipt = {
            "schema_version": 1,
            "method": "statistical_knn_outlier_filter",
            "limitations": "No segmentation or occlusion test; thin structures may be removed and coherent background may remain.",
            "source": str(source),
            "destination": str(destination),
            "raw_sha256": raw_hash,
            "cleaned_sha256": cleaned_hash,
            "raw_points": len(records),
            "cleaned_points": int(selected.sum()),
            "removed_points": int((~selected).sum()),
            "neighbors_requested": neighbors,
            "mad_scale": mad_scale,
            "settings": settings,
        }
        os.replace(staged, destination)
        # A receipt is written only after the complete cloud is installed.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, prefix=".receipt-", suffix=".json", delete=False) as stream:
            staged_receipt = Path(stream.name)
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_receipt, receipt_path)
        return receipt
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--mad-scale", type=float, default=4.0)
    args = parser.parse_args()
    print(json.dumps(clean(args.source, args.output, neighbors=args.neighbors, mad_scale=args.mad_scale), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
