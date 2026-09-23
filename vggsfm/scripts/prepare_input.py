#!/usr/bin/env python3
"""Copy one raw image set into the independent VGGSfM data namespace.

The importer never deletes or mutates the source.  It checksum-verifies the
staged copy before atomically publishing the dataset directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
DATA_ROOT = Path(os.environ.get("CV802_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
if not DATA_ROOT.is_absolute():
    raise RuntimeError("CV802_DATA_ROOT must be an absolute path")
DESTINATION_ROOT = DATA_ROOT.resolve() / "vggsfm" / "inputs"
CODE_ROOT = Path(__file__).resolve().parents[1]
if CODE_ROOT == DESTINATION_ROOT or CODE_ROOT in DESTINATION_ROOT.parents:
    raise RuntimeError("CV802_DATA_ROOT must be outside the source-code checkout")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(value: str) -> str:
    if not value or len(value) > 80 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise argparse.ArgumentTypeError("dataset contains unsafe characters")
    return value


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise RuntimeError(f"source directory is missing: {directory}")
    paths = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if len(paths) < 3:
        raise RuntimeError(f"expected at least 3 images in {directory}; found {len(paths)}")
    return paths


def write_json(path: Path, payload: object) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".receipt.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=identifier, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--source-masks", type=Path)
    parser.add_argument(
        "--sample-count", type=int,
        help="Create an explicitly labelled pilot subset, evenly spaced in basename capture order",
    )
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "input copying must run inside the verified Slurm allocation; "
            "refusing bulk relocation from the login node"
        )

    DESTINATION_ROOT.mkdir(parents=True, exist_ok=True)
    destination = DESTINATION_ROOT / args.dataset
    if destination.exists():
        raise RuntimeError(f"destination already exists; refusing to merge or overwrite: {destination}")
    sources = image_files(args.source_images.resolve(strict=True))
    original_image_count = len(sources)
    masks: list[Path] = []
    if args.source_masks:
        masks = image_files(args.source_masks.resolve(strict=True))
        source_root = args.source_images.resolve(strict=True)
        mask_root = args.source_masks.resolve(strict=True)
        source_relatives = {item.relative_to(source_root).as_posix() for item in sources}
        mask_relatives = {item.relative_to(mask_root).as_posix() for item in masks}
        if mask_relatives != source_relatives:
            raise RuntimeError("mask relative paths must exactly equal image relative paths")

    if args.sample_count is not None:
        if not 3 <= args.sample_count <= len(sources):
            raise RuntimeError("sample-count must be between 3 and the full input count")
        ordered = sorted(sources, key=lambda path: (path.name.casefold(), str(path)))
        indices = [round(index * (len(ordered) - 1) / (args.sample_count - 1))
                   for index in range(args.sample_count)]
        sources = [ordered[index] for index in indices]
        if masks:
            selected_relatives = {path.relative_to(source_root) for path in sources}
            masks = [path for path in masks if path.relative_to(mask_root) in selected_relatives]

    staging = DESTINATION_ROOT / f".{args.dataset}.{uuid.uuid4().hex}.staging"
    staging.mkdir(exist_ok=False)
    (staging / "images").mkdir()
    if masks:
        (staging / "masks").mkdir()
    records = []
    source_root = args.source_images.resolve(strict=True)
    for source in sources:
        relative = source.relative_to(source_root)
        target = staging / "images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash or source.stat().st_size != target.stat().st_size:
            raise RuntimeError(f"copy verification failed for {source}")
        records.append(
            {
                "path": (Path("images") / relative).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": target_hash,
            }
        )
    mask_root = args.source_masks.resolve(strict=True) if args.source_masks else None
    for source in masks:
        assert mask_root is not None
        relative = source.relative_to(mask_root)
        target = staging / "masks" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash or source.stat().st_size != target.stat().st_size:
            raise RuntimeError(f"copy verification failed for {source}")
        records.append(
            {
                "path": (Path("masks") / relative).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": target_hash,
            }
        )
    write_json(
        staging / "input_receipt.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "source_images": str(args.source_images.resolve(strict=True)),
            "source_masks": str(args.source_masks.resolve(strict=True)) if args.source_masks else None,
            "image_count": len(sources),
            "original_image_count": original_image_count,
            "selection": "evenly_spaced_basename_order_pilot" if args.sample_count else "all_images",
            "mask_count": len(masks),
            "files": records,
        },
    )
    os.replace(staging, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
