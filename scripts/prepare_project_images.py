#!/usr/bin/env python3
"""Build or install the checksum-verified CV802 image release bundle.

The archive is a transport artifact, not runtime storage.  On install, images
are copied into each engine's independent input namespace under
``CV802_DATA_ROOT``.  No file is written to the source-code checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
DATA_ROOT = Path(os.environ.get("CV802_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
DATASETS = ("light_shirt", "dark_shirt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_NAME = "manifest.json"


def sha256_stream(stream, output=None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while block := stream.read(4 * 1024 * 1024):
        digest.update(block)
        size += len(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest(), size


def safe_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def build_bundle(destination: Path, light: Path, dark: Path) -> None:
    """Create the portable ZIP and embed each file's SHA-256 and byte size."""
    destination = destination.expanduser().resolve()
    records: dict[str, dict[str, object]] = {}
    sources: dict[str, list[Path]] = {}
    for dataset, source in (("light_shirt", light), ("dark_shirt", dark)):
        source = source.expanduser().resolve(strict=True)
        if not source.is_dir():
            raise ValueError(f"not an image directory: {source}")
        paths = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"no supported images found in {source}")
        sources[dataset] = paths
        records[dataset] = {"image_count": len(paths), "files": []}

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"schema_version": 1, "datasets": records}
    roots = {"light_shirt": light.expanduser().resolve(), "dark_shirt": dark.expanduser().resolve()}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as bundle:
        for dataset, paths in sources.items():
            rows = records[dataset]["files"]
            assert isinstance(rows, list)
            for source in paths:
                relative = source.relative_to(roots[dataset]).as_posix()
                archive_name = f"{dataset}/images/{relative}"
                if any(row["name"] == archive_name for row in rows):
                    raise ValueError(f"duplicate image basename in {dataset}: {relative}")
                digest = hashlib.sha256()
                size = 0
                with source.open("rb") as reader, bundle.open(archive_name, "w") as writer:
                    while block := reader.read(4 * 1024 * 1024):
                        writer.write(block)
                        digest.update(block)
                        size += len(block)
                rows.append({"name": archive_name, "bytes": size, "sha256": digest.hexdigest()})
        bundle.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Created {destination} ({destination.stat().st_size:,} bytes)")
    print(json.dumps({name: len(rows) for name, rows in sources.items()}, sort_keys=True))


def read_manifest(bundle: zipfile.ZipFile) -> dict[str, object]:
    try:
        manifest = json.loads(bundle.read(MANIFEST_NAME))
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("image ZIP has no valid manifest.json") from error
    if manifest.get("schema_version") != 1 or set(manifest.get("datasets", {})) != set(DATASETS):
        raise ValueError("unsupported or incomplete image manifest")
    listed: set[str] = set()
    names = set(bundle.namelist())
    for dataset in DATASETS:
        entry = manifest["datasets"][dataset]
        files = entry.get("files")
        if not isinstance(files, list) or len(files) != entry.get("image_count") or not files:
            raise ValueError(f"invalid image list for {dataset}")
        for row in files:
            name = row.get("name")
            member = safe_member(str(name))
            if len(member.parts) < 3 or member.parts[:2] != (dataset, "images"):
                raise ValueError(f"unexpected image path: {name!r}")
            if member.suffix.lower() not in IMAGE_EXTENSIONS or name not in names or name in listed:
                raise ValueError(f"missing, duplicate, or unsupported archive member: {name!r}")
            listed.add(name)
    if names != listed | {MANIFEST_NAME}:
        raise ValueError("image ZIP contains unmanifested entries")
    return manifest


def install_bundle(archive: Path) -> None:
    """Verify and atomically install images in the SfM and VGGSfM inputs."""
    if not DATA_ROOT.is_absolute():
        raise ValueError("CV802_DATA_ROOT must be an absolute path")
    data_root = DATA_ROOT.resolve()
    code_root = Path(__file__).resolve().parents[1]
    if data_root == code_root or code_root in data_root.parents:
        raise ValueError("CV802_DATA_ROOT must be outside the source-code checkout")
    destinations = {
        (dataset, method): data_root / method / "inputs" / dataset / "images"
        for dataset in DATASETS for method in ("sfm", "vggsfm")
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing image inputs: " + ", ".join(map(str, existing))
        )
    staged: list[tuple[Path, Path]] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            manifest = read_manifest(bundle)
            for (dataset, method), destination in destinations.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=".images-install-", dir=destination.parent))
                staged.append((temporary, destination))
                for row in manifest["datasets"][dataset]["files"]:
                    name = row["name"]
                    relative = PurePosixPath(name).relative_to(PurePosixPath(dataset) / "images")
                    output_path = temporary.joinpath(*relative.parts)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(name) as reader, output_path.open("wb") as writer:
                        digest, size = sha256_stream(reader, writer)
                    if size != row["bytes"] or digest != row["sha256"]:
                        raise ValueError(f"checksum or byte-count mismatch: {name}")
                if sum(1 for path in temporary.rglob("*") if path.is_file()) != manifest["datasets"][dataset]["image_count"]:
                    raise ValueError(f"installed image count mismatch: {dataset}")
        # Publish only after every image has been read and checksum-verified.
        for temporary, destination in staged:
            temporary.rename(destination)
        print(f"Installed verified images below {data_root}")
        for (dataset, method), destination in destinations.items():
            print(f"  {method}/{dataset}: {destination} ({sum(1 for p in destination.rglob('*') if p.is_file())} images)")
    except Exception:
        for temporary, destination in staged:
            if temporary.exists():
                shutil.rmtree(temporary)
            # Roll back only destinations published during this invocation.
            if destination.exists() and not temporary.exists():
                shutil.rmtree(destination)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the image bundle from existing image folders")
    build.add_argument("--light-images", type=Path, required=True)
    build.add_argument("--dark-images", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    install = commands.add_parser("install", help="verify and stage a downloaded image bundle")
    install.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_bundle(args.output, args.light_images, args.dark_images)
    else:
        install_bundle(args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
