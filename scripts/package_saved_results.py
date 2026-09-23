#!/usr/bin/env python3
"""Package/install the small, image-free point clouds shown by the UI.

The bundle contains only catalogued PLYs and the manifests/publication records
needed for integrity checks. Archive paths mirror DATA_ROOT so the existing
read-only browser can use them without the original photographs.
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

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
DATA_ROOT = Path(os.environ.get("CV802_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
MANIFEST_NAME = "_cv802_saved_results_manifest.json"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def _catalog_files(data_root: Path) -> tuple[dict[str, Path], set[Path]]:
    history = json.loads((CODE_ROOT / "sfm/configs/historical_e1_e10.json").read_text())
    views = json.loads((CODE_ROOT / "sfm/configs/desktop_saved_views.json").read_text())
    files: dict[str, Path] = {}
    publication_targets: set[Path] = set()

    def add(relative: str) -> Path:
        rel = safe_name(relative)
        target = (data_root / Path(*rel.parts)).resolve(strict=True)
        try:
            target.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"catalog path escapes CV802_DATA_ROOT: {relative}") from error
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"result asset is not a regular file: {target}")
        files[rel.as_posix()] = target
        return target

    for experiment in history["experiments"]:
        for result in experiment["subjects"].values():
            if result.get("available"):
                add("sfm/historical/transferred/sfm/reconstructions/" + result["relative_ply"])

    for view in views["views"]:
        for result in view["subjects"].values():
            if not result.get("available"):
                continue
            ply = add(result["relative_ply"])
            if result.get("manifest_relative"):
                add(result["manifest_relative"])
            if result.get("publication_relative"):
                publication_targets.add(add(result["publication_relative"]))
    return files, publication_targets


def build_bundle(archive_path: Path) -> None:
    if not DATA_ROOT.is_absolute():
        raise ValueError("CV802_DATA_ROOT must be an absolute path")
    data_root = DATA_ROOT.resolve(strict=True)
    files, publication_files = _catalog_files(data_root)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "description": "Catalogued E1-E10, MVS, and VGGSfM display point clouds; no source photos.",
        "files": [],
    }
    rows: list[dict[str, object]] = []
    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, source in sorted(files.items()):
            data = source.read_bytes()
            if source.suffix == ".json":
                publication = json.loads(data)
                if source in publication_files:
                    point_path = source.parent / "point_cloud.ply"
                    point_relative = point_path.name
                    cloud = publication.get("point_cloud", {})
                    cloud["path"] = point_relative
                    for row in publication.get("artifacts", []):
                        if Path(str(row.get("path", ""))).name == point_path.name and row.get("sha256") == cloud.get("sha256"):
                            row["path"] = point_relative
                if isinstance(publication.get("files"), list):
                    for row in publication["files"]:
                        if Path(str(row.get("path", ""))).name == "point_cloud.ply":
                            row["path"] = "point_cloud.ply"
                data = (json.dumps(publication, indent=2, sort_keys=True) + "\n").encode()
            bundle.writestr(name, data)
            rows.append({"path": name, "bytes": len(data), "sha256": digest_bytes(data)})
        manifest["files"] = rows
        bundle.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Created {archive_path} ({archive_path.stat().st_size:,} bytes; {len(rows)} files)")


def _read_manifest(bundle: zipfile.ZipFile) -> list[dict[str, object]]:
    try:
        payload = json.loads(bundle.read(MANIFEST_NAME))
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("saved-result ZIP has no valid manifest") from error
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError("unsupported saved-result manifest")
    rows = payload["files"]
    names = {safe_name(str(row.get("path", ""))).as_posix() for row in rows}
    if len(names) != len(rows) or set(bundle.namelist()) != names | {MANIFEST_NAME}:
        raise ValueError("saved-result ZIP has duplicate, missing, or unmanifested entries")
    return rows


def install_bundle(archive_path: Path) -> None:
    if not DATA_ROOT.is_absolute():
        raise ValueError("CV802_DATA_ROOT must be an absolute path")
    data_root = DATA_ROOT.resolve()
    if CODE_ROOT == data_root or CODE_ROOT in data_root.parents:
        raise ValueError("CV802_DATA_ROOT must be outside the source-code checkout")
    with zipfile.ZipFile(archive_path) as bundle:
        rows = _read_manifest(bundle)
        destinations: list[tuple[str, Path, dict[str, object]]] = []
        for row in rows:
            name = str(row["path"])
            relative = safe_name(name)
            destination = (data_root / Path(*relative.parts)).resolve(strict=False)
            try:
                destination.relative_to(data_root)
            except ValueError as error:
                raise ValueError(f"asset escapes CV802_DATA_ROOT: {name}") from error
            destinations.append((name, destination, row))
        existing = [str(path) for _, path, _ in destinations if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite saved results: " + ", ".join(existing[:5]))

        staged: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for name, destination, row in destinations:
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(prefix=".cv802-result-", dir=destination.parent)
                os.close(fd)
                temporary = Path(temp_name)
                staged.append((temporary, destination))
                digest = hashlib.sha256()
                size = 0
                with bundle.open(name) as source, temporary.open("wb") as output:
                    while block := source.read(4 * 1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"checksum or byte-count mismatch: {name}")
            for temporary, destination in staged:
                temporary.rename(destination)
                published.append(destination)
        except Exception:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            for destination in published:
                destination.unlink(missing_ok=True)
            raise
    print(f"Installed {len(destinations)} verified saved-result files under {data_root}")
    print("Start sfm/browse_historical.py to view them. No input photos are needed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="package saved result files using the project catalogs")
    build.add_argument("--output", type=Path, required=True)
    install = commands.add_parser("install", help="verify and install a saved-result ZIP")
    install.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_bundle(args.output)
    else:
        install_bundle(args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
