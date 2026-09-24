"""Verify and stage Git-bundled reviewed MVS inputs into the active data root."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Callable


CODE_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = CODE_ROOT / "datasets" / "reviewed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path: Path, record: dict) -> None:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]):
        raise ValueError(f"Bundled reviewed asset failed checksum: {path}")


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe reviewed asset path: {value}")
    return path


def _verify_install(destination: Path, manifest: dict, masks: list[dict]) -> None:
    check(destination / "input_provenance.json", manifest["input_provenance"])
    for name, record in manifest["sparse_files"].items():
        if safe_relative(name).name != name:
            raise ValueError("Sparse model name is not a filename")
        check(destination / "sparse" / name, record)
    if manifest["dataset"] == "light_shirt":
        check(destination / "mask_manifest.json", manifest["mask_manifest"])
        for row in masks:
            relative = safe_relative(row["mask_relative"])
            if relative.parts[0] != "masks":
                raise ValueError("Mask outside masks/")
            check(destination / relative, row)


def install_mvs_reference(dataset: str, data_root: Path, output: Callable[[str], None]) -> Path:
    """Stage E10/E3 cameras (and light masks) without any legacy data path."""
    asset = BUNDLE / "mvs" / dataset
    manifest_path = asset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Repository is missing reviewed MVS assets: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset") != dataset or not isinstance(manifest.get("sparse_files"), dict):
        raise ValueError("Invalid reviewed MVS manifest")
    archive = asset / "sparse.tar.gz"
    check(archive, manifest["sparse_archive"])
    check(asset / "input_provenance.json", manifest["input_provenance"])
    masks: list[dict] = []
    if dataset == "light_shirt":
        check(asset / "mask_manifest.json", manifest["mask_manifest"])
        mask_asset = BUNDLE / "mac_masks" / dataset
        receipt = json.loads((mask_asset / "receipt.json").read_text())
        if receipt.get("status") != "complete" or receipt.get("mask_count") != len(receipt.get("rows", [])):
            raise ValueError("Bundled Mac mask receipt is incomplete")
        masks = receipt["rows"]
        mvs_manifest = json.loads((asset / "mask_manifest.json").read_text())
        expected = {row["image_relative"]: row["sha256"] for row in masks}
        actual = {name: row["sha256"] for name, row in mvs_manifest["images"].items()}
        if expected != actual:
            raise ValueError("MVS and VGGSfM bundled light masks differ")
        for row in masks:
            relative = safe_relative(row["mask_relative"])
            check(mask_asset / relative, row)

    parent = data_root / "mvs" / "reference_inputs"
    destination = parent / dataset
    if destination.exists():
        _verify_install(destination, manifest, masks)
        output(f"Reused Git-bundled {dataset} E10/E3 MVS calibration")
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-reviewed-", dir=parent) as temporary:
        staging = Path(temporary)
        (staging / "sparse").mkdir()
        expected_names = {f"sparse/{name}" for name in manifest["sparse_files"]}
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if {member.name for member in members} != expected_names or any(
                not member.isfile() or member.issym() or member.islnk() for member in members
            ):
                raise ValueError("Sparse archive contains unexpected entries")
            for member in members:
                name = Path(member.name).name
                record = manifest["sparse_files"][name]
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                target = staging / "sparse" / name
                with target.open("wb") as writer:
                    shutil.copyfileobj(stream, writer)
                check(target, record)
        shutil.copyfile(asset / "input_provenance.json", staging / "input_provenance.json")
        if dataset == "light_shirt":
            shutil.copyfile(asset / "mask_manifest.json", staging / "mask_manifest.json")
            mask_asset = BUNDLE / "mac_masks" / dataset
            for row in masks:
                relative = safe_relative(row["mask_relative"])
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mask_asset / relative, target)
        _verify_install(staging, manifest, masks)
        if destination.exists():
            raise FileExistsError(f"Another process created {destination}")
        os.replace(staging, destination)
    output(f"Installed Git-bundled {dataset} E10/E3 MVS calibration")
    return destination
