#!/usr/bin/env python3
"""One-time maintainer packaging of the reviewed assignment inputs.

Runtime never invokes this file: the generated, verified assets are committed
to Git. It is here to document exactly how the original data was normalized.
Run only with an explicit prior assignment data root, outside the checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


EXPERIMENTS = {
    "light_shirt": "light_e10_colmap_mvs_1600",
    "dark_shirt": "dark_e3_colmap_mvs_1024_raw_v1",
}
MASK_SOURCES = {
    "light_shirt": "light_shirt_cleanup_masks",
    "dark_shirt": "dark_shirt_cleanup_inputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def copy_verified(source: Path, destination: Path, expected_hash: str | None = None) -> dict:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Missing or symlinked source: {source}")
    digest = sha256(source)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"Original receipt checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(destination) != digest or destination.stat().st_size != source.stat().st_size:
        raise ValueError(f"Copied asset checksum mismatch: {destination}")
    return {"bytes": destination.stat().st_size, "sha256": digest}


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe path in historical receipt: {value}")
    return path


def package_masks(prior: Path, output: Path, dataset: str) -> dict:
    source = prior / "vggsfm" / "inputs" / MASK_SOURCES[dataset]
    receipt = json.loads((source / "receipt.json").read_text())
    rows = receipt.get("rows", receipt.get("masks"))
    if receipt.get("status") != "complete" or len(rows) != receipt.get("mask_count"):
        raise ValueError(f"Incomplete historical {dataset} mask receipt")
    destination = output / "mac_masks" / dataset
    portable_rows = []
    for row in rows:
        relative = safe_relative(row["mask_relative"])
        if relative.parts[0] != "masks":
            raise ValueError("Mask outside masks/ folder")
        copied = copy_verified(source / relative, destination / relative, row["sha256"])
        if copied["bytes"] != row["bytes"]:
            raise ValueError("Mask size disagrees with receipt")
        portable_rows.append({key: row[key] for key in (
            "image_relative", "mask_relative", "bytes", "sha256", "width", "height", "mode",
        )})
    if len({row["image_relative"] for row in portable_rows}) != len(portable_rows):
        raise ValueError("Duplicate mask image names")
    result = {"schema_version": 1, "status": "complete", "dataset": dataset,
              "mask_count": len(portable_rows), "rows": portable_rows,
              "provenance": "Historical Apple Vision person masks, original image dimensions; no 3D geometry or poses."}
    if dataset == "dark_shirt":
        original = receipt["crutch_corridors"]
        relative = safe_relative(original["path"])
        copied = copy_verified(source / relative, destination / relative, original["sha256"])
        result["crutch_corridors"] = {"path": relative.as_posix(), **copied}
    write_json(destination / "receipt.json", result)
    return {"receipt": {"bytes": (destination / "receipt.json").stat().st_size,
                        "sha256": sha256(destination / "receipt.json")},
            "mask_count": len(portable_rows)}


def package_mvs(prior: Path, output: Path, dataset: str) -> dict:
    name = EXPERIMENTS[dataset]
    original = prior / "mvs" / "experiments" / name
    destination = output / "mvs" / dataset
    sparse = original / "inputs" / "sparse"
    files = sorted(path for path in sparse.iterdir() if path.is_file())
    if not {"cameras.bin", "images.bin", "points3D.bin"}.issubset(path.name for path in files):
        raise ValueError(f"Incomplete reviewed sparse model: {sparse}")
    destination.mkdir(parents=True, exist_ok=True)
    sparse_records = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                      for path in files}
    archive = destination / "sparse.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path in files:
            tar.add(path, arcname=f"sparse/{path.name}", recursive=False)
    archive_record = {"bytes": archive.stat().st_size, "sha256": sha256(archive)}
    provenance = json.loads((original / "manifests" / "input_provenance.json").read_text())
    images = [{key: row[key] for key in ("role", "destination_relative", "size_bytes", "sha256")}
              for row in provenance["files"] if row.get("role") == "registered_image"]
    if len(images) < 2:
        raise ValueError("No registered images in MVS provenance")
    write_json(destination / "input_provenance.json",
               {"schema_version": 1, "experiment": name, "files": images})
    manifest = None
    if dataset == "light_shirt":
        source_manifest = json.loads((original / "inputs" / "mask_manifest.json").read_text())
        # Retain the image-to-mask evidence, not the historical absolute path.
        source_manifest["source_manifest"] = {
            "sha256": source_manifest.get("source_manifest", {}).get("sha256"),
            "note": "Original Apple Vision segmentation receipt; portable masks are bundled separately.",
        }
        write_json(destination / "mask_manifest.json", source_manifest)
        manifest = {"bytes": (destination / "mask_manifest.json").stat().st_size,
                    "sha256": sha256(destination / "mask_manifest.json")}
    result = {"schema_version": 1, "dataset": dataset, "experiment": name,
              "sparse_archive": archive_record, "sparse_files": sparse_records,
              "input_provenance": {"bytes": (destination / "input_provenance.json").stat().st_size,
                                   "sha256": sha256(destination / "input_provenance.json")},
              "mask_manifest": manifest}
    write_json(destination / "manifest.json", result)
    if archive.stat().st_size >= 100_000_000:
        raise ValueError(f"Archive exceeds GitHub's per-file limit: {archive}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-data-root", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets" / "reviewed")
    args = parser.parse_args()
    if not args.prior_data_root.is_absolute():
        parser.error("prior data root must be absolute")
    output = args.output.resolve()
    summary = {dataset: {
        "masks": package_masks(args.prior_data_root, output, dataset),
        "mvs": package_mvs(args.prior_data_root, output, dataset),
    } for dataset in EXPERIMENTS}
    write_json(output / "manifest.json", {"schema_version": 1, "datasets": summary,
                                          "scope": "Assignment-specific Mac masks, 2D crutch corridors, and reviewed MVS sparse calibrations. Photos are in datasets/<dataset>/images."})
    print(json.dumps({dataset: {"masks": row["masks"]["mask_count"],
                                "sparse_archive_bytes": row["mvs"]["sparse_archive"]["bytes"]}
                      for dataset, row in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
