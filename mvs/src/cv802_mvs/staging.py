"""Self-contained, checksum-verified MVS input staging."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

from .colmap_model import image_dimensions, load_model_records, referenced_image_paths
from .constants import SCHEMA_VERSION
from .errors import InputValidationError, StageConflictError
from .io_utils import atomic_json, sha256_file, stable_digest, utc_now
from .lifecycle import archive_existing
from .paths import PathPolicy, safe_experiment_name


def _verified_copy(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one stable file, rejecting concurrent source modification."""
    source = source.resolve(strict=True)
    before = source.stat()
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise InputValidationError(f"Source changed while it was copied: {source}")
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash or destination.stat().st_size != before.st_size:
        raise InputValidationError(f"Checksum verification failed while copying {source}")
    if destination.is_symlink():
        raise InputValidationError(f"Staged destination unexpectedly remained a symlink: {destination}")
    return {
        "source_path": str(source),
        "destination_relative": destination.as_posix(),
        "size_bytes": before.st_size,
        "source_mtime_ns_before": before.st_mtime_ns,
        "source_mtime_ns_after": after.st_mtime_ns,
        "sha256": source_hash,
    }


def _mask_source(
    masks_root: Path, source_manifest: Path, record: dict[str, Any], image_name: str
) -> Path:
    expected_relative = Path(image_name).with_suffix(".png")
    candidates = [masks_root / expected_relative]
    manifest_relative = record.get("mask_relative_path")
    if isinstance(manifest_relative, str):
        relative = Path(manifest_relative)
        if not relative.is_absolute() and ".." not in relative.parts:
            candidates.append(masks_root / relative)
            if relative.parts and relative.parts[0] == "masks":
                candidates.append(masks_root.joinpath(*relative.parts[1:]))
            candidates.append(source_manifest.parent / relative)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_file():
            return resolved
    raise InputValidationError(f"No mask file found for registered image {image_name!r}")


def _verify_existing_provenance(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageConflictError(f"Cannot resume staged inputs: {error}") from error
    root = path.parents[1]
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise StageConflictError("Cannot resume: input provenance has no file records")
    for record in records:
        if not isinstance(record, dict):
            raise StageConflictError("Cannot resume: malformed input provenance record")
        destination = root / str(record.get("destination_relative", ""))
        try:
            destination.resolve(strict=True).relative_to(root.resolve())
        except (FileNotFoundError, ValueError) as error:
            raise StageConflictError(f"Cannot resume: missing or unsafe staged file {destination}") from error
        if destination.stat().st_size != record.get("size_bytes"):
            raise StageConflictError(f"Cannot resume: size changed for {destination}")
        if sha256_file(destination) != record.get("sha256"):
            raise StageConflictError(f"Cannot resume: hash changed for {destination}")
    calculated_digest = stable_digest(
        [
            [record.get("destination_relative"), record.get("size_bytes"), record.get("sha256")]
            for record in records
        ]
    )
    if calculated_digest != manifest.get("files_digest"):
        raise StageConflictError("Cannot resume: input provenance record digest changed")
    return manifest


def verify_staged_inputs(experiment_root: Path) -> dict[str, Any]:
    """Public integrity check used before every dense run."""
    return _verify_existing_provenance(
        experiment_root / "manifests" / "input_provenance.json"
    )


def stage_inputs(
    *,
    policy: PathPolicy,
    experiment: str,
    images_source: Path,
    model_source: Path,
    masks_source: Path | None = None,
    mask_manifest_source: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy the registered model/images/masks into an independent MVS experiment."""
    if resume and overwrite:
        raise StageConflictError("--resume and --overwrite are mutually exclusive")
    safe_experiment_name(experiment)
    experiment_root = policy.experiment(experiment)
    experiment_root.mkdir(parents=True, exist_ok=True)
    provenance_path = experiment_root / "manifests" / "input_provenance.json"
    inputs = experiment_root / "inputs"
    if inputs.exists() or provenance_path.exists():
        if resume:
            return _verify_existing_provenance(provenance_path)
        if not overwrite:
            raise StageConflictError(
                f"Staged inputs already exist in {experiment_root}; use --resume or --overwrite"
            )
        archive_existing(experiment_root, reason="restage")

    images_root = policy.require_data_source(images_source, "images source")
    model_root = policy.require_data_source(model_source, "sparse model source")
    masks_root = (
        policy.require_data_source(masks_source, "masks source") if masks_source else None
    )
    source_manifest = (
        policy.require_data_source(mask_manifest_source, "mask manifest source")
        if mask_manifest_source
        else None
    )
    if (masks_root is None) != (source_manifest is None):
        raise InputValidationError(
            "masks_source and mask_manifest_source must be supplied together"
        )

    model = load_model_records(model_root)
    image_sources = referenced_image_paths(model, images_root)
    staging = experiment_root / f".inputs-staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    file_records: list[dict[str, Any]] = []
    try:
        for name, source in sorted(image_sources.items()):
            destination = staging / "images" / Path(name)
            record = _verified_copy(source, destination)
            record["destination_relative"] = (Path("inputs") / "images" / name).as_posix()
            record["role"] = "registered_image"
            file_records.append(record)

        # Copy every regular file in the resolved model directory, including
        # optional rigs/frames files, but never a database or parent workspace.
        for source in sorted(item for item in model.directory.iterdir() if item.is_file()):
            destination = staging / "sparse" / source.name
            record = _verified_copy(source, destination)
            record["destination_relative"] = (
                Path("inputs") / "sparse" / source.name
            ).as_posix()
            record["role"] = "sparse_model"
            file_records.append(record)

        rebased_mask_manifest: dict[str, Any] | None = None
        if masks_root and source_manifest:
            try:
                original_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise InputValidationError(f"Cannot read source mask manifest: {error}") from error
            original_records = original_manifest.get("images")
            if not original_manifest.get("complete") or not isinstance(original_records, dict):
                raise InputValidationError("Mask manifest must be complete and contain an images map")
            rebased_records: dict[str, Any] = {}
            for image in model.images:
                original = original_records.get(image.name)
                if not isinstance(original, dict) or original.get("usable") is not True:
                    raise InputValidationError(
                        f"Mask manifest has no usable mask for {image.name!r}"
                    )
                source = policy.require_data_source(
                    _mask_source(masks_root, source_manifest, original, image.name),
                    f"mask for {image.name}",
                )
                destination_relative = Path(image.name).with_suffix(".png")
                destination = staging / "masks" / destination_relative
                record = _verified_copy(source, destination)
                expected_hash = original.get("mask_sha256")
                if expected_hash and record["sha256"] != expected_hash:
                    raise InputValidationError(
                        f"Mask hash disagrees with historical manifest for {image.name}"
                    )
                image_size = image_dimensions(staging / "images" / Path(image.name))
                mask_size = image_dimensions(destination)
                if image_size != mask_size:
                    raise InputValidationError(
                        f"Mask/image dimensions disagree for {image.name}: {mask_size} vs {image_size}"
                    )
                record["destination_relative"] = (
                    Path("inputs") / "masks" / destination_relative
                ).as_posix()
                record["role"] = "foreground_mask"
                file_records.append(record)
                rebased_records[image.name] = {
                    "mask_relative_path": (Path("masks") / destination_relative).as_posix(),
                    "sha256": record["sha256"],
                    "width": image_size[0],
                    "height": image_size[1],
                    "usable": True,
                    "original_status": original.get("status"),
                }
            original_destination = staging / "provenance" / "original_mask_manifest.json"
            original_record = _verified_copy(source_manifest, original_destination)
            original_record["destination_relative"] = (
                Path("inputs") / "provenance" / "original_mask_manifest.json"
            ).as_posix()
            original_record["role"] = "historical_mask_manifest"
            file_records.append(original_record)
            rebased_mask_manifest = {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "ready_for_dense_input_masking": True,
                "created_at": utc_now(),
                "value_semantics": original_manifest.get("method", {}).get("value_semantics"),
                "pixel_coordinates": original_manifest.get("method", {}).get("pixel_coordinates"),
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": sha256_file(source_manifest),
                },
                "images": rebased_records,
            }
            atomic_json(staging / "mask_manifest.json", rebased_mask_manifest)
            generated_mask_manifest = staging / "mask_manifest.json"
            file_records.append(
                {
                    "source_path": None,
                    "destination_relative": "inputs/mask_manifest.json",
                    "size_bytes": generated_mask_manifest.stat().st_size,
                    "sha256": sha256_file(generated_mask_manifest),
                    "role": "rebased_mask_manifest",
                }
            )

        # Validate the independently copied model before publishing the staging
        # directory.  The rename makes readers see either no inputs or all inputs.
        staged_model = load_model_records(staging / "sparse")
        referenced_image_paths(staged_model, staging / "images")
        os.replace(staging, inputs)
    except Exception:
        failed = experiment_root / "archive" / f"failed-staging-{uuid.uuid4().hex}"
        failed.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            os.replace(staging, failed)
        raise

    total_bytes = sum(int(record["size_bytes"]) for record in file_records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "experiment": experiment,
        "method": "COLMAP CUDA PatchMatch Stereo",
        "source": {
            "images": str(images_root),
            "sparse_model_requested": str(model_root),
            "sparse_model_resolved": str(model.directory),
            "sparse_model_format": model.format,
            "masks": str(masks_root) if masks_root else None,
            "mask_manifest": str(source_manifest) if source_manifest else None,
        },
        "registered_image_count": len(model.images),
        "camera_count": len(model.cameras),
        "file_count": len(file_records),
        "total_bytes": total_bytes,
        "symlinks_in_destination": 0,
        "files_digest": stable_digest(
            [
                [record["destination_relative"], record["size_bytes"], record["sha256"]]
                for record in file_records
            ]
        ),
        "files": file_records,
    }
    atomic_json(provenance_path, manifest)
    # Re-read every published file by hash before reporting success.
    return _verify_existing_provenance(provenance_path)
