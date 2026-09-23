"""Optional foreground-retention preprocessing in distorted image coordinates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any

from .colmap_model import load_model_records, referenced_image_paths
from .config import ExperimentPaths, MVSConfig
from .errors import InputValidationError, StageConflictError
from .io_utils import atomic_json, sha256_file, stable_digest, utc_now


def create_masked_images(config: MVSConfig, paths: ExperimentPaths) -> dict[str, Any]:
    """Black out non-person pixels before COLMAP undistorts the images.

    The saved Apple Vision masks live in the original *distorted* source pixel
    grid.  Applying them here (before ``image_undistorter``) preserves that
    coordinate contract.  This is input masking, not the historical sparse
    multi-view point-consensus filter.
    """
    if config.masking_mode != "black_background":
        raise InputValidationError("Masked-image stage requested while masking.mode is not enabled")
    if paths.masks is None or paths.mask_manifest is None:
        raise InputValidationError("Mask paths are missing from configuration")
    try:
        from PIL import Image, ImageFilter
    except ImportError as error:
        raise InputValidationError(
            "Pillow is required for masking.mode=black_background; install the pinned MVS environment"
        ) from error

    try:
        manifest = json.loads(paths.mask_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputValidationError(f"Cannot read staged mask manifest: {error}") from error
    records = manifest.get("images")
    if not manifest.get("complete") or not isinstance(records, dict):
        raise InputValidationError("Staged mask manifest is incomplete")

    model = load_model_records(paths.model)
    images = referenced_image_paths(model, paths.images)
    output = paths.masked_images
    if output.exists():
        partial = paths.root / "archive" / f"partial-masking-{uuid.uuid4().hex}"
        partial.parent.mkdir(parents=True, exist_ok=True)
        os.replace(output, partial)
    temporary = paths.work / f".masked-images-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    output_records: list[dict[str, Any]] = []
    try:
        for name, image_path in sorted(images.items()):
            record = records.get(name)
            if not isinstance(record, dict) or record.get("usable") is not True:
                raise InputValidationError(f"No approved staged mask for {name!r}")
            mask_relative = record.get("mask_relative_path")
            if not isinstance(mask_relative, str):
                raise InputValidationError(f"Mask record has no relative path for {name!r}")
            mask_path = (paths.mask_manifest.parent / mask_relative).resolve(strict=True)
            try:
                mask_path.relative_to(paths.root.resolve())
            except ValueError as error:
                raise InputValidationError(f"Mask path escapes experiment for {name!r}") from error
            if sha256_file(mask_path) != record.get("sha256"):
                raise InputValidationError(f"Staged mask hash changed for {name!r}")
            with Image.open(image_path) as source_image, Image.open(mask_path) as source_mask:
                image = source_image.convert("RGB")
                mask = source_mask.convert("L")
                if image.size != mask.size:
                    raise InputValidationError(
                        f"Mask dimensions {mask.size} do not match image {image.size} for {name}"
                    )
                # A binary foreground decision is explicit and reproducible.
                mask = mask.point(
                    lambda value: 255 if value >= config.mask_threshold else 0,
                    mode="L",
                )
                if config.mask_dilation_pixels:
                    mask = mask.filter(
                        ImageFilter.MaxFilter(2 * config.mask_dilation_pixels + 1)
                    )
                histogram = mask.histogram()
                foreground_pixels = sum(histogram[1:])
                pixel_count = image.width * image.height
                foreground_fraction = foreground_pixels / pixel_count
                if not 0.0001 < foreground_fraction < 0.98:
                    raise InputValidationError(
                        f"Implausible foreground fraction {foreground_fraction:.6f} for {name}; "
                        "check mask polarity and threshold"
                    )
                black = Image.new("RGB", image.size, (0, 0, 0))
                masked = Image.composite(image, black, mask)
                destination = temporary / Path(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                suffix = destination.suffix.lower()
                if suffix in {".jpg", ".jpeg"}:
                    masked.save(destination, format="JPEG", quality=95, subsampling=0)
                elif suffix == ".png":
                    masked.save(destination, format="PNG", compress_level=6)
                else:
                    masked.save(destination)
            output_records.append(
                {
                    "image": name,
                    "source_sha256": sha256_file(image_path),
                    "mask_sha256": record["sha256"],
                    "output_sha256": sha256_file(destination),
                    "output_size_bytes": destination.stat().st_size,
                    "foreground_fraction_after_threshold_and_dilation": foreground_fraction,
                }
            )
        os.replace(temporary, output)
    except Exception:
        failed = paths.root / "archive" / f"failed-masking-{uuid.uuid4().hex}"
        failed.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            os.replace(temporary, failed)
        raise

    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "mode": "black_background",
        "coordinate_system": "original distorted image pixels; mask applied before undistortion",
        "foreground_semantics": "values >= threshold retained; background set to RGB zero",
        "not_equivalent_to": "historical sparse 90% multi-view point-consensus cleanup",
        "threshold": config.mask_threshold,
        "dilation_pixels_at_native_resolution": config.mask_dilation_pixels,
        "image_count": len(output_records),
        "records_digest": stable_digest(output_records),
        "images": output_records,
    }
    atomic_json(paths.manifests / "masked_images.json", result)
    return result

