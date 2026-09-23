"""Exact provenance-map matching between E10 and official VGGSfM names."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .colmap_source import CameraPose


class MappingError(RuntimeError):
    """The immutable VGGSfM request map is absent or ambiguous."""


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise MappingError(f"{label} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise MappingError(f"unsafe {label}: {value!r}")
    return path


def load_official_name_map(request_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Return official VGGSfM name -> original nested image name."""

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MappingError(f"cannot read VGGSfM request {request_path}: {error}") from error
    if not isinstance(payload, dict):
        raise MappingError("VGGSfM request root must be a JSON object")
    records = payload.get("official_image_name_map")
    if not isinstance(records, list) or len(records) < 3:
        raise MappingError("VGGSfM request has fewer than three image-name map entries")

    mapping: dict[str, str] = {}
    original_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MappingError(f"image-name map entry {index} is not an object")
        official_path = _safe_relative(
            record.get("official"), label=f"official name at entry {index}"
        )
        source_path = _safe_relative(
            record.get("source"), label=f"source name at entry {index}"
        )
        if len(source_path.parts) < 2 or source_path.parts[0] != "images":
            raise MappingError(
                f"source name at entry {index} must start with 'images/': {source_path}"
            )
        original = PurePosixPath(*source_path.parts[1:]).as_posix()
        official = official_path.as_posix()
        if official in mapping:
            raise MappingError(f"duplicate official image name in request: {official}")
        if original in original_names:
            raise MappingError(f"duplicate original image name in request: {original}")
        mapping[official] = original
        original_names.add(original)
    return mapping, payload


@dataclass(frozen=True)
class CameraMatches:
    vggsfm_centers: np.ndarray
    e10_centers: np.ndarray
    names: tuple[tuple[str, str], ...]
    e10_registered_count: int
    vggsfm_registered_count: int
    map_entry_count: int
    vggsfm_originals_missing_from_e10: tuple[str, ...]
    e10_images_missing_from_vggsfm: tuple[str, ...]

    def coverage(self) -> dict[str, float | int]:
        matched = len(self.names)
        return {
            "matched_camera_count": matched,
            "e10_registered_camera_count": self.e10_registered_count,
            "vggsfm_registered_camera_count": self.vggsfm_registered_count,
            "request_map_entry_count": self.map_entry_count,
            "fraction_of_e10_registered": matched / self.e10_registered_count,
            "fraction_of_vggsfm_registered": matched / self.vggsfm_registered_count,
            "fraction_of_request_map": matched / self.map_entry_count,
        }


def match_camera_centers(
    e10_poses: tuple[CameraPose, ...],
    vggsfm_poses: tuple[CameraPose, ...],
    official_to_original: dict[str, str],
) -> CameraMatches:
    """Match only by the recorded exact name map; never guess by basename."""

    e10_by_name = {pose.name: pose for pose in e10_poses}
    vgg_by_name = {pose.name: pose for pose in vggsfm_poses}
    if len(e10_by_name) != len(e10_poses):
        raise MappingError("E10 model has duplicate registered image names")
    if len(vgg_by_name) != len(vggsfm_poses):
        raise MappingError("VGGSfM model has duplicate registered image names")

    missing_map = sorted(set(vgg_by_name) - set(official_to_original))
    if missing_map:
        rendered = ", ".join(missing_map[:5])
        raise MappingError(
            "registered VGGSfM images are absent from the immutable request map: "
            f"{rendered}"
        )

    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    names: list[tuple[str, str]] = []
    missing_from_e10: list[str] = []
    matched_e10: set[str] = set()
    for official_name in sorted(vgg_by_name):
        original_name = official_to_original[official_name]
        e10_pose = e10_by_name.get(original_name)
        if e10_pose is None:
            missing_from_e10.append(original_name)
            continue
        source_rows.append(vgg_by_name[official_name].center)
        target_rows.append(e10_pose.center)
        names.append((official_name, original_name))
        matched_e10.add(original_name)

    if len(names) < 3:
        raise MappingError(
            f"only {len(names)} registered cameras match between VGGSfM and E10"
        )
    source = np.asarray(source_rows, dtype=np.float64)
    target = np.asarray(target_rows, dtype=np.float64)
    if source.shape != target.shape or source.shape[1:] != (3,):
        raise MappingError("internal camera correspondence shape mismatch")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise MappingError("matched camera centers contain NaN or infinity")

    return CameraMatches(
        vggsfm_centers=source,
        e10_centers=target,
        names=tuple(names),
        e10_registered_count=len(e10_poses),
        vggsfm_registered_count=len(vggsfm_poses),
        map_entry_count=len(official_to_original),
        vggsfm_originals_missing_from_e10=tuple(sorted(missing_from_e10)),
        e10_images_missing_from_vggsfm=tuple(sorted(set(e10_by_name) - matched_e10)),
    )
