"""Dependency-free validation and conversion for COLMAP binary models.

The official VGGSfM runner writes a ``pycolmap.Reconstruction``.  This module
reads only the stable on-disk fields needed by our wrapper, avoiding a second
PyCOLMAP dependency in the orchestration process.  The binary layouts follow
COLMAP's public ``read_write_model.py`` contract.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import BinaryIO, Iterator

from .constants import REQUIRED_COLMAP_FILES
from .errors import OutputValidationError


@dataclass(frozen=True)
class Point3D:
    point_id: int
    x: float
    y: float
    z: float
    red: int
    green: int
    blue: int
    reprojection_error: float
    track_length: int
    track: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class RegisteredImage:
    image_id: int
    quaternion_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    camera_id: int
    name: str
    points3d_ids: tuple[int, ...]


# Stable camera identifiers and parameter layouts from COLMAP's public
# ``read_write_model.py``. Keeping this table local makes validation
# dependency-free and prevents an unknown layout from shifting all later
# binary records.
_CAMERA_MODELS: dict[int, tuple[str, int, tuple[int, ...]]] = {
    0: ("SIMPLE_PINHOLE", 3, (0,)),
    1: ("PINHOLE", 4, (0, 1)),
    2: ("SIMPLE_RADIAL", 4, (0,)),
    3: ("RADIAL", 5, (0,)),
    4: ("OPENCV", 8, (0, 1)),
    5: ("OPENCV_FISHEYE", 8, (0, 1)),
    6: ("FULL_OPENCV", 12, (0, 1)),
    7: ("FOV", 5, (0, 1)),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, (0,)),
    9: ("RADIAL_FISHEYE", 5, (0,)),
    10: ("THIN_PRISM_FISHEYE", 12, (0, 1)),
}

_FLOAT32_MAX = 3.4028234663852886e38
_INVALID_IMAGE_ID = (1 << 32) - 1
_INVALID_POINT3D_ID = (1 << 64) - 1
_MAX_IMAGE_NAME_BYTES = 1_048_576
_QUATERNION_NORM_TOLERANCE = 1e-3


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise OutputValidationError(
            f"Truncated COLMAP binary while reading {context}: "
            f"expected {size} bytes, got {len(data)}"
        )
    return data


def _read_u64(stream: BinaryIO, *, context: str) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8, context=context))[0]


def _validate_count_fits_file(
    *, count: int, file_size: int, minimum_record_size: int, context: str
) -> None:
    available = max(file_size - 8, 0)
    if count > available // minimum_record_size:
        raise OutputValidationError(
            f"COLMAP {context} declares {count} records, which cannot fit in "
            f"its {file_size}-byte file"
        )


def _require_end_of_file(stream: BinaryIO, *, context: str) -> None:
    if stream.read(1):
        raise OutputValidationError(
            f"COLMAP {context} contains unexpected trailing bytes"
        )


def _read_cameras(cameras_bin: Path) -> tuple[Camera, ...]:
    with cameras_bin.open("rb") as stream:
        count = _read_u64(stream, context="camera count")
        if count == 0:
            raise OutputValidationError("COLMAP cameras.bin contains zero cameras")
        _validate_count_fits_file(
            count=count,
            file_size=cameras_bin.stat().st_size,
            minimum_record_size=24,
            context="cameras.bin",
        )

        cameras: list[Camera] = []
        seen_ids: set[int] = set()
        for index in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ",
                _read_exact(stream, 24, context=f"camera {index} header"),
            )
            # COLMAP's public ID types use zero as a valid value.  In
            # particular, PyCOLMAP reconstructions written by VGGSfM number
            # cameras/images from zero.  Only negative values decoded from the
            # signed on-disk field are invalid here.
            if camera_id < 0:
                raise OutputValidationError(
                    f"COLMAP camera {index} has negative ID {camera_id}"
                )
            if camera_id in seen_ids:
                raise OutputValidationError(
                    f"COLMAP cameras.bin contains duplicate camera ID {camera_id}"
                )
            seen_ids.add(camera_id)
            if model_id not in _CAMERA_MODELS:
                raise OutputValidationError(
                    f"COLMAP camera {camera_id} uses unsupported model ID {model_id}"
                )
            model_name, parameter_count, focal_indices = _CAMERA_MODELS[model_id]
            if width == 0 or height == 0:
                raise OutputValidationError(
                    f"COLMAP camera {camera_id} has non-positive dimensions "
                    f"{width}x{height}"
                )
            parameters = struct.unpack(
                f"<{parameter_count}d",
                _read_exact(
                    stream,
                    parameter_count * 8,
                    context=f"camera {camera_id} parameters",
                ),
            )
            if not all(math.isfinite(value) for value in parameters):
                raise OutputValidationError(
                    f"COLMAP camera {camera_id} contains a non-finite intrinsic parameter"
                )
            if any(parameters[parameter_index] <= 0.0 for parameter_index in focal_indices):
                raise OutputValidationError(
                    f"COLMAP camera {camera_id} has a non-positive focal length"
                )
            cameras.append(
                Camera(
                    camera_id=camera_id,
                    model_id=model_id,
                    model_name=model_name,
                    width=width,
                    height=height,
                    parameters=tuple(parameters),
                )
            )
        _require_end_of_file(stream, context="cameras.bin")
    return tuple(cameras)


def _read_registered_images(images_bin: Path) -> tuple[RegisteredImage, ...]:
    file_size = images_bin.stat().st_size
    with images_bin.open("rb") as stream:
        count = _read_u64(stream, context="registered image count")
        if count == 0:
            raise OutputValidationError("COLMAP images.bin contains zero registered images")
        _validate_count_fits_file(
            count=count,
            file_size=file_size,
            minimum_record_size=73,  # pose + empty-name terminator + point count
            context="images.bin",
        )

        images: list[RegisteredImage] = []
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        for image_index in range(count):
            unpacked = struct.unpack(
                "<idddddddi",
                _read_exact(stream, 64, context=f"image {image_index} pose record"),
            )
            image_id = unpacked[0]
            quaternion = tuple(unpacked[1:5])
            translation = tuple(unpacked[5:8])
            camera_id = unpacked[8]
            if image_id < 0:
                raise OutputValidationError(
                    f"COLMAP image {image_index} has negative ID {image_id}"
                )
            if image_id in seen_ids:
                raise OutputValidationError(
                    f"COLMAP images.bin contains duplicate image ID {image_id}"
                )
            seen_ids.add(image_id)
            if camera_id < 0:
                raise OutputValidationError(
                    f"COLMAP image {image_id} references negative camera ID {camera_id}"
                )
            if not all(math.isfinite(value) for value in quaternion + translation):
                raise OutputValidationError(
                    f"COLMAP image {image_id} contains a non-finite pose value"
                )
            quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
            if not math.isclose(
                quaternion_norm,
                1.0,
                rel_tol=_QUATERNION_NORM_TOLERANCE,
                abs_tol=_QUATERNION_NORM_TOLERANCE,
            ):
                raise OutputValidationError(
                    f"COLMAP image {image_id} quaternion norm is "
                    f"{quaternion_norm:.9g}, expected 1"
                )

            name_bytes = bytearray()
            while True:
                byte = _read_exact(stream, 1, context=f"image {image_id} name")
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
                if len(name_bytes) > _MAX_IMAGE_NAME_BYTES:
                    raise OutputValidationError("COLMAP image name exceeds 1 MiB")
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OutputValidationError("COLMAP image name is not UTF-8") from exc
            name_path = PurePosixPath(name)
            if (
                not name
                or name_path.is_absolute()
                or any(part in {"", ".", ".."} for part in name_path.parts)
            ):
                raise OutputValidationError(
                    f"COLMAP image {image_id} has an unsafe or empty name: {name!r}"
                )
            if name in seen_names:
                raise OutputValidationError(
                    f"COLMAP images.bin contains duplicate image name {name!r}"
                )
            seen_names.add(name)

            point_count = _read_u64(stream, context=f"image {image_id} point count")
            remaining = file_size - stream.tell()
            if point_count > remaining // 24:
                raise OutputValidationError(
                    f"COLMAP image {image_id} declares {point_count} observations, "
                    "which exceed the remaining file bounds"
                )
            point3d_ids: list[int] = []
            for point_index in range(point_count):
                x, y, encoded_point3d_id = struct.unpack(
                    "<ddQ",
                    _read_exact(
                        stream,
                        24,
                        context=f"image {image_id} observation {point_index}",
                    ),
                )
                if not math.isfinite(x) or not math.isfinite(y):
                    raise OutputValidationError(
                        f"COLMAP image {image_id} observation {point_index} "
                        "contains a non-finite coordinate"
                    )
                # The unsigned all-ones value is COLMAP's explicit unlinked
                # observation sentinel. Point ID zero itself is valid.
                point3d_id = (
                    -1
                    if encoded_point3d_id == _INVALID_POINT3D_ID
                    else encoded_point3d_id
                )
                point3d_ids.append(point3d_id)
            images.append(
                RegisteredImage(
                    image_id=image_id,
                    quaternion_wxyz=quaternion,  # type: ignore[arg-type]
                    translation_xyz=translation,  # type: ignore[arg-type]
                    camera_id=camera_id,
                    name=name,
                    points3d_ids=tuple(point3d_ids),
                )
            )
        _require_end_of_file(stream, context="images.bin")
    return tuple(images)


def validate_model_directory(model_dir: Path) -> Path:
    if not model_dir.is_dir():
        raise OutputValidationError(f"COLMAP model directory is missing: {model_dir}")
    missing = [name for name in REQUIRED_COLMAP_FILES if not (model_dir / name).is_file()]
    if missing:
        raise OutputValidationError(
            f"COLMAP model at {model_dir} is missing: {', '.join(missing)}"
        )
    empty = [name for name in REQUIRED_COLMAP_FILES if (model_dir / name).stat().st_size == 0]
    if empty:
        raise OutputValidationError(
            f"COLMAP model at {model_dir} contains empty files: {', '.join(empty)}"
        )
    return model_dir


def discover_model(scene_dir: Path) -> Path:
    """Find exactly one official binary sparse model below ``scene/sparse``."""

    sparse = scene_dir / "sparse"
    if not sparse.is_dir():
        raise OutputValidationError(f"Official VGGSfM did not create {sparse}")
    candidates: list[Path] = []
    if all((sparse / name).is_file() for name in REQUIRED_COLMAP_FILES):
        candidates.append(sparse)
    for camera_file in sorted(sparse.rglob("cameras.bin")):
        parent = camera_file.parent
        if parent != sparse and all((parent / name).is_file() for name in REQUIRED_COLMAP_FILES):
            candidates.append(parent)
    unique = sorted(set(candidates))
    if not unique:
        raise OutputValidationError(
            f"No complete binary COLMAP model found below {sparse}"
        )
    if len(unique) != 1:
        rendered = ", ".join(str(item) for item in unique)
        raise OutputValidationError(
            f"Ambiguous official output: found {len(unique)} COLMAP models: {rendered}"
        )
    return validate_model_directory(unique[0])


def camera_count(cameras_bin: Path) -> int:
    return len(_read_cameras(cameras_bin))


def registered_image_count(images_bin: Path) -> int:
    """Fully validate ``images.bin`` and return its registered image count."""

    return len(_read_registered_images(images_bin))


def iter_points3d(points_bin: Path) -> tuple[int, Iterator[Point3D]]:
    """Return point count and a streaming iterator whose file closes on exhaustion."""

    stream = points_bin.open("rb")
    try:
        count = _read_u64(stream, context="3D point count")
        _validate_count_fits_file(
            count=count,
            file_size=points_bin.stat().st_size,
            minimum_record_size=51,
            context="points3D.bin",
        )
    except Exception:
        stream.close()
        raise

    def iterator() -> Iterator[Point3D]:
        seen_ids: set[int] = set()
        try:
            for index in range(count):
                record = _read_exact(stream, 43, context=f"3D point {index}")
                point_id, x, y, z, red, green, blue, error = struct.unpack(
                    "<QdddBBBd", record
                )
                if point_id == _INVALID_POINT3D_ID:
                    raise OutputValidationError(
                        f"COLMAP 3D point {index} uses the reserved invalid ID"
                    )
                if point_id in seen_ids:
                    raise OutputValidationError(
                        f"COLMAP points3D.bin contains duplicate point ID {point_id}"
                    )
                seen_ids.add(point_id)
                if not all(math.isfinite(value) for value in (x, y, z, error)):
                    raise OutputValidationError(
                        f"COLMAP point {point_id} contains a non-finite value"
                    )
                if any(abs(value) > _FLOAT32_MAX for value in (x, y, z)):
                    raise OutputValidationError(
                        f"COLMAP point {point_id} cannot be represented in float32 PLY"
                    )
                track_length = _read_u64(stream, context=f"3D point {point_id} track length")
                remaining = points_bin.stat().st_size - stream.tell()
                if track_length > remaining // 8:
                    raise OutputValidationError(
                        f"COLMAP point {point_id} track exceeds the remaining file bounds"
                    )
                track: list[tuple[int, int]] = []
                seen_track_entries: set[tuple[int, int]] = set()
                for track_index in range(track_length):
                    image_id, point2d_index = struct.unpack(
                        "<II",
                        _read_exact(
                            stream,
                            8,
                            context=f"3D point {point_id} track entry {track_index}",
                        ),
                    )
                    if image_id == _INVALID_IMAGE_ID:
                        raise OutputValidationError(
                            f"COLMAP point {point_id} track uses the reserved invalid image ID"
                        )
                    entry = (image_id, point2d_index)
                    if entry in seen_track_entries:
                        raise OutputValidationError(
                            f"COLMAP point {point_id} contains duplicate track entry {entry}"
                        )
                    seen_track_entries.add(entry)
                    track.append(entry)
                yield Point3D(
                    point_id=point_id,
                    x=x,
                    y=y,
                    z=z,
                    red=red,
                    green=green,
                    blue=blue,
                    reprojection_error=error,
                    track_length=track_length,
                    track=tuple(track),
                )
            _require_end_of_file(stream, context="points3D.bin")
        finally:
            stream.close()

    return count, iterator()


def validate_binary_model(model_dir: Path) -> dict[str, object]:
    """Fully validate a binary COLMAP model and return normalized metadata.

    Validation covers binary bounds and trailing bytes, finite and valid camera
    intrinsics, normalized finite poses, unique image IDs/names, camera
    references, finite point geometry, and reciprocal image/track references.
    Exact ``image_names`` and ``image_name_to_id`` values are returned for
    lossless comparison against an input image manifest.
    """

    model_dir = validate_model_directory(model_dir)
    cameras = _read_cameras(model_dir / "cameras.bin")
    images = _read_registered_images(model_dir / "images.bin")

    camera_ids = {camera.camera_id for camera in cameras}
    image_by_id = {image.image_id: image for image in images}
    for image in images:
        if image.camera_id not in camera_ids:
            raise OutputValidationError(
                f"COLMAP image {image.image_id} ({image.name!r}) references missing "
                f"camera ID {image.camera_id}"
            )

    point_count, points = iter_points3d(model_dir / "points3D.bin")
    if point_count == 0:
        raise OutputValidationError("COLMAP points3D.bin contains zero points")
    point_ids: set[int] = set()
    track_references: set[tuple[int, int, int]] = set()
    track_observation_count = 0
    for point in points:
        point_ids.add(point.point_id)
        for image_id, point2d_index in point.track:
            image = image_by_id.get(image_id)
            if image is None:
                raise OutputValidationError(
                    f"COLMAP point {point.point_id} track references missing image "
                    f"ID {image_id}"
                )
            if point2d_index >= len(image.points3d_ids):
                raise OutputValidationError(
                    f"COLMAP point {point.point_id} track index {point2d_index} is "
                    f"outside image {image_id}'s {len(image.points3d_ids)} observations"
                )
            observed_id = image.points3d_ids[point2d_index]
            if observed_id != point.point_id:
                raise OutputValidationError(
                    f"COLMAP point {point.point_id} track disagrees with image "
                    f"{image_id} observation {point2d_index}, which stores {observed_id}"
                )
            track_references.add((image_id, point2d_index, point.point_id))
            track_observation_count += 1

    linked_observations: set[tuple[int, int, int]] = set()
    total_image_observations = 0
    for image in images:
        total_image_observations += len(image.points3d_ids)
        for point2d_index, point3d_id in enumerate(image.points3d_ids):
            if point3d_id == -1:
                continue
            if point3d_id not in point_ids:
                raise OutputValidationError(
                    f"COLMAP image {image.image_id} observation {point2d_index} "
                    f"references missing point3D ID {point3d_id}"
                )
            reference = (image.image_id, point2d_index, point3d_id)
            linked_observations.add(reference)
            if reference not in track_references:
                raise OutputValidationError(
                    f"COLMAP image {image.image_id} observation {point2d_index} "
                    f"references point {point3d_id}, but its track omits the observation"
                )
    if track_references != linked_observations:
        raise OutputValidationError(
            "COLMAP image observations and point tracks are not reciprocal"
        )

    return {
        "camera_count": len(cameras),
        "registered_image_count": len(images),
        "point_count": point_count,
        "image_observation_count": total_image_observations,
        "linked_observation_count": track_observation_count,
        "image_names": [image.name for image in images],
        "image_name_to_id": {image.name: image.image_id for image in images},
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "model_id": camera.model_id,
                "model_name": camera.model_name,
                "width": camera.width,
                "height": camera.height,
                "parameters": list(camera.parameters),
            }
            for camera in cameras
        ],
        "images": [
            {
                "image_id": image.image_id,
                "name": image.name,
                "camera_id": image.camera_id,
                "quaternion_wxyz": list(image.quaternion_wxyz),
                "translation_xyz": list(image.translation_xyz),
                "point2d_count": len(image.points3d_ids),
            }
            for image in images
        ],
    }


def convert_points3d_to_ply(points_bin: Path, ply_path: Path) -> dict[str, object]:
    """Write a compact binary little-endian PLY with per-point RGB colors."""

    count, points = iter_points3d(points_bin)
    if count == 0:
        raise OutputValidationError("COLMAP points3D.bin contains zero points")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment RGB values preserved from VGGSfM COLMAP points\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    error_sum = 0.0
    available_error_sum = 0.0
    available_error_count = 0
    observation_sum = 0
    tracked_error_sum = 0.0
    tracked_error_count = 0
    tracked_observation_sum = 0
    tracked_point_count = 0
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_path.open("wb") as output:
        output.write(header)
        written = 0
        for point in points:
            output.write(
                struct.pack(
                    "<fffBBB",
                    point.x,
                    point.y,
                    point.z,
                    point.red,
                    point.green,
                    point.blue,
                )
            )
            for axis, value in enumerate((point.x, point.y, point.z)):
                minima[axis] = min(minima[axis], value)
                maxima[axis] = max(maxima[axis], value)
            error_sum += point.reprojection_error
            if point.reprojection_error >= 0.0:
                available_error_sum += point.reprojection_error
                available_error_count += 1
            observation_sum += point.track_length
            if point.track_length > 0:
                if point.reprojection_error >= 0.0:
                    tracked_error_sum += point.reprojection_error
                    tracked_error_count += 1
                tracked_observation_sum += point.track_length
                tracked_point_count += 1
            written += 1
    if written != count:
        raise OutputValidationError(
            f"PLY conversion wrote {written} points but COLMAP declared {count}"
        )
    raw_mean_error = error_sum / count
    raw_mean_track_length = observation_sum / count
    tracked_mean_error = (
        tracked_error_sum / tracked_error_count if tracked_error_count else None
    )
    tracked_mean_track_length = (
        tracked_observation_sum / tracked_point_count if tracked_point_count else None
    )
    return {
        "point_count": count,
        # Quality metrics use only bundle-adjusted/tracked points whose error is
        # actually available. PyCOLMAP/VGGSfM can persist -1 as an uncomputed
        # sentinel; accepting it is necessary for compatibility, but averaging
        # it would invent a physically meaningless negative reprojection error.
        "mean_reprojection_error_px": tracked_mean_error,
        "mean_track_length": tracked_mean_track_length,
        "tracked_point_count": tracked_point_count,
        "trackless_point_count": count - tracked_point_count,
        "tracked_mean_reprojection_error_px": tracked_mean_error,
        "tracked_reprojection_error_sample_count": tracked_error_count,
        "tracked_reprojection_error_unavailable_count": (
            tracked_point_count - tracked_error_count
        ),
        "tracked_mean_track_length": tracked_mean_track_length,
        "available_mean_reprojection_error_px_all_points": (
            available_error_sum / available_error_count
            if available_error_count
            else None
        ),
        "available_reprojection_error_sample_count_all_points": available_error_count,
        "unavailable_reprojection_error_count_all_points": count - available_error_count,
        "legacy_raw_mean_reprojection_error_px_including_negative_sentinels": raw_mean_error,
        "negative_reprojection_errors_treated_as_unavailable": True,
        "raw_mean_track_length_all_points": raw_mean_track_length,
        "trackless_points_bundle_adjusted": False,
        "bounds_min": minima,
        "bounds_max": maxima,
        "ply_format": "binary_little_endian",
        "has_rgb": True,
    }
