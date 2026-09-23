"""Minimal, read-only COLMAP model parser used for staging and validation.

The pipeline intentionally does not depend on PyCOLMAP: dense MVS is executed
by the CUDA-enabled COLMAP command-line binary, while this parser only audits
camera/image metadata before expensive work starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import struct
from typing import BinaryIO

from .errors import InputValidationError


# COLMAP camera model ID -> (name, number of parameters).  IDs 0--10 cover the
# public models used by COLMAP 3.7--3.13 and the transferred assignment.
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    model_name: str
    width: int
    height: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]


@dataclass(frozen=True)
class ModelRecords:
    directory: Path
    format: str
    cameras: tuple[CameraRecord, ...]
    images: tuple[ImageRecord, ...]


def _model_format(path: Path) -> str | None:
    for extension, label in ((".bin", "binary"), (".txt", "text")):
        if all((path / f"{stem}{extension}").is_file() for stem in ("cameras", "images", "points3D")):
            return label
    return None


def resolve_model_directory(path: Path) -> tuple[Path, str]:
    """Resolve either a model directory or a parent containing one model."""
    path = path.resolve(strict=True)
    direct = _model_format(path)
    if direct:
        return path, direct
    candidates = [
        (child, model_format)
        for child in sorted(path.iterdir())
        if child.is_dir() and (model_format := _model_format(child)) is not None
    ]
    if len(candidates) != 1:
        descriptions = ", ".join(str(candidate[0]) for candidate in candidates) or "none"
        raise InputValidationError(
            f"Expected exactly one complete COLMAP model under {path}; found {descriptions}"
        )
    return candidates[0]


def _read_exact(stream: BinaryIO, count: int, label: str) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise InputValidationError(f"Truncated COLMAP binary while reading {label}")
    return data


def _skip_exact(stream: BinaryIO, count: int, label: str, file_size: int) -> None:
    position = stream.tell()
    if count < 0 or position + count > file_size:
        raise InputValidationError(f"Truncated COLMAP binary while skipping {label}")
    stream.seek(count, 1)


def _read_c_string(stream: BinaryIO, label: str, maximum: int = 1024 * 1024) -> str:
    data = bytearray()
    while len(data) <= maximum:
        character = stream.read(1)
        if not character:
            raise InputValidationError(f"Truncated COLMAP binary while reading {label}")
        if character == b"\0":
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise InputValidationError(f"Non-UTF-8 COLMAP image name in {label}") from error
        data.extend(character)
    raise InputValidationError(f"Implausibly long COLMAP string in {label}")


def _read_binary_cameras(path: Path) -> tuple[CameraRecord, ...]:
    records: list[CameraRecord] = []
    with path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8, "camera count"))[0]
        if count > 10_000_000:
            raise InputValidationError(f"Implausible camera count in {path}: {count}")
        for index in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<IiQQ", _read_exact(stream, 24, f"camera {index} header")
            )
            if model_id not in CAMERA_MODELS:
                raise InputValidationError(
                    f"Unsupported camera model ID {model_id} in {path}; convert the model to text "
                    "with the installed COLMAP version before staging"
                )
            model_name, parameter_count = CAMERA_MODELS[model_id]
            parameters = struct.unpack(
                f"<{parameter_count}d",
                _read_exact(stream, 8 * parameter_count, f"camera {camera_id} parameters"),
            )
            records.append(CameraRecord(camera_id, model_name, width, height, parameters))
        if stream.read(1):
            raise InputValidationError(f"Unexpected trailing bytes in {path}")
    return tuple(records)


def _read_binary_images(path: Path) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8, "image count"))[0]
        if count > 100_000_000:
            raise InputValidationError(f"Implausible image count in {path}: {count}")
        for index in range(count):
            unpacked = struct.unpack("<I7dI", _read_exact(stream, 64, f"image {index} header"))
            image_id = unpacked[0]
            qvec = tuple(unpacked[1:5])
            tvec = tuple(unpacked[5:8])
            camera_id = unpacked[8]
            name = _read_c_string(stream, f"image {image_id} name")
            point_count = struct.unpack(
                "<Q", _read_exact(stream, 8, f"image {image_id} point count")
            )[0]
            if point_count > 1_000_000_000:
                raise InputValidationError(
                    f"Implausible 2D point count for image {image_id}: {point_count}"
                )
            _skip_exact(
                stream, point_count * 24, f"image {image_id} points", file_size
            )
            records.append(ImageRecord(image_id, camera_id, name, qvec, tvec))
        if stream.read(1):
            raise InputValidationError(f"Unexpected trailing bytes in {path}")
    return tuple(records)


def _read_text_cameras(path: Path) -> tuple[CameraRecord, ...]:
    records: list[CameraRecord] = []
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise InputValidationError(f"Malformed camera record at {path}:{number}")
        try:
            camera_id, width, height = int(fields[0]), int(fields[2]), int(fields[3])
            parameters = tuple(float(value) for value in fields[4:])
        except ValueError as error:
            raise InputValidationError(f"Malformed camera record at {path}:{number}") from error
        records.append(CameraRecord(camera_id, fields[1], width, height, parameters))
    return tuple(records)


def _read_text_images(path: Path) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    expecting_points = False
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if expecting_points:
            # Every image metadata line is followed by exactly one POINTS2D line,
            # which may be empty.  Its contents are not needed for this audit.
            expecting_points = False
            continue
        if not line:
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise InputValidationError(f"Malformed image record at {path}:{number}")
        try:
            image_id = int(fields[0])
            qvec = tuple(float(value) for value in fields[1:5])
            tvec = tuple(float(value) for value in fields[5:8])
            camera_id = int(fields[8])
        except ValueError as error:
            raise InputValidationError(f"Malformed image record at {path}:{number}") from error
        records.append(ImageRecord(image_id, camera_id, fields[9], qvec, tvec))
        expecting_points = True
    return tuple(records)


def _safe_image_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise InputValidationError(f"Unsafe COLMAP image name: {name!r}")
    return path


def load_model_records(path: Path) -> ModelRecords:
    directory, model_format = resolve_model_directory(path)
    if model_format == "binary":
        cameras = _read_binary_cameras(directory / "cameras.bin")
        images = _read_binary_images(directory / "images.bin")
    else:
        cameras = _read_text_cameras(directory / "cameras.txt")
        images = _read_text_images(directory / "images.txt")
    _validate_records(cameras, images)
    return ModelRecords(directory, model_format, cameras, images)


def _validate_records(
    cameras: tuple[CameraRecord, ...], images: tuple[ImageRecord, ...]
) -> None:
    if not cameras:
        raise InputValidationError("COLMAP model contains no cameras")
    if len(images) < 2:
        raise InputValidationError("Dense MVS needs at least two registered images")
    camera_ids = [camera.camera_id for camera in cameras]
    if len(camera_ids) != len(set(camera_ids)):
        raise InputValidationError("COLMAP model contains duplicate camera IDs")
    image_ids = [image.image_id for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise InputValidationError("COLMAP model contains duplicate image IDs")
    names = [image.name for image in images]
    if len(names) != len(set(names)):
        raise InputValidationError("COLMAP model contains duplicate image names")
    known_cameras = set(camera_ids)
    for camera in cameras:
        if camera.width <= 0 or camera.height <= 0:
            raise InputValidationError(f"Camera {camera.camera_id} has invalid dimensions")
        if not camera.parameters or not all(math.isfinite(value) for value in camera.parameters):
            raise InputValidationError(f"Camera {camera.camera_id} has non-finite calibration")
        if camera.parameters[0] <= 0:
            raise InputValidationError(f"Camera {camera.camera_id} has non-positive focal length")
    for image in images:
        _safe_image_name(image.name)
        if image.camera_id not in known_cameras:
            raise InputValidationError(
                f"Image {image.image_id} refers to missing camera {image.camera_id}"
            )
        pose = (*image.qvec, *image.tvec)
        if not all(math.isfinite(value) for value in pose):
            raise InputValidationError(f"Image {image.image_id} has a non-finite pose")
        quaternion_norm = math.sqrt(sum(value * value for value in image.qvec))
        if not 0.99 <= quaternion_norm <= 1.01:
            raise InputValidationError(
                f"Image {image.image_id} quaternion norm is {quaternion_norm:.8g}, not approximately 1"
            )


def referenced_image_paths(records: ModelRecords, image_root: Path) -> dict[str, Path]:
    """Resolve every registered image while blocking traversal and case surprises."""
    root = image_root.resolve(strict=True)
    result: dict[str, Path] = {}
    for image in records.images:
        relative = _safe_image_name(image.name)
        candidate = (root / Path(*relative.parts)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise InputValidationError(f"Image {image.name!r} resolves outside {root}") from error
        if not candidate.is_file():
            raise InputValidationError(f"Registered image is not a file: {candidate}")
        result[image.name] = candidate
    return result


def image_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without decoding pixels or importing Pillow."""
    with path.open("rb") as stream:
        signature = stream.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            if signature[12:16] != b"IHDR":
                raise InputValidationError(f"PNG has no leading IHDR chunk: {path}")
            return struct.unpack(">II", signature[16:24])
        if signature[:2] != b"\xff\xd8":
            raise InputValidationError(f"Unsupported image format for dimension audit: {path}")
        stream.seek(2)
        while True:
            marker_start = stream.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_data = stream.read(2)
            if len(length_data) != 2:
                break
            length = struct.unpack(">H", length_data)[0]
            if length < 2:
                break
            if marker and marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = _read_exact(stream, length - 2, f"JPEG size marker in {path}")
                if len(payload) < 5:
                    break
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            stream.seek(length - 2, 1)
    raise InputValidationError(f"Could not read image dimensions: {path}")
