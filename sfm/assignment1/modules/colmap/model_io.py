"""Small helpers for reading COLMAP's text sparse-model format.

COLMAP writes sparse reconstructions in binary by default.  ``api.py`` uses
COLMAP's ``model_converter`` command to produce the documented text format and
then loads only the fields needed by the starter GUI.
"""

import os.path as osp

import numpy as np


_SINGLE_FOCAL_LENGTH_MODELS = {
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}


def qvec_to_rotation_matrix(qvec):
    """Convert COLMAP's ``[qw, qx, qy, qz]`` quaternion to a 3x3 matrix."""
    qvec = np.asarray(qvec, dtype=np.float64)
    if qvec.shape != (4,):
        raise ValueError(f"Expected a quaternion with 4 values, got {qvec.shape}")

    norm = np.linalg.norm(qvec)
    if norm == 0:
        raise ValueError("Cannot convert a zero-length quaternion")

    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qw * qz,
                2 * qx * qz + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def camera_to_pinhole_intrinsics(camera):
    """Return the pinhole fields required by the Open3D starter GUI.

    Distortion parameters remain in COLMAP's model; Open3D's camera visualizer
    only needs the focal lengths and principal point.
    """
    model = camera["model"]
    params = np.asarray(camera["params"], dtype=np.float64)

    if model in _SINGLE_FOCAL_LENGTH_MODELS:
        if len(params) < 3:
            raise ValueError(f"Camera model {model} requires at least 3 parameters")
        fx = fy = float(params[0])
        cx, cy = map(float, params[1:3])
    else:
        if len(params) < 4:
            raise ValueError(f"Camera model {model} requires at least 4 parameters")
        fx, fy, cx, cy = map(float, params[:4])

    return {
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _read_cameras(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"Malformed camera at {path}:{line_number}")

            camera_id = int(fields[0])
            cameras[camera_id] = {
                "id": camera_id,
                "model": fields[1],
                "width": int(fields[2]),
                "height": int(fields[3]),
                "params": np.asarray(fields[4:], dtype=np.float64),
            }

    return cameras


def _read_images(path):
    images = {}

    with open(path, "r", encoding="utf-8") as stream:
        lines = iter(enumerate(stream, start=1))
        for line_number, raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            # maxsplit preserves image names containing spaces.
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"Malformed registered image at {path}:{line_number}")

            image_id = int(fields[0])
            images[image_id] = {
                "id": image_id,
                "qvec": np.asarray(fields[1:5], dtype=np.float64),
                "tvec": np.asarray(fields[5:8], dtype=np.float64),
                "camera_id": int(fields[8]),
                "name": fields[9],
            }

            # The next physical line stores POINTS2D triples.  The starter GUI
            # does not need them, but it must still consume the line so that it
            # is not mistaken for another image record.
            try:
                next(lines)
            except StopIteration:
                pass

    return images


def _read_points3d(path):
    points3d = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            if len(fields) < 8:
                raise ValueError(f"Malformed 3D point at {path}:{line_number}")

            point_id = int(fields[0])
            points3d[point_id] = {
                "id": point_id,
                "xyz": np.asarray(fields[1:4], dtype=np.float64),
                "rgb": np.asarray(fields[4:7], dtype=np.uint8),
                "error": float(fields[7]),
            }

    return points3d


def read_text_model(model_dir):
    """Read cameras, registered images, and 3D points from a text model."""
    camera_path = osp.join(model_dir, "cameras.txt")
    image_path = osp.join(model_dir, "images.txt")
    point_path = osp.join(model_dir, "points3D.txt")

    missing = [
        path
        for path in (camera_path, image_path, point_path)
        if not osp.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "COLMAP text model is incomplete; missing: " + ", ".join(missing)
        )

    return (
        _read_cameras(camera_path),
        _read_images(image_path),
        _read_points3d(point_path),
    )
