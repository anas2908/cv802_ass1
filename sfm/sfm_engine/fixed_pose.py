"""Read-only reuse of historical affine/DSP features and fixed-pose triangulation.

This intentionally has no feature extractor, matcher, Open3D import, or legacy
resume/lock integration. Only a newly created private database is writable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .pipeline import (
    EXPERIMENT_NAME, IMAGE_EXTENSIONS, SFM_DATA_ROOT, _validate_pycolmap,
    atomic_json, image_manifest, reconstruction_metrics, require_under,
    sha256_file, validate_colored_ply,
)

PAIR_BASE = 2147483647


@dataclass(frozen=True)
class FixedPoseConfig:
    experiment_name: str
    image_dir: Path
    baseline_model: Path
    database: Path
    feature_database: Path
    feature_metadata: Path
    num_threads: int = 4

    def validate(self):
        if not EXPERIMENT_NAME.fullmatch(self.experiment_name):
            raise ValueError("invalid experiment name")
        if re.match(r"^E(?:[1-9]|10)(?:$|[_.-])", self.experiment_name, re.I):
            raise ValueError("E1-E10 are historical names; choose a new experiment ID")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")
        for name in ("image_dir", "baseline_model", "database", "feature_database", "feature_metadata"):
            path = getattr(self, name)
            require_under(path, SFM_DATA_ROOT, name)
            if not path.exists():
                raise FileNotFoundError(path)
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ValueError(f"symlink input is not an immutable physical source: {path}")
        for name in ("database", "feature_database"):
            if ".working" in getattr(self, name).name:
                raise ValueError("never reuse a historical .working database")
        return self


def check_quiet_database(path: Path) -> None:
    """Reject uncheckpointed state and any visible same-user writable handle.

    Zero-byte transferred WAL and stale SHM files are neither opened nor removed.
    Fingerprints before/after reading detect changes from another host too.
    This is not a distributed lock: only published, inactive inputs are supported.
    """
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise RuntimeError(f"database has pending or active writes: {sidecar}")
    if not Path("/proc").is_dir():
        return  # macOS has no /proc; stable-file hashes and sidecar checks still apply.
    identity = (path.stat().st_dev, path.stat().st_ino)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            for descriptor in (process / "fd").iterdir():
                try:
                    stat = descriptor.stat()
                    if (stat.st_dev, stat.st_ino) != identity:
                        continue
                    fields = (process / "fdinfo" / descriptor.name).read_text().splitlines()
                    flags = next(int(line.split()[1], 8) for line in fields if line.startswith("flags:"))
                    if flags & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                        raise RuntimeError(f"source database has an active writable handle: PID {process.name}")
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


@contextmanager
def readonly_database(path: Path):
    check_quiet_database(path)
    # immutable avoids SQLite touching source SHM/WAL, even for read-only access.
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


def fingerprint(path: Path) -> dict:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"input changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


def frozen_inputs(config: FixedPoseConfig) -> dict:
    images = sorted(p for p in config.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if len(images) < 2:
        raise ValueError("at least two input images are required")
    for path in images:
        require_under(path, config.image_dir, "image")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"image must not be a symlink: {path}")
    suffix = next((s for s in (".bin", ".txt") if all((config.baseline_model / (k + s)).is_file() for k in ("cameras", "images", "points3D"))), None)
    if suffix is None:
        raise ValueError("baseline must be a complete COLMAP model directory")
    model_files = [config.baseline_model / (name + suffix) for name in ("cameras", "images", "points3D", "rigs", "frames") if (config.baseline_model / (name + suffix)).exists()]
    for path in model_files:
        if path.is_symlink():
            raise ValueError(f"model file must not be a symlink: {path}")
    return {
        "images": image_manifest(config.image_dir, images),
        "model": [fingerprint(path) for path in model_files],
        "database": fingerprint(config.database),
        "feature_database": fingerprint(config.feature_database),
        "feature_metadata": fingerprint(config.feature_metadata),
    }


def table_digest(connection, table: str, key: str) -> str:
    """Stable content hash independent of SQLite page packing or journal mode."""
    digest = hashlib.sha256()
    for row in connection.execute(f"SELECT * FROM {table} ORDER BY {key}"):
        for value in row:
            encoded = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def validate_database(connection, reconstruction, image_dir: Path) -> dict:
    """Require matching image IDs, scaled calibration, features, and pair indices."""
    import numpy as np
    import pycolmap

    if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise ValueError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("SQLite foreign-key check failed")
    registered = {int(i): reconstruction.images[i] for i in reconstruction.reg_image_ids()}
    rows = connection.execute("SELECT image_id,name,camera_id FROM images").fetchall()
    if set(registered) != {row[0] for row in rows}:
        raise ValueError("database image IDs differ from baseline registered images")
    feature_rows = {}
    sizes = {}
    for image_id, name, camera_id in rows:
        image = registered[image_id]
        if name != image.name or camera_id != image.camera_id:
            raise ValueError(f"database/baseline image or camera reference differs: {name}")
        path = require_under(image_dir / name, image_dir, "database image reference")
        photo = pycolmap.Bitmap.read(path, as_rgb=False)
        if photo is None:
            raise ValueError(f"COLMAP cannot read input image: {path}")
        size = (photo.width, photo.height)
        del photo
        if camera_id in sizes and sizes[camera_id] != size:
            raise ValueError(f"mixed image dimensions in camera group {camera_id}")
        sizes[camera_id] = size
        keypoints = connection.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
        descriptors = connection.execute("SELECT rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
        if not keypoints or not descriptors or keypoints[0] < 1 or keypoints[0] != descriptors[0]:
            raise ValueError(f"missing or incompatible features: {name}")
        count, cols, blob = keypoints
        if cols != 6 or blob is None or len(blob) != count * cols * 4:
            raise ValueError(f"expected retained affine six-column keypoints: {name}")
        points = np.frombuffer(blob, dtype="<f4").reshape(count, cols)
        if not np.isfinite(points).all() or (points[:, :2] < 0).any() or (points[:, 0] > size[0]).any() or (points[:, 1] > size[1]).any():
            raise ValueError(f"invalid keypoint coordinates: {name}")
        if descriptors[1] != 128 or descriptors[2] is None or len(descriptors[2]) != count * 128:
            raise ValueError(f"expected retained 128-byte SIFT descriptors: {name}")
        feature_rows[image_id] = count
    cameras = connection.execute("SELECT camera_id,model,width,height,params FROM cameras").fetchall()
    if {row[0] for row in cameras} != set(reconstruction.cameras):
        raise ValueError("database/baseline camera IDs differ")
    # Same rescaling used by the historical quality path; world poses never change.
    for camera_id, model, width, height, blob in cameras:
        camera = reconstruction.cameras[camera_id]
        camera.rescale(*sizes[camera_id])
        params = np.frombuffer(blob, dtype="<f8")
        if int(camera.model.value) != model or (width, height) != sizes[camera_id] or len(params) != len(camera.params) or not np.allclose(params, camera.params, rtol=1e-12, atol=1e-12):
            raise ValueError(f"database intrinsics disagree with scaled E1 camera {camera_id}")
    pair_counts = {}
    for table in ("matches", "two_view_geometries"):
        nonempty = 0
        for pair_id, count, cols, blob in connection.execute(f"SELECT pair_id,rows,cols,data FROM {table}"):
            first, second = divmod(pair_id, PAIR_BASE)
            if first >= second or first not in registered or second not in registered:
                raise ValueError(f"invalid image reference in {table}: {pair_id}")
            if count == 0:
                continue
            if cols != 2 or blob is None or len(blob) != count * 8:
                raise ValueError(f"invalid match array in {table}: {pair_id}")
            matches = np.frombuffer(blob, dtype="<u4").reshape(count, 2)
            if (matches[:, 0] >= feature_rows[first]).any() or (matches[:, 1] >= feature_rows[second]).any():
                raise ValueError(f"match index exceeds feature rows: {pair_id}")
            nonempty += 1
        pair_counts[table] = nonempty
    if not pair_counts["two_view_geometries"]:
        raise ValueError("database has no verified nonempty pairs")
    return {"images": len(rows), "features": sum(feature_rows.values()), "nonempty_pairs": pair_counts,
            "table_sha256": {table: table_digest(connection, table, key) for table, key in (("keypoints", "image_id"), ("descriptors", "image_id"), ("matches", "pair_id"), ("two_view_geometries", "pair_id"))}}


def calibration(reconstruction) -> dict:
    return {"cameras": {str(k): {"model": str(v.model), "width": v.width, "height": v.height, "params": list(v.params)} for k, v in reconstruction.cameras.items()},
            "poses": {str(k): {"name": reconstruction.images[k].name, "camera_id": reconstruction.images[k].camera_id, "matrix": reconstruction.images[k].cam_from_world().matrix().tolist()} for k in reconstruction.reg_image_ids()}}


def verify_fixed_calibration(expected: dict, actual: dict) -> dict:
    import numpy as np
    if set(expected["cameras"]) != set(actual["cameras"]) or set(expected["poses"]) != set(actual["poses"]):
        raise ValueError("triangulation changed registered image/camera sets")
    largest = 0.0
    for kind, vector in (("cameras", "params"), ("poses", "matrix")):
        for key, before in expected[kind].items():
            after = actual[kind][key]
            if {k: v for k, v in before.items() if k != vector} != {k: v for k, v in after.items() if k != vector}:
                raise ValueError("triangulation changed camera/image identity")
            delta = float(np.max(np.abs(np.asarray(before[vector]) - np.asarray(after[vector]))))
            if not np.isfinite(delta) or delta > 1e-12:
                raise ValueError(f"triangulation changed fixed {kind}: {delta}")
            largest = max(largest, delta)
    return {"fixed_extrinsics": True, "fixed_intrinsics": True, "max_absolute_difference": largest, "tolerance": 1e-12}


def triangulation_options(threads: int) -> dict:
    return {"num_threads": threads, "extract_colors": True, "random_seed": 0,
            "fix_existing_frames": True, "ba_refine_focal_length": False,
            "ba_refine_principal_point": False, "ba_refine_extra_params": False,
            "ba_refine_sensor_from_rig": False, "ba_use_gpu": False,
            "mapper": {"fix_existing_frames": True, "filter_max_reproj_error": 3.5},
            "triangulation": {"ignore_two_view_tracks": True, "min_angle": 1.5}}


def run(config: FixedPoseConfig, *, dry_run: bool = False) -> dict:
    import pycolmap
    config.validate()
    _validate_pycolmap(pycolmap)
    output = require_under(SFM_DATA_ROOT / "experiments" / config.experiment_name, SFM_DATA_ROOT / "experiments", "new experiment")
    existing = output.exists()
    if existing and not (output / "receipt.json").is_file():
        raise FileExistsError(f"incomplete experiment already exists; choose a new ID: {output}")
    before = frozen_inputs(config)
    reconstruction = pycolmap.Reconstruction(config.baseline_model)
    if {reconstruction.images[i].name for i in reconstruction.reg_image_ids()} != {row["path"] for row in before["images"]}:
        raise ValueError("baseline registered names differ from input images")
    metadata = json.loads(config.feature_metadata.read_text())
    if metadata.get("database_size") != before["feature_database"]["bytes"]:
        raise ValueError("feature checkpoint size differs from its provenance")
    signature = metadata["signature"]
    expected_images = {row[0]: row[1] for row in signature["images"]}
    if expected_images != {row["path"]: row["bytes"] for row in before["images"]}:
        raise ValueError("quality feature provenance/image names or sizes disagree")
    sift = signature["extraction"]["sift"]
    if not sift.get("estimate_affine_shape") or not sift.get("domain_size_pooling"):
        raise ValueError("feature provenance must explicitly retain affine/DSP SIFT")
    with readonly_database(config.database) as connection:
        evidence = validate_database(connection, reconstruction, config.image_dir)
    with readonly_database(config.feature_database) as features:
        for table in ("keypoints", "descriptors"):
            if table_digest(features, table, "image_id") != evidence["table_sha256"][table]:
                raise ValueError(f"matched database does not preserve feature checkpoint {table}")
    frozen_calibration = calibration(reconstruction)
    identity = {"recipe": "headless-fixed-pose-reuse-v1", "config": asdict(config), "inputs": before}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    plan = {"identity_sha256": identity_hash, "config": asdict(config), "source_validation": evidence,
            "device": "cpu", "feature_extraction": "reused affine/DSP checkpoint unchanged", "matching": "reused existing verified pairs unchanged",
            "triangulation_options": triangulation_options(config.num_threads), "output": str(output)}
    if existing:
        previous = json.loads((output / "receipt.json").read_text())
        if previous.get("complete") is not True or previous.get("identity_sha256") != identity_hash:
            raise ValueError("existing experiment identity differs; choose a new ID")
        model = output / "outputs" / "colmap" / "sparse" / "0"
        records = [fingerprint(require_under(p, output, "cached model file")) for p in sorted(model.iterdir()) if p.is_file()]
        if records != previous.get("model_files"):
            raise ValueError("cached model files differ from success receipt")
        cached = pycolmap.Reconstruction(model)
        verify_fixed_calibration(frozen_calibration, calibration(cached))
        metrics = reconstruction_metrics(cached)
        cloud = validate_colored_ply(output / "outputs" / "sparse_colored.ply", metrics["points3D"])
        if cloud != previous.get("ply"):
            raise ValueError("cached PLY differs from success receipt")
        if frozen_inputs(config) != before:
            raise RuntimeError("source inputs changed during cached validation")
        # No write, triangulation, or new attempt occurs on this path.
        return {**previous, "cached_reuse": True, "dry_run": dry_run}
    if dry_run:
        if frozen_inputs(config) != before:
            raise RuntimeError("source inputs changed during dry-run validation")
        return {"dry_run": True, **plan}
    if str(SFM_DATA_ROOT).startswith("/l/") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cluster reconstruction requires an active Slurm compute job")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", {**identity, **plan})
    started = time.monotonic()
    try:
        private = output / "database.db"
        # Copy bytes only after rejecting pending WAL; immutable SQLite was never writable.
        shutil.copyfile(config.database, private)
        if sha256_file(private) != before["database"]["sha256"]:
            raise RuntimeError("private database differs from frozen source")
        model = output / "outputs" / "colmap" / "sparse" / "0"
        model.mkdir(parents=True)
        scaled = output / "baseline_scaled"
        scaled.mkdir()
        reconstruction.write(scaled)
        result = pycolmap.triangulate_points(reconstruction=reconstruction, database_path=private,
            image_path=config.image_dir, output_path=model, clear_points=True,
            refine_intrinsics=False, options=triangulation_options(config.num_threads))
        invariance = verify_fixed_calibration(frozen_calibration, calibration(result))
        metrics = reconstruction_metrics(result)
        result.write(model)
        invariance = verify_fixed_calibration(frozen_calibration, calibration(pycolmap.Reconstruction(model)))
        ply = output / "outputs" / "sparse_colored.ply"
        result.export_PLY(ply)
        ply_receipt = validate_colored_ply(ply, metrics["points3D"])
        # Private DB may have upgraded bookkeeping; scientific arrays must be identical.
        with readonly_database(private) as connection:
            after_tables = {table: table_digest(connection, table, key) for table, key in (("keypoints", "image_id"), ("descriptors", "image_id"), ("matches", "pair_id"), ("two_view_geometries", "pair_id"))}
        if after_tables != evidence["table_sha256"]:
            raise RuntimeError("triangulation changed reused features or matching arrays")
        if frozen_inputs(config) != before:
            raise RuntimeError("historical source changed during reconstruction")
        check_quiet_database(config.database)
        receipt = {"complete": True, **plan, "metrics": metrics, "calibration_invariance": invariance,
                   "source_unchanged": True, "features_and_matches_preserved": True,
                   "ply": ply_receipt, "model_files": [fingerprint(p) for p in sorted(model.iterdir()) if p.is_file()],
                   "pycolmap": pycolmap.__version__, "elapsed_seconds": time.monotonic() - started,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        atomic_json(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        atomic_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error), "identity_sha256": identity_hash})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate/hash sources without creating an experiment")
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text())
    for name in ("image_dir", "baseline_model", "database", "feature_database", "feature_metadata"):
        path = Path(values[name])
        values[name] = path if path.is_absolute() else SFM_DATA_ROOT / path
    print(json.dumps(run(FixedPoseConfig(**values), dry_run=args.dry_run), indent=2, default=str))
    return 0
