#!/usr/bin/env python3
"""Review saved pilot inference without changing its failed publication receipt.

This command never invokes inference. It reads a finished failed attempt,
validates the original COLMAP binaries twice, and exports a separately named
review. The original request, receipt, logs, model and images are frozen by
size/mtime/ctime/inode/SHA-256 before and after review. Existing reviews are
never overwritten. A review is not a successful full-dataset experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine import colmap
from vggsfm_engine.io_utils import (
    atomic_write_json, file_record, read_json, validate_identifier,
)
from vggsfm_engine.paths import production_layout, require_within


def successful_inference(log: str) -> dict:
    """Read the final executor marker, never infer success from model files alone."""
    matches = list(re.finditer(
        r"^===== official VGGSfM inference exit=(-?\d+) "
        r"elapsed=(\d+(?:\.\d+)?)s @ (.+?) =====$", log, re.MULTILINE,
    ))
    if not matches or matches[-1].group(1) != "0":
        raise ValueError("The final official inference completion marker is not exit=0")
    marker = matches[-1]
    elapsed = float(marker.group(2))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Inference elapsed time must be finite and positive")
    return {
        "status": "succeeded", "exit_code": 0,
        "elapsed_seconds": elapsed, "elapsed_as_logged": marker.group(2),
        "completed_at": marker.group(3), "exact_completion_marker": marker.group(0),
        "timing_scope": "Official inference subprocess and monitor shutdown; excludes wrapper publication and this review.",
    }


def summarize_samples(path: Path) -> dict:
    """Summarize saved inference-PID samples; do not poll the current GPU."""
    pid_match = re.fullmatch(r"resources-(\d+)\.jsonl", path.name)
    if not pid_match:
        raise ValueError("Resource filename must identify its original inference PID")
    root_pid = int(pid_match.group(1))
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError("Empty resource evidence")
    previous = -math.inf
    for sample in samples:
        timestamp = sample["time_unix"]
        if not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError("Resource timestamps must be finite and strictly increasing")
        previous = timestamp
        if root_pid not in sample["pids"]:
            raise ValueError("Resource sample is not attributed to the requested process tree")
        for key in ("rss_kib", "gpu_mib"):
            value = sample[key]
            if value is None and key == "gpu_mib":
                continue
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid sampled {key}")
    gpu_values = [sample["gpu_mib"] for sample in samples if sample["gpu_mib"] is not None]
    return {
        "source": file_record(path), "root_pid": root_pid, "samples": len(samples),
        "first_sample_unix": samples[0]["time_unix"],
        "last_sample_unix": samples[-1]["time_unix"],
        "observed_span_seconds": samples[-1]["time_unix"] - samples[0]["time_unix"],
        "peak_process_tree_rss_kib": max(sample["rss_kib"] for sample in samples),
        "peak_process_tree_gpu_mib": max(gpu_values) if gpu_values else None,
        "gpu_samples_available": len(gpu_values),
        "scope": "Saved samples of the official inference PID and its descendants, not node-wide GPU usage.",
        "limitations": "Approximately two-second samples can miss transient peaks; summed process RSS can double-count shared pages. Not a full-125-image estimate.",
    }


def requested_names(request: dict) -> tuple[dict, dict]:
    """Require a lossless source-to-official-name map, including unregistered views."""
    rows = request["input_manifest"]["images"]
    sources = {row["path"]: row for row in rows}
    mappings = request["official_image_name_map"]
    official = {row["official"]: row["source"] for row in mappings}
    if len(sources) != len(rows) or len(official) != len(mappings):
        raise ValueError("Duplicate source or official image names")
    if len(official) != len(sources) or set(official.values()) != set(sources):
        raise ValueError("Official image mapping does not exactly cover requested inputs")
    for name in official:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Unsafe official image name: {name}")
    return sources, official


def frozen_record(path: Path) -> dict:
    before = path.stat()
    record = file_record(path)
    after = path.stat()
    attributes = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino")
    if any(getattr(before, key) != getattr(after, key) for key in attributes):
        raise ValueError(f"Input changed while hashing: {path}")
    record.update({key: getattr(after, key) for key in attributes})
    return record


def review(run_id: str, attempt: str, review_id: str, resource_name: str,
           expected_pycolmap: str = "3.10.0") -> dict:
    layout = production_layout()
    for value, label in ((run_id, "run ID"), (attempt, "attempt"), (review_id, "review ID")):
        validate_identifier(value, label=label)
    if not re.fullmatch(r"resources-\d+\.jsonl", resource_name):
        raise ValueError("Resource filename must be resources-PID.jsonl")
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    root = layout.assert_member(layout.experiment_root(run_id), label="experiment", must_exist=True)
    attempt_root = require_within(root / "attempts" / attempt, root, label="attempt", must_exist=True)
    destination = require_within(root / review_id, root, label="new review")
    if destination.exists():
        raise ValueError(f"Review already exists; refusing overwrite: {destination}")
    request_path, receipt_path = root / "request.json", attempt_root / "receipt.json"
    log_path, resources_path = attempt_root / "official.log", attempt_root / resource_name
    model = colmap.discover_model(attempt_root / "scene")
    for path in (request_path, receipt_path, log_path, resources_path, model):
        require_within(path, root, label="pilot evidence", must_exist=True)
    request, receipt = read_json(request_path), read_json(receipt_path)
    if receipt.get("status") != "failed" or receipt.get("error_type") != "OutputValidationError":
        raise ValueError("Review requires a finished attempt with failed output validation")
    if receipt.get("request_fingerprint") != request.get("request_fingerprint"):
        raise ValueError("Attempt does not match the frozen request")
    if receipt.get("run_id") != run_id or int(receipt["attempt"]) != int(attempt):
        raise ValueError("Receipt run/attempt identity differs")
    dataset = validate_identifier(request["dataset"], label="dataset")
    dataset_root = layout.assert_member(layout.dataset_root(dataset), label="pilot inputs", must_exist=True)
    input_receipt_path = dataset_root / "input_receipt.json"
    input_receipt = read_json(input_receipt_path)
    sources, official = requested_names(request)
    if not 0 < len(sources) < int(input_receipt["original_image_count"]):
        raise ValueError("This review command requires an explicitly smaller pilot subset")

    evidence_paths = [request_path, receipt_path, log_path, resources_path, input_receipt_path]
    evidence_paths += [model / name for name in colmap.REQUIRED_COLMAP_FILES]
    image_validation = []
    for name, source in sorted(official.items()):
        copied = require_within(dataset_root / source, dataset_root, label="pilot image", must_exist=True)
        staged = require_within(attempt_root / "scene" / "images" / name, layout.root,
                                label="official staged image", must_exist=True)
        for path in (copied, staged):
            current = file_record(path)
            if (current["bytes"], current["sha256"]) != (sources[source]["bytes"], sources[source]["sha256"]):
                raise ValueError(f"Image differs from requested manifest: {path}")
            evidence_paths.append(path)
        image_validation.append({"official": name, "source": source, **sources[source]})
    # Resolve duplicate physical paths only once when scene images are symlinks.
    frozen = {str(path): frozen_record(path) for path in dict.fromkeys(evidence_paths)}
    inference = successful_inference(log_path.read_text())
    resources = summarize_samples(resources_path)
    completed_unix = datetime.fromisoformat(inference["completed_at"]).timestamp()
    if resources["last_sample_unix"] > completed_unix + 1 or resources["first_sample_unix"] < completed_unix - inference["elapsed_seconds"] - 2:
        raise ValueError("Selected resource samples are outside the successful inference interval")
    validation = colmap.validate_binary_model(model)
    registered = set(validation["image_names"])
    if not registered or not registered <= set(official):
        raise ValueError("Registered model images do not match requested official names")

    # This import does not run neural inference or require an available GPU.
    import pycolmap
    if pycolmap.__version__ != expected_pycolmap:
        raise ValueError(f"Expected PyCOLMAP {expected_pycolmap}; found {pycolmap.__version__}")
    reconstruction = pycolmap.Reconstruction(str(model))
    reloaded = {
        "version": pycolmap.__version__, "camera_count": reconstruction.num_cameras(),
        "registered_image_count": reconstruction.num_reg_images(),
        "point_count": reconstruction.num_points3D(),
        "image_names": sorted(image.name for image in reconstruction.images.values()),
    }
    for key in ("camera_count", "registered_image_count", "point_count"):
        if reloaded[key] != validation[key]:
            raise ValueError(f"PyCOLMAP reload disagrees with binary validator: {key}")
    if reloaded["image_names"] != sorted(registered):
        raise ValueError("PyCOLMAP reloaded different registered names")

    destination.mkdir(exist_ok=False)
    ply_path = destination / "point_cloud.ply"
    point_metrics = colmap.convert_points3d_to_ply(model / "points3D.bin", ply_path)
    with ply_path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 4096:
                raise ValueError("Invalid exported PLY header")
            header += line
    if point_metrics["point_count"] != validation["point_count"] or ply_path.stat().st_size - len(header) != 15 * validation["point_count"]:
        raise ValueError("Exported RGB PLY count or payload size disagrees with model")
    for path, before in frozen.items():
        if frozen_record(Path(path)) != before:
            raise ValueError(f"Original evidence changed during review: {path}")
    report = {
        "schema_version": 1, "status": "review_complete", "run_id": run_id,
        "attempt": attempt, "review_id": review_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing pilot inference reviewed/exported on CPU; not a new inference or full-dataset result.",
        "original_wrapper_status": receipt["status"], "original_wrapper_error": receipt["error"],
        "original_receipt_preserved": True, "inference_rerun": False,
        "official_inference": inference, "profile": request["profile"],
        "independence": request["independence"], "upstream": request["upstream"],
        "request_fingerprint": request["request_fingerprint"],
        "input_count": len(sources), "full_dataset_count": input_receipt["original_image_count"],
        "input_selection": input_receipt["selection"],
        "image_manifest_and_staged_hashes_verified": True, "input_images": image_validation,
        "all_requested_images_registered": registered == set(official),
        "registered_source_images": sorted(official[name] for name in registered),
        "unregistered_source_images": sorted(official[name] for name in set(official) - registered),
        "binary_validation": validation, "pycolmap_reload": reloaded,
        "point_metrics": point_metrics, "point_cloud": file_record(ply_path),
        "sampled_inference_resources": resources,
        "source_model": str(model),
        "model_files": [frozen[str(model / name)] for name in colmap.REQUIRED_COLMAP_FILES],
        "original_failure_receipt": frozen[str(receipt_path)],
        "original_official_log": frozen[str(log_path)],
        "frozen_inputs_before_and_after_equal": True, "frozen_inputs": list(frozen.values()),
        "review_source": file_record(Path(__file__).resolve()),
        "validator_source": file_record(Path(colmap.__file__).resolve()),
        "resource_sampler_source": file_record(Path(colmap.__file__).with_name("resources.py")),
        "review_execution": {"python": sys.executable, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
    }
    atomic_write_json(destination / "review.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", default="0001")
    parser.add_argument("--review-id", default="review-v1")
    parser.add_argument("--resources", required=True, help="Finished inference resources-PID.jsonl filename")
    parser.add_argument("--expected-pycolmap", default="3.10.0")
    args = parser.parse_args()
    result = review(args.run_id, args.attempt, args.review_id, args.resources, args.expected_pycolmap)
    print(json.dumps({key: result[key] for key in (
        "status", "run_id", "input_count", "full_dataset_count", "all_requested_images_registered",
        "point_metrics", "point_cloud", "official_inference", "sampled_inference_resources",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
