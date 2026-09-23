#!/usr/bin/env python3
"""Publish an already inspected dark-cleanup derivative; never run reconstruction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv802_mvs.io_utils import atomic_json, sha256_file, utc_now
from cv802_mvs.paths import PathPolicy, safe_experiment_name
from cv802_mvs.validation import validate_colored_ply
from clean_dark_person_crutches import read_vertices, records


def publish(derived_id):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an existing authorized CPU allocation step")
    import numpy as np
    policy = PathPolicy.production()
    root = policy.require_method(policy.method_root / "outputs" / safe_experiment_name(derived_id), "cleanup", must_exist=True)
    if (root / "publication.json").exists():
        raise FileExistsError("Existing publication is immutable")
    receipt = json.loads((root / "cleanup_receipt.json").read_text())
    review = json.loads((root / "visual_review.json").read_text())
    if receipt["status"] != "validated_pending_visual_review" or not review["accepted_for_separate_saved_view"]:
        raise ValueError("Validated result and explicit visual approval required")
    if receipt["derived_id"] != derived_id or review["derived_id"] != derived_id:
        raise ValueError("Review/run identity mismatch")
    for record in receipt["source_records_after"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("Frozen source changed after cleanup")
    cloud = validate_colored_ply(root / "point_cloud.ply")
    if cloud != receipt["point_cloud"] or sha256_file(root / "point_cloud.ply") != review["point_cloud_sha256"]:
        raise ValueError("Reviewed point cloud identity changed")
    audit = np.load(root / "selection_audit.npz", allow_pickle=False)
    original = policy.method_root / "experiments" / receipt["raw_experiment"] / "outputs/fused.ply"
    _, before = read_vertices(original)
    _, after = read_vertices(root / "point_cloud.ply")
    indices = audit["source_vertex_indices"]
    if np.any(np.diff(indices) <= 0) or before[indices].tobytes() != after.tobytes():
        raise ValueError("Published vertex bytes are not the declared ordered source subset")
    expected_body = (audit["usable"] >= 6) & (audit["foreground"] >= np.ceil(.9 * audit["usable"] - 1e-12))
    if not np.array_equal(expected_body, audit["body"]) or not np.array_equal(np.flatnonzero(expected_body | audit["protected"]), indices):
        raise ValueError("Pointwise support audit does not reproduce the selected rows")
    for row in receipt["overlays"] + receipt["audit_files"]:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("Review/audit artifact changed")
    if set(review["inspected_overlays"]) != {Path(row["path"]).name for row in receipt["overlays"]}:
        raise ValueError("Every generated overlay must be inspected")
    preview_receipt = json.loads(Path(review["preview_receipt"]).read_text())
    if preview_receipt["source"]["sha256"] != review["point_cloud_sha256"]:
        raise ValueError("Preview belongs to another point cloud")
    result = {"schema_version": 1, "status": "complete", "created_at": utc_now(), "derived_id": derived_id,
              "point_count": len(after), "point_cloud": cloud, "visual_review": review["status"],
              "raw_sources_unchanged": True, "subset_bytes_reverified": True,
              "pointwise_body_and_union_selection_reverified": True,
              "artifacts": records([root / name for name in ("point_cloud.ply", "cleanup_receipt.json", "visual_review.json", "selection_audit.npz", "projection_rows.json")]
                                   + [Path(row["path"]) for row in receipt["overlays"]]
                                   + [Path(review["preview"]), Path(review["preview_receipt"])]),
              "source_code": records([Path(__file__)]),
              "quality_label": "body90 silhouette consensus plus crutch geometry candidates; no visibility/accuracy certification",
              "slurm_job_id": os.environ["SLURM_JOB_ID"]}
    atomic_json(root / "publication.json", result)
    atomic_json(root / "state.json", {"status": "complete", "point_count": len(after), "publication": str(root / "publication.json")})
    print(json.dumps({"status": "complete", "point_count": len(after), "publication": str(root / "publication.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-id", default="dark-e3-mvs-body90-crutches-v1")
    publish(parser.parse_args().derived_id)
