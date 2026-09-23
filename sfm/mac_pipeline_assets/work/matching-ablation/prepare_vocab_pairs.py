#!/usr/bin/env python3
"""Retrieve E8 image pairs with COLMAP's official trained vocabulary tree.

Read preserved E3 feature checkpoints; work on an independent database copy.
This performs image retrieval only, not pairwise matching or reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np
import pycolmap
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TREE_NAME = "vocab_tree_faiss_flickr100K_words32K.bin"
TREE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/" + TREE_NAME
sys.path.insert(0, str(ROOT / "work/quality-sfm"))
from audit_database import _select_checkpoint
from prepare_quality_datasets import budget_guided, graph_components, summarize


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def finite_budget_guided(candidates, n, angle, shared, groups, budget):
    """Original E3 selector when feasible; same priorities if coverage is impossible.

    The only fallback changes are stopping when no uncovered vertex can be
    helped, and respecting the budget. It never adds an unretrieved image pair.
    """
    try:
        selected, audit = budget_guided(candidates, n, angle, shared, groups, budget)
        audit["exact_e3_selector_succeeded"] = True
        return selected, audit
    except ValueError as error:
        reason = str(error)
    degree = np.zeros(n, dtype=int)
    cross = np.zeros(n, dtype=int)
    bins = [set() for _ in range(n)]
    selected = set()
    def category(i, j):
        return next((k for k, (lo, hi) in enumerate(((3, 10), (10, 20), (20, 40))) if lo <= angle[i, j] < hi), None)
    max_log = max((math.log1p(int(shared[i, j])) for i, j in candidates), default=1.)
    def quality(i, j):
        return math.log1p(int(shared[i, j])) / max_log + (.4 if 3 <= angle[i, j] <= 25 else 0)
    def terms(i, j):
        b = category(i, j)
        new_bins = int(b is not None and b not in bins[i]) + int(b is not None and b not in bins[j])
        new_cross = int(cross[i] == 0) + int(cross[j] == 0) if groups[i] != groups[j] else 0
        return new_cross, new_bins
    def add(pair):
        selected.add(pair)
        i, j = pair
        degree[i] += 1
        degree[j] += 1
        if groups[i] != groups[j]:
            cross[i] += 1
            cross[j] += 1
        b = category(i, j)
        if b is not None:
            bins[i].add(b)
            bins[j].add(b)
    while np.any(degree < 2) and len(selected) < budget:
        choices = []
        for pair in candidates - selected:
            i, j = pair
            gain = int(degree[i] < 2) + int(degree[j] < 2)
            if gain:
                x, b = terms(i, j)
                choices.append(((gain, int(degree[i] == 0) + int(degree[j] == 0), x + b,
                    quality(i, j), -max(degree[i], degree[j]), -i, -j), pair))
        if not choices:
            break
        add(max(choices)[1])
    while len(selected) < min(budget, len(candidates)):
        choices = []
        for pair in candidates - selected:
            i, j = pair
            x, b = terms(i, j)
            value = 4. * x + 2.2 * b + quality(i, j) - .035 * (degree[i] + degree[j])
            choices.append(((value, quality(i, j), -i, -j), pair))
        add(max(choices)[1])
    return selected, {"exact_e3_selector_succeeded": False, "e3_selector_failure": reason,
        "fallback": "Same E3 greedy priorities, stopping infeasible coverage; no new ordinary pairs",
        "budget": budget, "strict_candidates": len(candidates), "selected_pairs": len(selected),
        "minimum_guided_degree": int(degree.min()), "images_with_two_or_more_edges": int((degree >= 2).sum()),
        "images_with_cross_group_edge": int((cross > 0).sum())}


def prepare(subject, args):
    source = ROOT / "reconstructions" / (subject + "_quality")
    out = args.output_root.resolve() / subject
    if out.exists():
        raise FileExistsError(f"Preserving existing retrieval output: {out}")
    config = json.loads((source / "sfm_refine.json").read_text())
    saved = json.loads((source / "colmap/sparse/sfm_inputs.json").read_text())
    metadata_path, metadata, feature_key, ignored = _select_checkpoint(source, config, saved)
    checkpoint = metadata_path.with_suffix(".db")
    tree = args.tree.resolve()
    watched = {p: sha256(p) for p in (checkpoint, metadata_path, tree,
        source / "sfm_refine.json", source / "colmap/sparse/sfm_inputs.json",
        ROOT / "work/quality-sfm/prepare_quality_datasets.py")}
    if checkpoint.stat().st_size != metadata["database_size"]:
        raise ValueError("Checkpoint size does not match its committed signature")
    visual_index = pycolmap.VisualIndex.read(tree)
    tree_properties = {"visual_words": visual_index.num_visual_words(), "descriptor_dimension": visual_index.desc_dim(),
        "embedding_dimension": visual_index.embedding_dim(), "feature_type": str(visual_index.feature_type()),
        "preexisting_image_count": visual_index.num_images()}
    del visual_index
    if tree_properties["descriptor_dimension"] != 128 or tree_properties["preexisting_image_count"]:
        raise ValueError("Expected the trained SIFT vocabulary without indexed task images")
    out.mkdir(parents=True)
    own_database = out / "retrieval_features.db"
    shutil.copy2(checkpoint, own_database)
    # Python sqlite in immutable read-only mode never creates source WAL/SHM files.
    with sqlite3.connect(own_database.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        images = {int(i): name for i, name in db.execute("SELECT image_id,name FROM images ORDER BY image_id")}
        counts = {table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")}
        feature_total = int(db.execute("SELECT SUM(rows) FROM descriptors").fetchone()[0])
    if counts["matches"] or counts["two_view_geometries"]:
        raise ValueError("Feature checkpoint unexpectedly contains pairwise matches")
    query_ids = sorted(images)
    # COLMAP includes the indexed query itself. Ask for K+1, then remove self
    # and truncate explicitly to K nonself neighbors in the returned score order.
    options = pycolmap.VocabTreePairingOptions({"num_images": args.neighbors + 1,
        "num_nearest_neighbors": 5, "num_checks": 64, "num_images_after_verification": 0,
        "max_num_features": -1, "vocab_tree_path": str(tree), "num_threads": args.threads})
    retrieval = {}
    raw_self = raw_pairs = 0
    started = time.monotonic()
    with pycolmap.Database.open(own_database) as db:
        generator = pycolmap.VocabTreePairGenerator(options, db, query_ids)
        indexed_seconds = time.monotonic() - started
        while not generator.has_finished():
            batch = generator.next()
            if not batch:
                raise ValueError("Vocabulary query returned an empty result batch; inspect retrieval log")
            query = int(batch[0][0])
            if query in retrieval or any(int(left) != query for left, right in batch):
                raise ValueError("Unexpected retrieval batch grouping")
            raw_pairs += len(batch)
            raw_self += sum(int(a) == int(b) for a, b in batch)
            nonself = [int(b) for a, b in batch if int(a) != int(b)]
            retrieval[query] = nonself[:args.neighbors]
        del generator
    elapsed = time.monotonic() - started
    if set(retrieval) != set(images):
        raise ValueError("Not every source image completed vocabulary retrieval")
    names = sorted(images.values())
    name_index = {name: i for i, name in enumerate(names)}
    ordinary = {tuple(sorted((name_index[images[a]], name_index[images[b]])))
        for a, neighbors in retrieval.items() for b in neighbors}
    if any(i == j for i, j in ordinary):
        raise AssertionError("Self pair survived retrieval cleanup")
    model_path = ROOT / "reconstructions" / subject / "subject_preview/colmap/sparse/0"
    for p in model_path.iterdir():
        if p.is_file(): watched[p] = sha256(p)
    model = pycolmap.Reconstruction(model_path)
    by_name = {im.name: im for im in model.images.values()}
    if set(by_name) != set(names):
        raise ValueError("Baseline person model and feature checkpoint images differ")
    n = len(names)
    xyz = np.asarray([p.xyz for p in model.points3D.values()])
    person_center = np.median(xyz, axis=0)
    directions = np.asarray([by_name[name].projection_center() for name in names]) - person_center
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    angle = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    np.fill_diagonal(angle, np.inf)
    groups = np.asarray([str(Path(name).parent) for name in names])
    imageid_index = {int(image.image_id): name_index[image.name] for image in model.images.values()}
    rows, cols = [], []
    for row, point in enumerate(model.points3D.values()):
        for col in {imageid_index[int(e.image_id)] for e in point.track.elements}:
            rows.append(row); cols.append(col)
    incidence = coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(xyz), n)).tocsr()
    shared = (incidence.T @ incidence).toarray()
    np.fill_diagonal(shared, 0)
    _, ordinary_neighbors = graph_components(n, ordinary)
    strict, strict_choices = set(), {}
    for i in range(n):
        chosen, reasons = set(), []
        for lo, hi in ((3, 10), (10, 20), (20, 40)):
            candidates = [j for j in ordinary_neighbors[i] if lo <= angle[i, j] < hi and shared[i, j] > 0]
            candidates.sort(key=lambda j: (-int(shared[i, j]), angle[i, j], names[j]))
            if candidates:
                j = candidates[0]; chosen.add(j); strict.add(tuple(sorted((i, j))))
                reasons.append({"image": names[j], "reason": f"best_shared_{lo}_{hi}deg_within_vocab"})
        candidates = [j for j in ordinary_neighbors[i] if j not in chosen and groups[i] != groups[j]
            and 4 <= angle[i, j] <= 45 and shared[i, j] > 0]
        candidates.sort(key=lambda j: (-int(shared[i, j]), angle[i, j], names[j]))
        if candidates:
            j = candidates[0]; strict.add(tuple(sorted((i, j))))
            reasons.append({"image": names[j], "reason": "best_shared_distinct_cross_group_4_45deg_within_vocab"})
        strict_choices[names[i]] = reasons
    budget = {"light_shirt": 238, "black_shirt_crutches": 552}[subject]
    guided, guided_audit = finite_budget_guided(strict, n, angle, shared, groups, budget)
    if not guided <= ordinary:
        raise AssertionError("Guided pass introduced an unretrieved image pair")
    def write_pairs(filename, pairs):
        (out / filename).write_text("".join(f"{names[i]} {names[j]}\n" for i, j in sorted(pairs)))
    write_pairs("matching_pairs.txt", ordinary)
    write_pairs("guided_pairs.txt", guided)
    write_pairs("guided_candidates_strict.txt", strict)
    (out / "retrieved_neighbors.json").write_text(json.dumps({images[i]: [images[j] for j in retrieval[i]]
        for i in sorted(retrieval, key=images.get)}, indent=2) + "\n")
    components, ordinary_neighbors = graph_components(n, ordinary)
    guided_components, guided_neighbors = graph_components(n, guided)
    baseline_pairs = {tuple(sorted(line.split())) for line in (source / "matching_pairs.txt").read_text().splitlines() if line.strip()}
    current_pairs = {(names[i], names[j]) for i, j in ordinary}
    with sqlite3.connect(own_database.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        after_counts = {table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in counts}
    if after_counts != counts:
        raise AssertionError("Pair retrieval changed feature or match table row counts")
    for p, digest in watched.items():
        if sha256(p) != digest:
            raise AssertionError(f"Read-only input changed: {p}")
    report = {"subject": subject, "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Genuine COLMAP VocabTreePairGenerator retrieval only; no feature matching, guided matching or triangulation",
        "pycolmap": pycolmap.__version__, "tree": {"path": str(tree), "url": TREE_URL,
            "official_listing": "https://demuc.de/colmap/", "bytes": tree.stat().st_size,
            "sha256": watched[tree], **tree_properties},
        "feature_checkpoint": str(checkpoint), "feature_checkpoint_sha256": watched[checkpoint],
        "feature_key": feature_key, "ignored_old_checkpoints": ignored,
        "feature_signature": metadata["signature"], "features_indexed": feature_total,
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in options.todict().items()},
        "num_images_requested_includes_self": True, "requested_nonself_neighbors": args.neighbors,
        "actual_nonself_neighbor_counts": summarize([len(v) for v in retrieval.values()]),
        "raw_returned_pairs": raw_pairs, "raw_self_results_removed": raw_self,
        "num_images_after_verification_0_means": "No spatial reranking in image retrieval; normal feature matching geometric verification happens later",
        "indexed_seconds": indexed_seconds, "retrieval_total_seconds": elapsed,
        "images": n, "ordinary_pairs": len(ordinary), "guided_pairs": len(guided),
        "e3_ordinary_pairs": len(baseline_pairs), "ordinary_pair_count_ratio_to_e3": len(ordinary) / len(baseline_pairs),
        "ordinary_pairs_in_common_with_e3": len(current_pairs & baseline_pairs),
        "ordinary_components": len(components), "ordinary_component_sizes": sorted(map(len, components), reverse=True),
        "ordinary_degree": summarize([len(v) for v in ordinary_neighbors]),
        "guided_components": len(guided_components), "guided_degree": summarize([len(v) for v in guided_neighbors]),
        "guided_images_with_fewer_than_two_edges": [names[i] for i, v in enumerate(guided_neighbors) if len(v) < 2],
        "ordinary_angle_degrees": summarize([angle[i, j] for i, j in ordinary]),
        "guided_angle_degrees": summarize([angle[i, j] for i, j in guided]),
        "guided_selector": guided_audit, "guided_strict_selections": strict_choices,
        "ordinary_pairs_policy": "Undirected union of each image's first K nonself vocabulary-retrieved neighbors; no geometry filtering, temporal additions or connectivity bridges",
        "guided_policy": "Same E3 angular bins/cross-group shared-body candidate priorities and budget, restricted to vocabulary ordinary pairs",
        "guided_is_subset_of_genuine_vocab_pairs": True,
        "database_table_counts_before_and_after": counts, "source_inputs_unchanged_sha256": {str(p): digest for p, digest in watched.items()},
        "output_sha256": {name: sha256(out / name) for name in ["matching_pairs.txt", "guided_pairs.txt", "retrieved_neighbors.json"]},
        "script_sha256": sha256(__file__),
        "limitations": ["The ordinary pair count is determined by top-K retrieval and is not exactly equal to E3; runtime and coverage comparisons must report both counts.",
            "Guided matching stays enabled with the same budget policy, but its selected pairs change as a consequence of the retrieved graph.",
            "Only guided subset selection uses earlier geometry/shared-body points; ordinary pair retrieval uses learned visual words on the same saved descriptors.",
            "This is an experiment on 125/290 images, not a large-scale vocabulary-tree speed benchmark."]}
    (out / "retrieval_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("subject", "images", "ordinary_pairs", "guided_pairs",
        "ordinary_components", "ordinary_pair_count_ratio_to_e3", "retrieval_total_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", nargs="*", choices=["light_shirt", "black_shirt_crutches"])
    parser.add_argument("--tree", type=Path, default=HERE / "vocab" / TREE_NAME)
    parser.add_argument("--output-root", type=Path, default=HERE / "vocab")
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.neighbors < 1 or args.threads < 1:
        parser.error("neighbors and threads must be positive")
    for subject in args.subjects or ["light_shirt", "black_shirt_crutches"]:
        prepare(subject, args)
