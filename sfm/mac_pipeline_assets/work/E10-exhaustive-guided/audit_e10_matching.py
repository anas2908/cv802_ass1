#!/usr/bin/env python3
"""Read-only final proof of E10 exhaustive guided matching.

Run after reconstruction finishes, against its INTERNAL quality dataset, not
the consensus-90 preview (whose GUI database intentionally contains no matches):

  .venv/bin/python work/E10-exhaustive-guided/audit_e10_matching.py \
      work/E10-exhaustive-guided/datasets/light_shirt_quality --output REPORT.json

No reconstruction or matching is run. The existing database audit supplies
feature-byte comparison, saved recipe selection and one feature-checkpoint hash.
The active .working.db is never opened; an exclusive writer prevents this audit.
Documented worker migrations must preserve every completed parent pair and every
non-matching database row exactly. Only worker count and checkpoint batch cadence
may change; no feature, matching, or verification threshold change is accepted.
"""
from __future__ import annotations

import argparse
from contextlib import closing, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import combinations, zip_longest
import json
from pathlib import Path
import sqlite3
import sys

import pycolmap

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT / "work" / "quality-sfm"))
from audit_database import audit_database, _select_checkpoint


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_state(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pair_file(path, names):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pair = line.split()
        require(len(pair) == 2 and pair[0] != pair[1] and set(pair) <= names,
                f"Invalid pair at {path}:{number}")
        rows.append(tuple(sorted(pair)))
    require(len(rows) == len(set(rows)), f"Duplicate requested pairs: {path}")
    return set(rows)


def current_matching_recipe(config, key, ordinary_hash, guided_hash):
    require(config.get("resume_matching") is True, "E10 must enable resume_matching")
    batch = config.get("matching_batch_size", 128)
    threads = config.get("matching_threads", 1)
    require(type(batch) is int and batch > 0, "Invalid matching_batch_size")
    require(type(threads) is int and threads in (1, 2, 4), "This E10 audit supports one, two or four matching threads")
    matching = pycolmap.FeatureMatchingOptions({"num_threads": threads, "guided_matching": True, "use_gpu": False})
    verification = pycolmap.TwoViewGeometryOptions()
    pairing = pycolmap.ImportedPairingOptions({"block_size": min(32, batch)}).todict()
    pairing["match_list_path"] = "<current batch pair file>"
    return json.loads(json.dumps({
        "checkpoint_version": 1, "features": key, "pycolmap": pycolmap.__version__, "device": "cpu",
        "matching_pairs_sha256": ordinary_hash, "guided_pairs_sha256": guided_hash,
        "matching_options": matching.todict(), "verification_options": verification.todict(),
        "pairing_options": pairing, "matching_batch_size": batch,
    }, default=str))


def matching_counts(db, expected_ids):
    require(db.execute("PRAGMA quick_check").fetchall() == [("ok",)], "Matching database failed quick_check")
    result = {}
    for table in ("matches", "two_view_geometries"):
        ids = {row[0] for row in db.execute(f"SELECT pair_id FROM {table}")}
        require(ids == expected_ids,
                f"{table}: expected all pairs; missing={len(expected_ids - ids)}, foreign={len(ids - expected_ids)}")
        count, correspondences, zero, negative = db.execute(
            f"SELECT COUNT(*), COALESCE(SUM(rows),0), COALESCE(SUM(rows=0),0), COALESCE(SUM(rows<0),0) FROM {table}"
        ).fetchone()
        require(negative == 0, f"Negative correspondence count in {table}")
        result[table] = {"exact_all_pair_ID_coverage": True, "pair_records": count,
                         "correspondences": correspondences, "zero_correspondence_pairs": zero,
                         "positive_correspondence_pairs": count - zero}
    labels = {int(value): name for name, value in pycolmap.TwoViewGeometryConfiguration.__members__.items()}
    configurations = []
    usable = unusable = 0
    for config, count, positive, correspondences in db.execute(
        "SELECT config,COUNT(*),SUM(rows>0),SUM(rows) FROM two_view_geometries GROUP BY config ORDER BY config"
    ):
        label = labels.get(config, f"UNKNOWN_{config}")
        require(config in labels, f"Unknown two-view configuration: {config}")
        good = positive if label not in {"UNDEFINED", "DEGENERATE", "WATERMARK"} else 0
        usable += good
        unusable += count - good
        configurations.append({"configuration": config, "name": label, "pairs": count,
                               "pairs_with_inliers": positive, "verified_correspondences": correspondences})
    result["two_view_geometries"].update(
        configuration_counts=configurations, usable_geometry_pairs=usable,
        invalid_or_unusable_geometry_pairs=unusable,
        usable_definition="At least one inlier and configuration other than UNDEFINED, DEGENERATE or WATERMARK; not a ground-truth correctness claim",
    )
    return result


def same_matching_records(first, second):
    result = {}
    for table in ("matches", "two_view_geometries"):
        first_schema = first.execute(f"PRAGMA table_info({table})").fetchall()
        second_schema = second.execute(f"PRAGMA table_info({table})").fetchall()
        require(first_schema == second_schema, f"Checkpoint/final {table} schemas differ")
        count = 0
        for left, right in zip_longest(first.execute(f"SELECT * FROM {table} ORDER BY pair_id"),
                                      second.execute(f"SELECT * FROM {table} ORDER BY pair_id")):
            require(left == right, f"Checkpoint/final {table} records differ at pair {(left or right)[0]}")
            count += 1
        result[table] = {"records_compared": count, "every_field_and_blob_identical": True}
    return result


def worker_only_change(parent, child):
    """Prove only execution settings changed; return their exact differences."""
    old = {k: v for k, v in parent.items() if k != "checkpoint_key"}
    new = {k: v for k, v in child.items() if k != "checkpoint_key"}
    before, after = old["recipe"], new["recipe"]
    old_threads, new_threads = (v["matching_options"]["num_threads"] for v in (before, after))
    require(type(old_threads) is int and type(new_threads) is int
            and (old_threads, new_threads) in {(1, 2), (2, 4), (4, 2)},
            "Unsupported worker transition; expected 1->2, 2->4 or 4->2")
    for recipe in (before, after):
        batch = recipe["matching_batch_size"]
        require(type(batch) is int and batch > 0
                and recipe["pairing_options"]["block_size"] == min(32, batch),
                "Checkpoint cadence or derived pairing block size is invalid")
        require(recipe["matching_options"]["guided_matching"] is True
                and recipe["matching_options"]["use_gpu"] is False and recipe["device"] == "cpu",
                "Every migration stage must remain guided CPU matching")
    expected = json.loads(json.dumps(old))
    expected["recipe"]["matching_options"]["num_threads"] = new_threads
    expected["recipe"]["matching_batch_size"] = after["matching_batch_size"]
    expected["recipe"]["pairing_options"]["block_size"] = after["pairing_options"]["block_size"]
    require(expected == new, "Migration changed features, thresholds or identities beyond execution settings")
    changes = {"num_threads": {"before": old_threads, "after": new_threads}}
    for name, old_value, new_value in (
            ("matching_batch_size", before["matching_batch_size"], after["matching_batch_size"]),
            ("pairing_options.block_size", before["pairing_options"]["block_size"], after["pairing_options"]["block_size"])):
        if old_value != new_value:
            changes[name] = {"before": old_value, "after": new_value}
    return changes


def inherited_database_records(parent, child, expected_ids):
    """Compare committed parent rows, including empty/invalid pair attempts."""
    require(parent.execute("PRAGMA quick_check").fetchall() == [("ok",)],
            "Parent checkpoint failed quick_check")
    pair_ids = {table: {r[0] for r in parent.execute(f"SELECT pair_id FROM {table}")}
                for table in ("matches", "two_view_geometries")}
    inherited = pair_ids["matches"]
    require(inherited and inherited == pair_ids["two_view_geometries"] and inherited <= expected_ids,
            "Parent does not contain an equal, nonempty subset of committed raw/geometric pairs")
    child_ids = [{r[0] for r in child.execute(f"SELECT pair_id FROM {table}")} for table in pair_ids]
    require(child_ids[0] == child_ids[1] and inherited <= child_ids[0] <= expected_ids,
            "Child raw/geometric pair coverage is inconsistent or lost inherited pairs")
    records = {}
    for table in pair_ids:
        require(parent.execute(f"PRAGMA table_info({table})").fetchall()
                == child.execute(f"PRAGMA table_info({table})").fetchall(),
                f"Parent/child {table} schemas differ")
        zero = 0
        for row in parent.execute(f"SELECT * FROM {table} ORDER BY pair_id"):
            require(row == child.execute(f"SELECT * FROM {table} WHERE pair_id=?", (row[0],)).fetchone(),
                    f"Inherited {table} fields or blobs changed at pair {row[0]}")
            zero += row[1] == 0
        records[table] = {"records_compared": len(inherited), "zero_correspondence_pairs": zero,
                          "every_field_and_blob_identical": True}
    # Matching must not change image identities, cameras, features, rigs, or any
    # other non-matching table. Stream full rows rather than loading descriptors.
    tables = []
    for db in (parent, child):
        tables.append({r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                       if not r[0].startswith("sqlite_") and r[0] not in pair_ids})
    require(tables[0] == tables[1], "Parent/child non-matching table sets differ")
    unchanged = {}
    for table in sorted(tables[0]):
        quoted = '"' + table.replace('"', '""') + '"'
        schema = parent.execute(f"PRAGMA table_info({quoted})").fetchall()
        require(schema == child.execute(f"PRAGMA table_info({quoted})").fetchall(),
                f"Parent/child {table} schemas differ")
        primary = [r[1] for r in sorted(schema, key=lambda r: r[5]) if r[5]]
        order = ",".join('"' + name.replace('"', '""') + '"' for name in primary) or "rowid"
        count = 0
        for left, right in zip_longest(parent.execute(f"SELECT * FROM {quoted} ORDER BY {order}"),
                                      child.execute(f"SELECT * FROM {quoted} ORDER BY {order}")):
            require(left == right, f"Parent/child non-matching data differ in {table}")
            count += 1
        unchanged[table] = {"records_compared": count, "every_field_and_blob_identical": True}
    return inherited, {"matching_tables": records, "non_matching_tables": unchanged,
                       "child_completed_pairs": len(child_ids[0])}


def audit_worker_migration(config, manifest_path, manifest, all_manifests, child_db,
                           expected_ids, feature_digest, watched):
    cache = manifest_path.parent.parent
    records = {}
    for path in sorted((WORK / "worker_migrations").glob("*.json")):
        value = read(path)
        records.setdefault(value.get("target_checkpoint_key"), []).append((path, value))
    selected_key = manifest["checkpoint_key"]
    visited, reverse_hops = {selected_key}, []
    current_config = config
    with ExitStack() as stack:
        while manifest["recipe"]["matching_options"]["num_threads"] != 1:
            candidates = records.get(manifest["checkpoint_key"], [])
            require(len(candidates) == 1, "Each migrated checkpoint needs exactly one recorded parent")
            record_path, record = candidates[0]
            watched[record_path] = file_state(record_path)
            source_key = record["source_checkpoint_key"]
            require(source_key not in visited, "Cycle in checkpoint migration lineage")
            visited.add(source_key)
            parent_folder = cache / source_key
            parent_path, parent_db_path = parent_folder / "manifest.json", parent_folder / "checkpoint.db"
            require(Path(record["source_checkpoint_dir"]).resolve() == parent_folder.resolve()
                    and Path(record["target_checkpoint_dir"]).resolve() == manifest_path.parent.resolve(),
                    "Migration directories differ from the audited checkpoint chain")
            require(not parent_folder.is_symlink() and parent_db_path.is_file() and not parent_db_path.is_symlink(),
                    "Parent must be a preserved independent checkpoint")
            require(file_state(parent_db_path)[:2] != file_state(manifest_path.parent / "checkpoint.db")[:2],
                    "Parent and child checkpoint databases must have independent inodes")
            parent_manifest = read(parent_path)
            for path, value in ((parent_path, parent_manifest), (manifest_path, manifest)):
                content = {k: v for k, v in value.items() if k != "checkpoint_key"}
                require(value["checkpoint_key"] == path.parent.name == json_hash(content),
                        "Migration manifest content-derived key mismatch")
                watched[path] = file_state(path)
            changes = worker_only_change(parent_manifest, manifest)
            require(digest(parent_path) == record["source_manifest_sha256"]
                    and digest(manifest_path) == record["target_manifest_sha256"], "Migration manifest hash mismatch")
            require(record["only_matching_option_changed"] == changes,
                    "Migration record differs from the actual execution-option changes")
            before, after = record["config_before"], record["config_after"]
            expected_config = dict(before)
            for key in ("matching_threads", "matching_batch_size"):
                require(type(before.get(key)) is int and type(after.get(key)) is int,
                        "Migration execution configuration must contain integer worker and batch values")
                expected_config[key] = after[key]
            require(expected_config == after == current_config,
                    "Migration changed configuration beyond worker count/cadence, or configuration chain differs")
            for configuration, value in ((before, parent_manifest), (after, manifest)):
                require(configuration["matching_threads"] == value["recipe"]["matching_options"]["num_threads"]
                        and configuration["matching_batch_size"] == value["recipe"]["matching_batch_size"],
                        "Migration config does not match its checkpoint execution settings")
            require(record["source_feature_database_sha256"] == parent_manifest["feature_database_sha256"]
                    == manifest["feature_database_sha256"] == feature_digest,
                    "Feature bytes differ across the migration")
            lock_path = cache / (source_key + ".lock")
            require(lock_path.is_file(), "Parent checkpoint lock is missing")
            lock = stack.enter_context(lock_path.open("rb"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            watched[parent_db_path] = file_state(parent_db_path)
            wal = Path(str(parent_db_path) + "-wal")
            require(not wal.exists() or wal.stat().st_size == 0, "Parent snapshot has a nonempty WAL")
            parent_digest = digest(parent_db_path)
            require(parent_digest == record["source_checkpoint_sha256"]
                    == record["target_initial_checkpoint_sha256"], "Preserved parent database hash mismatch")
            parent_db = stack.enter_context(closing(sqlite3.connect(
                parent_db_path.as_uri() + "?mode=ro&immutable=1", uri=True)))
            inherited, comparisons = inherited_database_records(parent_db, child_db, expected_ids)
            require(len(inherited) == record["completed_pairs_transferred"]
                    and json_hash(sorted(inherited)) == record["transferred_pair_ids_sha256"],
                    "Inherited completed pair IDs differ from the migration record")
            reverse_hops.append({"record": str(record_path), "record_sha256": digest(record_path),
                "source_checkpoint_key": source_key, "target_checkpoint_key": manifest["checkpoint_key"],
                "source_database_sha256": parent_digest, "source_manifest_sha256": record["source_manifest_sha256"],
                "target_manifest_sha256": record["target_manifest_sha256"], "only_execution_options_changed": changes,
                "inherited_completed_pairs": len(inherited), "inherited_pair_ids_sha256": json_hash(sorted(inherited)),
                "source_workers": before["matching_threads"], "target_workers": after["matching_threads"],
                "source_batch_size": before["matching_batch_size"], "target_batch_size": after["matching_batch_size"],
                **comparisons})
            current_config, manifest_path, manifest, child_db = before, parent_path, parent_manifest, parent_db
        require(not records.get(manifest["checkpoint_key"]), "Original one-worker seed has an unexpected parent")
    require({row["checkpoint_key"] for row in all_manifests} == visited,
            "Unexpected checkpoint outside the documented migration lineage")
    hops = list(reversed(reverse_hops))
    segments = [{"checkpoint_key": manifest["checkpoint_key"], "workers": 1,
                 "matching_batch_size": manifest["recipe"]["matching_batch_size"],
                 "completed_pairs_contributed": hops[0]["inherited_completed_pairs"] if hops else len(expected_ids)}]
    for hop in hops:
        segments.append({"checkpoint_key": hop["target_checkpoint_key"], "workers": hop["target_workers"],
                         "matching_batch_size": hop["target_batch_size"],
                         "completed_pairs_contributed": hop["child_completed_pairs"] - hop["inherited_completed_pairs"]})
    require(sum(v["completed_pairs_contributed"] for v in segments) == len(expected_ids),
            "Execution lineage contributions do not account for every final pair")
    return {"required": bool(hops), "verified": True, "final_checkpoint_key": selected_key,
            "original_checkpoint_key": manifest["checkpoint_key"], "migration_count": len(hops),
            "lineage": hops, "execution_segments": segments}


def audit(dataset):
    report = {"dataset": str(dataset), "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Final internal E10 matching database and stable checkpoint; no active working DB or reconstruction opened",
              "checks": {}, "violations": [], "warnings": [], "passed": False}
    watched = {}
    try:
        config_path = dataset / "sfm_refine.json"
        inputs_path = dataset / "colmap/sparse/sfm_inputs.json"
        canonical = dataset / "colmap/database.db"
        require(inputs_path.is_file() and canonical.is_file(), "Final internal E10 reconstruction is not ready")
        config, saved = read(config_path), read(inputs_path)
        metadata_path, metadata, feature_key, _ = _select_checkpoint(dataset, config, saved)
        ordinary_path, guided_path = ((dataset / config[k]).resolve() for k in ("matching_pairs", "guided_pairs"))
        ordinary_hash, guided_hash = digest(ordinary_path), digest(guided_path)
        recipe = current_matching_recipe(config, feature_key, ordinary_hash, guided_hash)
        outer_recipe = {"features": feature_key, "config": config,
                        "matching_pairs_sha256": ordinary_hash, "guided_pairs_sha256": guided_hash,
                        "triangulation": {"fixed_poses": True, "refine_intrinsics": False,
                                          "max_reprojection_error": 3.5, "min_angle": 1.5}}
        require(json_hash(outer_recipe) == saved["quality_refinement"], "Final saved quality recipe does not match current inputs")
        cache = dataset / "colmap/quality_matching_cache"
        manifests, candidates = [], []
        for path in sorted(cache.glob("*/manifest.json")):
            manifest = read(path)
            key = manifest.get("checkpoint_key")
            content = {k: v for k, v in manifest.items() if k != "checkpoint_key"}
            require(key == path.parent.name == json_hash(content), f"Invalid immutable checkpoint key: {path}")
            options = manifest["recipe"]["matching_options"]
            require(options.get("guided_matching") is True and options.get("use_gpu") is False
                    and type(options.get("num_threads")) is int and options["num_threads"] in (1, 2, 4)
                    and manifest["recipe"].get("device") == "cpu",
                    f"A checkpoint manifest is not guided/CPU/supported-worker-count: {path}")
            manifests.append({"manifest": str(path), "checkpoint_key": key, "sha256": digest(path),
                              "guided_CPU": True, "matching_threads": options["num_threads"], "key_matches_content": True})
            watched[path] = file_state(path)
            if manifest["recipe"] == recipe:
                candidates.append((path, manifest))
        require(len(candidates) == 1, f"Expected exactly one checkpoint for the final recipe; found {len(candidates)}")
        manifest_path, manifest = candidates[0]
        folder, key = manifest_path.parent, manifest["checkpoint_key"]
        stable, progress_path = folder / "checkpoint.db", folder / "progress.json"
        require(read(progress_path).get("complete") is True, "Matching is not complete; run this audit after completion")
        require(stable.is_file() and not stable.is_symlink(), "Missing independent stable checkpoint database")
        lock_path = cache / (key + ".lock")
        require(lock_path.is_file(), "Checkpoint writer lock is missing")
        # Shared nonblocking lock refuses an ongoing matching session. Opening
        # the existing lock read-only creates no files or changes to its content.
        with lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            for path in (config_path, inputs_path, canonical, metadata_path,
                         ordinary_path, guided_path, stable, progress_path):
                watched[path] = file_state(path)
            baseline = (dataset / config["baseline_dataset"]).resolve()
            database_audit = audit_database(baseline, dataset)
            report["feature_database_audit"] = database_audit
            require(database_audit["passed"], "Existing feature/database audit failed; see feature_database_audit")
            feature_digest = database_audit["checkpoint"]["database_sha256_computed_for_audit"]
            require(manifest["feature_database_sha256"] == feature_digest, "Stable matching recipe used another feature database")
            provenance_path = dataset / "experiment_provenance.json"
            provenance = read(provenance_path)
            source = Path(provenance["source_E3_dataset"]).resolve()
            source_config = read(source / "sfm_refine.json")
            source_metadata, source_signature, source_key, _ = _select_checkpoint(
                source, source_config, read(source / "colmap/sparse/sfm_inputs.json"))
            source_features = source_metadata.with_suffix(".db")
            watched[source_features] = file_state(source_features)
            require(source_key == feature_key and source_signature["signature"] == metadata["signature"],
                    "E10 feature signature differs from E3")
            source_digest = digest(source_features)
            require(source_digest == feature_digest, "E3 feature database differs from the E10 feature copy")
            prepared = [row for row in provenance["feature_copies"] if Path(row["copy"]).suffix == ".db"]
            require(len(prepared) == 1 and prepared[0]["sha256"] == feature_digest,
                    "Feature bytes differ from the frozen E10 preparation")
            with ExitStack() as stack:
                databases = {name: stack.enter_context(closing(sqlite3.connect(
                    path.as_uri() + "?mode=ro&immutable=1", uri=True)))
                    for name, path in (("final", canonical), ("stable", stable), ("E3_features", source_features))}
                for name in ("final", "stable"):
                    # A nonempty WAL could contain unpublished data; immutable
                    # access is permitted here only for closed complete files.
                    wal = Path(str(canonical if name == "final" else stable) + "-wal")
                    require(not wal.exists() or wal.stat().st_size == 0, f"Nonempty WAL next to {name} database")
                images = {name: image_id for image_id, name in databases["final"].execute("SELECT image_id,name FROM images")}
                require({name: image_id for image_id, name in databases["stable"].execute("SELECT image_id,name FROM images")} == images,
                        "Stable/final image IDs differ")
                require(manifest["images"] == [[name, image_id] for name, image_id in sorted(images.items())],
                        "Manifest image identities differ from the final database")
                names = set(images)
                require(names == {row[0] for row in saved["images"]}, "Final database image set differs from saved inputs")
                expected_names = set(combinations(sorted(names), 2))
                require(pair_file(ordinary_path, names) == pair_file(guided_path, names) == expected_names,
                        "Requested ordinary and guided files are not the exact full unordered image-pair set")
                expected_ids = {int(pycolmap.image_pair_to_pair_id(a, b)) for a, b in combinations(sorted(images.values()), 2)}
                require(len(expected_ids) == manifest["total_pairs"] == len(images) * (len(images) - 1) // 2,
                        "Manifest pair total does not equal N choose 2")
                report["matching_tables"] = {name: matching_counts(databases[name], expected_ids) for name in ("stable", "final")}
                report["checkpoint_vs_final_matching_records"] = same_matching_records(databases["stable"], databases["final"])
                report["worker_migration"] = audit_worker_migration(
                    config, manifest_path, manifest, manifests, databases["stable"],
                    expected_ids, feature_digest, watched)
                initial_counts = {table: databases["E3_features"].execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                                  for table in ("matches", "two_view_geometries")}
                require(all(value == 0 for value in initial_counts.values()), "Original E3 feature cache contains precomputed matching rows")
            scope_path = WORK / "code_scope_audit.json"
            scope = read(scope_path)
            api_path = ROOT / "assignment1/modules/colmap/api.py"
            require(digest(api_path) == scope["api_sha256"], "Backend changed since the reviewed E10 checkpoint implementation")
            progress = read(progress_path)
            require(progress.get("checkpoint_key") == key and progress.get("completed_pairs") == len(expected_ids)
                    and progress.get("remaining_pairs") == 0, "Final progress does not agree with stable database coverage")
            report.update(
                checkpoint_key=key, checkpoint_manifest=str(manifest_path), checkpoint_database=str(stable),
                checkpoint_manifest_audits=manifests, matching_recipe=recipe,
                feature_key=feature_key, feature_database_sha256=feature_digest,
                original_E3_feature_database=str(source_features), original_E3_feature_sha256=source_digest,
                source_feature_matching_counts=initial_counts,
                final_quality_recipe_sha256=saved["quality_refinement"], final_quality_recipe=outer_recipe,
                image_count=len(images), expected_unique_pairs=len(expected_ids), progress=progress,
                reviewed_backend_sha256=scope["api_sha256"],
                matching_mode_provenance=("The verified feature-only seed has zero matches/geometries; its byte hash is bound into "
                    "the immutable guided-only recipe. The reviewed backend enforces empty initial matching tables and calls guided=True "
                    "on every requested pair, with no separate ordinary-only pass. The documented worker lineage contributes "
                    "only the exact committed parent rows verified in worker_migration; worker count and checkpoint cadence "
                    "may change while features, pair coverage and quality thresholds remain unchanged. SQLite does not store a per-pair guided flag, so "
                    "mode provenance comes from this recipe/code/seed chain; zero-inlier or unsupported geometry remains a completed attempt."),
            )
            report["checks"] = {
                "every_checkpoint_manifest_guided_CPU_supported_worker_count": True,
                "worker_migration_preserves_options_features_and_inherited_pair_blobs": report["worker_migration"]["verified"],
                "selected_manifest_key_and_full_options_match_current_recipe": True,
                "final_saved_recipe_selects_the_E3_feature_key": True,
                "original_E3_and_E10_feature_checkpoint_bytes_identical": True,
                "final_database_features_identical_to_source_checkpoint": database_audit["checks"]["checkpoint_feature_rows_byte_identical_to_canonical"],
                "feature_only_seed_has_no_ordinary_matching_or_geometry_rows": True,
                "backend_matches_reviewed_guided_only_checkpoint_implementation": True,
                "both_pair_files_exactly_N_choose_2": True,
                "stable_and_final_both_tables_have_every_expected_pair_ID": True,
                "stable_and_final_matching_records_byte_identical": True,
                "all_matching_attempts_committed": True,
            }
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, RuntimeError) as error:
        report["violations"].append({"type": type(error).__name__, "message": str(error)})
    changed = [str(path) for path, state in watched.items() if not path.exists() or file_state(path) != state]
    report["checks"]["input_file_states_unchanged_during_audit"] = not changed
    if changed:
        report["violations"].append({"message": "Inputs changed during audit", "paths": changed})
    report["passed"] = not report["violations"] and all(report["checks"].values())
    report["limitations"] = [
        "All-pair completion includes failed/zero-inlier attempts; it does not mean every image pair overlaps or produces 3D points.",
        "Guidance needs a usable initially verified two-view geometry; requesting guided matching does not ensure that every pair can use it.",
        "Matching completeness and preserved feature bytes do not establish anatomical reconstruction accuracy.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Completed internal E10 quality dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve()),
                      "image_count": report.get("image_count"), "expected_pairs": report.get("expected_unique_pairs"),
                      "violations": report["violations"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
