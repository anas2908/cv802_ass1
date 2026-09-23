    @run_on_thread
    def _estimate_cameras(self, recompute):
        """Sparse SfM for the starter GUI. Requires pycolmap==4.2.0."""
        import json
        import os
        import shutil
        import tempfile
        from pathlib import Path

        import pycolmap

        image_dir = Path(self.image_dir).resolve()
        database_path = Path(self.database_path).resolve()
        sparse_dir = Path(self.sparse_dir).resolve()
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        if len(images) < 2:
            raise ValueError(
                f"Put at least two overlapping JPG/PNG images in {image_dir}. "
                "Extract ZIPs and convert HEIC photos first."
            )

        # Track inputs; explicit Fit Colmap applies the current GUI settings.
        signature = {
            "camera_model": self.camera_model,
            "matcher": self.matcher,
            "pycolmap": pycolmap.__version__,
            "images": [
                [p.relative_to(image_dir).as_posix(),
                 p.stat().st_size, p.stat().st_mtime_ns]
                for p in images
            ],
        }
        # Prepared photo/video datasets group each device, lens and clip in
        # its own subfolder, so video frames share one calibration per clip.
        grouped_cameras = (image_dir.parent / "camera_groups.json").is_file()
        if grouped_cameras:
            signature["camera_mode"] = "PER_FOLDER"
        metadata_name = "sfm_inputs.json"

        def largest_model(folder):
            best = None
            if not folder.is_dir():
                return None
            for candidate in [folder] + sorted(
                p for p in folder.iterdir() if p.is_dir()
            ):
                if not any(
                    all((candidate / (name + ext)).is_file()
                        for name in ("cameras", "images", "points3D"))
                    for ext in (".bin", ".txt")
                ):
                    continue
                try:
                    model = pycolmap.Reconstruction(candidate)
                except (RuntimeError, ValueError) as error:
                    print(f"Skipping unreadable model {candidate}: {error}")
                    continue
                if model.num_reg_images() < 2 or model.num_points3D() == 0:
                    continue
                if best is None or (
                    model.num_reg_images(), model.num_points3D()
                ) > (best.num_reg_images(), best.num_points3D()):
                    best = model
            return best

        # Optional, dataset-local refinement reuses calibrated baseline poses.
        # All helpers remain nested so this entire method is paste-ready.
        quality = None
        dataset_dir = Path(self.data_path).resolve()
        refine_path = dataset_dir / "sfm_refine.json"
        if refine_path.is_file():
            import hashlib

            config = json.loads(refine_path.read_text())
            if not isinstance(config, dict):
                raise ValueError("sfm_refine.json must contain a JSON object")
            quality = dict(config)
            for key in ("baseline_dataset", "boxes", "matching_pairs", "guided_pairs"):
                if not isinstance(config.get(key), str) or not config[key]:
                    raise ValueError(f"sfm_refine.json requires a {key} path")
                quality[key] = (dataset_dir / config[key]).resolve()
            quality["guided_lock"] = (dataset_dir / config.get(
                "guided_lock", "../../work/quality-sfm/guided.lock"
            )).resolve()
            cap = config.get("max_features", 18000)
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
                raise ValueError("max_features must be a positive integer")
            quality["max_features"] = cap
            resume_matching = config.get("resume_matching", False)
            if not isinstance(resume_matching, bool):
                raise ValueError("resume_matching must be a boolean")
            quality["resume_matching"] = resume_matching
            for key, default in (("matching_batch_size", 128), ("matching_threads", 1)):
                value = config.get(key, default)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                quality[key] = value
            baseline_dir = quality["baseline_dataset"]
            baseline_database = baseline_dir / "colmap" / "database.db"
            baseline_sparse = baseline_dir / "colmap" / "sparse"
            if not baseline_database.is_file() or not baseline_sparse.is_dir():
                raise FileNotFoundError(f"Baseline reconstruction is missing: {baseline_dir}")
            quality["extraction_options"] = {
                "max_image_size": -1, "num_threads": 1,
                "sift": {
                    "first_octave": -1, "max_num_features": 16384,
                    "peak_threshold": 0.004, "estimate_affine_shape": True,
                    "domain_size_pooling": True, "dsp_num_scales": 10,
                },
            }
            baseline_files = [baseline_database] + sorted(
                p for p in baseline_sparse.rglob("*") if p.is_file()
            )
            feature_signature = {
                "recipe_version": 1,
                "images": signature["images"],
                "pycolmap": pycolmap.__version__,
                "baseline": [[str(p), p.stat().st_size, p.stat().st_mtime_ns]
                             for p in baseline_files],
                "boxes_sha256": hashlib.sha256(quality["boxes"].read_bytes()).hexdigest(),
                "extraction": quality["extraction_options"],
                "crop_padding": 32, "spatial_cells": [8, 16], "max_features": cap,
            }
            quality["feature_signature"] = feature_signature
            quality["feature_key"] = hashlib.sha256(json.dumps(
                feature_signature, sort_keys=True
            ).encode()).hexdigest()
            recipe = {
                "features": quality["feature_key"], "config": config,
                "matching_pairs_sha256": hashlib.sha256(
                    quality["matching_pairs"].read_bytes()).hexdigest(),
                "guided_pairs_sha256": hashlib.sha256(
                    quality["guided_pairs"].read_bytes()).hexdigest(),
                "triangulation": {"fixed_poses": True, "refine_intrinsics": False,
                                  "max_reprojection_error": 3.5, "min_angle": 1.5},
            }
            if resume_matching:
                # Resolve defaults only for the opt-in recipe. The original
                # feature key and every pre-E10 recipe hash remain unchanged.
                matching = pycolmap.FeatureMatchingOptions({
                    "num_threads": quality["matching_threads"],
                    "guided_matching": True, "use_gpu": False,
                })
                verification = pycolmap.TwoViewGeometryOptions()
                pairing = pycolmap.ImportedPairingOptions({
                    "block_size": min(32, quality["matching_batch_size"]),
                }).todict()
                pairing["match_list_path"] = "<current batch pair file>"
                quality["resume_matching_options"] = matching
                quality["resume_verification_options"] = verification
                quality["resume_pairing_block_size"] = pairing["block_size"]
                quality["matching_recipe"] = json.loads(json.dumps({
                    "checkpoint_version": 1, "features": quality["feature_key"],
                    "pycolmap": pycolmap.__version__, "device": "cpu",
                    "matching_pairs_sha256": recipe["matching_pairs_sha256"],
                    "guided_pairs_sha256": recipe["guided_pairs_sha256"],
                    "matching_options": matching.todict(),
                    "verification_options": verification.todict(),
                    "pairing_options": pairing,
                    "matching_batch_size": quality["matching_batch_size"],
                }, default=str))
            signature["quality_refinement"] = hashlib.sha256(json.dumps(
                recipe, sort_keys=True
            ).encode()).hexdigest()

        def refine_from_baseline(new_database, new_sparse):
            import fcntl
            from PIL import Image as PILImage

            baseline = largest_model(quality["baseline_dataset"] / "colmap" / "sparse")
            if baseline is None:
                raise ValueError("No usable baseline model for quality refinement")
            by_name = {baseline.images[i].name: baseline.images[i]
                       for i in baseline.reg_image_ids()}
            if set(by_name) != {row[0] for row in signature["images"]}:
                raise ValueError("Quality images must match the registered baseline image names")
            sizes = {}
            for path in images:
                name = path.relative_to(image_dir).as_posix()
                with PILImage.open(path) as photo:
                    size = photo.size
                camera_id = by_name[name].camera_id
                if camera_id in sizes and sizes[camera_id] != size:
                    raise ValueError(f"Mixed image dimensions for baseline camera {camera_id}")
                sizes[camera_id] = size
            for camera_id, size in sizes.items():
                camera = baseline.cameras[camera_id]
                camera.rescale(*size)
                camera.has_prior_focal_length = True

            def read_pairs(path):
                result = []
                seen = set()
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    names = line.split()
                    if len(names) != 2 or any(name not in by_name for name in names):
                        raise ValueError(f"Invalid image pair in {path}, line {number}")
                    pair = tuple(sorted(names))
                    if pair[0] == pair[1]:
                        raise ValueError(f"Self-match in {path}, line {number}")
                    if pair not in seen:
                        seen.add(pair)
                        result.append(pair)
                return result

            pairs = read_pairs(quality["matching_pairs"])
            guided_pairs = read_pairs(quality["guided_pairs"])
            if not pairs or not set(guided_pairs).issubset(set(pairs)):
                raise ValueError("Matching pairs must be nonempty and contain every guided pair")
            if quality["resume_matching"] and (
                set(guided_pairs) != set(pairs)
                or len(pairs) != len(by_name) * (len(by_name) - 1) // 2
            ):
                raise ValueError("Resumable matching requires every unique image pair, all guided")
            boxes = json.loads(quality["boxes"].read_text())
            if isinstance(boxes, list):
                boxes = {row["image_name"]: row for row in boxes}
            if not isinstance(boxes, dict):
                raise ValueError("Quality boxes must be a dictionary or image-record list")

            cache_dir = database_path.parent / "quality_feature_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_db = cache_dir / (quality["feature_key"] + ".db")
            cache_metadata = cache_dir / (quality["feature_key"] + ".json")
            cached = False
            try:
                metadata = json.loads(cache_metadata.read_text())
                cached = (metadata.get("signature") == quality["feature_signature"]
                          and cache_db.stat().st_size == metadata.get("database_size"))
            except (OSError, ValueError, AttributeError):
                pass
            if cached:
                shutil.copy2(cache_db, new_database)
                print("SfM quality: reused the completed feature checkpoint.", flush=True)
            else:
                shutil.copy2(quality["baseline_dataset"] / "colmap" / "database.db", new_database)
                extractor = pycolmap.FeatureExtractor.create(
                    pycolmap.FeatureExtractionOptions(quality["extraction_options"]),
                    pycolmap.Device.cpu,
                )
                total_features = 0
                with pycolmap.Database.open(new_database) as database:
                    database.clear_two_view_geometries()
                    database.clear_matches()
                    database.clear_keypoints()
                    database.clear_descriptors()
                    for camera in baseline.cameras.values():
                        database.update_camera(camera)
                    for number, path in enumerate(images, 1):
                        name = path.relative_to(image_dir).as_posix()
                        expected = by_name[name]
                        stored = database.read_image_with_name(name)
                        if stored is None or stored.image_id != expected.image_id or stored.camera_id != expected.camera_id:
                            raise ValueError(f"Baseline database/image IDs disagree for {name}")
                        record = boxes.get(name, {})
                        box = record.get("padded_bbox_xyxy") or record.get("feature_crop_bbox_xyxy")
                        with PILImage.open(path) as photo:
                            width, height = photo.size
                            if box is None:
                                x0, y0, x1, y1 = 0, 0, width, height
                            else:
                                rectangle = np.asarray(box, dtype=float)
                                if rectangle.shape != (4,) or not np.isfinite(rectangle).all():
                                    raise ValueError(f"Invalid feature crop for {name}")
                                x0 = max(0, int(np.floor(rectangle[0])) - 32)
                                y0 = max(0, int(np.floor(rectangle[1])) - 32)
                                x1 = min(width, int(np.ceil(rectangle[2])) + 32)
                                y1 = min(height, int(np.ceil(rectangle[3])) + 32)
                                if x1 <= x0 or y1 <= y0:
                                    raise ValueError(f"Empty feature crop for {name}")
                            pixels = np.ascontiguousarray(
                                photo.crop((x0, y0, x1, y1)).convert("L"), dtype=np.uint8
                            )
                        keypoints, descriptors = extractor.extract_from_uint8_array(pixels)
                        matrix = np.asarray([
                            [k.x + x0, k.y + y0, k.a11, k.a12, k.a21, k.a22]
                            for k in keypoints
                        ], dtype=np.float32).reshape(-1, 6)
                        data = np.asarray(descriptors.data)
                        if data.ndim != 2 or data.shape != (len(matrix), 128) or data.dtype != np.uint8:
                            raise ValueError(f"Invalid SIFT descriptors for {name}")
                        valid = (np.isfinite(matrix).all(axis=1)
                                 & (matrix[:, 0] >= 0) & (matrix[:, 0] < width)
                                 & (matrix[:, 1] >= 0) & (matrix[:, 1] < height))
                        matrix, data = matrix[valid], data[valid]
                        cap = quality["max_features"]
                        if len(matrix) > cap:
                            columns = np.clip(((matrix[:, 0] - x0) * 8 / (x1 - x0)).astype(int), 0, 7)
                            rows = np.clip(((matrix[:, 1] - y0) * 16 / (y1 - y0)).astype(int), 0, 15)
                            cells = rows * 8 + columns
                            quota, counts, selected = max(1, cap // 128), np.zeros(128, dtype=int), []
                            for index, cell in enumerate(cells):
                                if counts[cell] < quota:
                                    selected.append(index)
                                    counts[cell] += 1
                                    if len(selected) == cap:
                                        break
                            if len(selected) < cap:
                                unused = np.ones(len(matrix), dtype=bool)
                                unused[selected] = False
                                selected.extend(np.flatnonzero(unused)[:cap - len(selected)].tolist())
                            selected = np.asarray(sorted(selected), dtype=int)
                            matrix, data = matrix[selected], data[selected]
                        database.write_keypoints(expected.image_id, np.ascontiguousarray(matrix))
                        database.write_descriptors(expected.image_id, pycolmap.FeatureDescriptors(
                            descriptors.type, np.ascontiguousarray(data)
                        ))
                        total_features += len(matrix)
                        print(f"SfM quality: features {number}/{len(images)} — {len(matrix):,} in {name}", flush=True)
                if total_features == 0:
                    raise RuntimeError("No subject features were extracted")
                # Commit the database first and metadata last; partial extraction
                # never becomes a reusable checkpoint. Different recipes have
                # different filenames, avoiding stale-metadata overwrite races.
                with tempfile.TemporaryDirectory(prefix=".checkpoint_", dir=cache_dir) as staging:
                    staged_db = Path(staging) / "features.db"
                    staged_metadata = Path(staging) / "signature.json"
                    shutil.copy2(new_database, staged_db)
                    staged_metadata.write_text(json.dumps({
                        "signature": quality["feature_signature"],
                        "database_size": staged_db.stat().st_size,
                    }, sort_keys=True))
                    os.replace(staged_db, cache_db)
                    os.replace(staged_metadata, cache_metadata)

            def match_with_checkpoints():
                import sqlite3
                from contextlib import closing
                from datetime import datetime, timezone

                expected_images = {name: image.image_id for name, image in by_name.items()}
                pair_by_id = {
                    int(pycolmap.image_pair_to_pair_id(expected_images[a], expected_images[b])): (a, b)
                    for a, b in pairs
                }
                expected_ids = set(pair_by_id)
                if len(expected_ids) != len(pairs):
                    raise ValueError("Image pair IDs are not unique")
                with cache_db.open("rb") as stream:
                    feature_digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest = {
                    "recipe": quality["matching_recipe"],
                    "feature_database_sha256": feature_digest,
                    "images": sorted(expected_images.items()), "total_pairs": len(pairs),
                }
                # Normalize tuples before comparing to a JSON round trip.
                manifest = json.loads(json.dumps(manifest, sort_keys=True))
                key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                manifest["checkpoint_key"] = key
                checkpoints = database_path.parent / "quality_matching_cache"
                checkpoints.mkdir(exist_ok=True)
                folder = checkpoints / key
                stable = folder / "checkpoint.db"
                working = folder / ".working.db"
                global_lock = quality["guided_lock"]
                global_lock.parent.mkdir(parents=True, exist_ok=True)

                def sync_directory(path):
                    descriptor = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)

                def atomic_json(path, content):
                    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
                    temporary = Path(temporary)
                    try:
                        with os.fdopen(descriptor, "w") as stream:
                            json.dump(content, stream, sort_keys=True, indent=2)
                            stream.write("\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, path)
                        sync_directory(path.parent)
                    finally:
                        temporary.unlink(missing_ok=True)

                def completed_pairs(path, expected=None, immutable=False):
                    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
                    with closing(sqlite3.connect(path.as_uri() + suffix, uri=True)) as db:
                        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                            raise RuntimeError(f"Matching checkpoint failed SQLite quick_check: {path}")
                        names = {name: image_id for image_id, name in db.execute("SELECT image_id, name FROM images")}
                        if names != expected_images:
                            raise ValueError(f"Matching checkpoint image IDs do not match the recipe: {path}")
                        raw = {row[0] for row in db.execute("SELECT pair_id FROM matches")}
                        geometry = {row[0] for row in db.execute("SELECT pair_id FROM two_view_geometries")}
                    if not raw <= expected_ids or not geometry <= expected_ids:
                        raise ValueError(f"Matching checkpoint contains foreign image pairs: {path}")
                    # Zero-match and INVALID-geometry rows are completed attempts.
                    # A raw-only/geometry-only row is never committed as complete.
                    if raw != geometry or (expected is not None and raw != expected):
                        raise RuntimeError(f"Matching batch has missing or incomplete pair rows: {path}")
                    return raw

                def backup_database(source, destination, expected):
                    descriptor, temporary = tempfile.mkstemp(
                        prefix="." + destination.name + "_", suffix=".db", dir=destination.parent
                    )
                    os.close(descriptor)
                    temporary = Path(temporary)
                    try:
                        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
                            with closing(sqlite3.connect(temporary)) as dst:
                                src.backup(dst)
                                dst.commit()
                                # The durable snapshot is a single closed file,
                                # independent of COLMAP's WAL / synchronous=OFF.
                                dst.execute("PRAGMA journal_mode=DELETE")
                                dst.execute("PRAGMA synchronous=FULL")
                        completed_pairs(temporary, expected, immutable=True)
                        with temporary.open("rb") as stream:
                            os.fsync(stream.fileno())
                        os.replace(temporary, destination)
                        sync_directory(destination.parent)
                    finally:
                        for suffix in ("", "-wal", "-shm", "-journal"):
                            Path(str(temporary) + suffix).unlink(missing_ok=True)

                def save_progress(completed, last_batch=0):
                    atomic_json(folder / "progress.json", {
                        "checkpoint_key": key, "checkpoint_database": str(stable),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "phase": "matching_complete" if completed == expected_ids else "matching",
                        "complete": completed == expected_ids,
                        "total_pairs": len(expected_ids), "completed_pairs": len(completed),
                        "remaining_pairs": len(expected_ids - completed),
                        "last_committed_batch_pairs": last_batch,
                        "matching_batch_size": quality["matching_batch_size"],
                        "matching_threads": quality["matching_threads"],
                        "guided_matching": True,
                        "note": "Matching progress only; triangulation follows. The stable database is authoritative on resume.",
                    })

                print(f"SfM quality: resumable all-pairs guided matching checkpoint {folder}", flush=True)
                # The lock lives outside the directory so initialization can
                # publish its manifest and first stable database atomically.
                with (checkpoints / (key + ".lock")).open("a+") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        if not folder.exists():
                            completed_pairs(new_database, set())
                            with tempfile.TemporaryDirectory(prefix=".matching_init_", dir=checkpoints) as temporary:
                                staged = Path(temporary) / "checkpoint"
                                staged.mkdir()
                                backup_database(new_database, staged / "checkpoint.db", set())
                                atomic_json(staged / "manifest.json", manifest)
                                staged.rename(folder)
                                sync_directory(checkpoints)
                        if (folder.is_symlink() or not stable.is_file() or stable.is_symlink()
                                or not (folder / "manifest.json").is_file()
                                or json.loads((folder / "manifest.json").read_text()) != manifest):
                            raise RuntimeError(f"Incomplete or incompatible matching checkpoint; preserved at {folder}")
                        completed = completed_pairs(stable, immutable=True)
                        # Only the last durable snapshot is trusted after an
                        # interruption, never a surviving working WAL/database.
                        for suffix in ("", "-wal", "-shm", "-journal"):
                            Path(str(working) + suffix).unlink(missing_ok=True)
                        shutil.copy2(stable, working)
                        save_progress(completed)
                        remaining = [pair_id for pair_id in pair_by_id if pair_id not in completed]
                        batch_size = quality["matching_batch_size"]
                        for start in range(0, len(remaining), batch_size):
                            batch = remaining[start:start + batch_size]
                            pair_file = folder / ".batch_pairs.txt"
                            pair_file.write_text("".join(" ".join(pair_by_id[i]) + "\n" for i in batch))
                            with global_lock.open("a+") as guided_lock:
                                fcntl.flock(guided_lock.fileno(), fcntl.LOCK_EX)
                                try:
                                    # COLMAP performs initial appearance matching
                                    # and verification inside this guided call.
                                    pycolmap.match_image_pairs(
                                        database_path=working,
                                        matching_options=quality["resume_matching_options"],
                                        verification_options=quality["resume_verification_options"],
                                        pairing_options={"match_list_path": str(pair_file),
                                                         "block_size": quality["resume_pairing_block_size"]},
                                        device=pycolmap.Device.cpu,
                                    )
                                finally:
                                    fcntl.flock(guided_lock.fileno(), fcntl.LOCK_UN)
                            expected = completed | set(batch)
                            completed_pairs(working, expected)
                            backup_database(working, stable, expected)
                            completed = expected
                            save_progress(completed, len(batch))
                            print(f"SfM quality: committed guided pairs {len(completed):,}/{len(expected_ids):,}", flush=True)
                        completed_pairs(stable, expected_ids, immutable=True)
                        backup_database(stable, new_database, expected_ids)
                    finally:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

            if quality["resume_matching"]:
                match_with_checkpoints()
            else:
                print(f"SfM quality: matching {len(pairs):,} overlapping image pairs...", flush=True)
                pycolmap.match_image_pairs(
                    database_path=new_database,
                    matching_options={"num_threads": threads, "guided_matching": False},
                    pairing_options={"match_list_path": str(quality["matching_pairs"])},
                    device=pycolmap.Device.cpu,
                )
            if guided_pairs and not quality["resume_matching"]:
                lock_path = quality["guided_lock"]
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"SfM quality: waiting for guided matching ({len(guided_pairs):,} pairs)...", flush=True)
                with lock_path.open("a+") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        with pycolmap.Database.open(new_database) as database:
                            for first, second in guided_pairs:
                                database.delete_two_view_geometry(by_name[first].image_id, by_name[second].image_id)
                        # Existing raw matches are reverified and guided again
                        # when their geometry is absent; other pairs are retained.
                        pycolmap.match_image_pairs(
                            database_path=new_database,
                            matching_options={"num_threads": 1, "guided_matching": True},
                            pairing_options={"match_list_path": str(quality["guided_pairs"])},
                            device=pycolmap.Device.cpu,
                        )
                    finally:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            print("SfM quality: triangulating with the calibrated camera poses fixed...", flush=True)
            result = pycolmap.triangulate_points(
                reconstruction=baseline, database_path=new_database,
                image_path=image_dir, output_path=new_sparse / "0",
                clear_points=True, refine_intrinsics=False,
                options={
                    "num_threads": threads, "extract_colors": True,
                    "mapper": {"filter_max_reproj_error": 3.5},
                    "triangulation": {"ignore_two_view_tracks": True, "min_angle": 1.5},
                },
            )
            if result.num_reg_images() != len(by_name) or result.num_points3D() == 0:
                raise RuntimeError("Quality triangulation did not retain the calibrated image set with valid points")
            return result

        reconstruction = None
        metadata_path = sparse_dir / metadata_name
        if not recompute:
            try:
                saved = (json.loads(metadata_path.read_text())
                         if metadata_path.exists() else None)
                # Opening an existing result should not recompute merely
                # because the fitting dropdowns differ from the saved model.
                # Fit Colmap passes recompute=True to apply new settings.
                same_inputs = (saved is None and quality is None) or (isinstance(saved, dict) and
                    saved.get("images") == signature["images"]
                    and saved.get("camera_mode", "AUTO")
                    == signature.get("camera_mode", "AUTO")
                    and saved.get("quality_refinement") == signature.get("quality_refinement")
                )
                if same_inputs:
                    reconstruction = largest_model(sparse_dir)
            except (OSError, ValueError):
                pass  # Incomplete cache: run SfM again.

        if reconstruction is None:
            matchers = {
                "exhaustive_matcher": pycolmap.match_exhaustive,
                "sequential_matcher": pycolmap.match_sequential,
                "vocab_tree_matcher": pycolmap.match_vocabtree,
            }
            if quality is None and self.matcher not in matchers:
                raise ValueError(f"Unsupported matcher: {self.matcher}")
            pairing_options = {}
            if quality is None and self.matcher == "sequential_matcher":
                # Requires filenames in capture order; no vocabulary download.
                pairing_options = {"overlap": 10, "loop_detection": False}
            elif quality is None and self.matcher == "vocab_tree_matcher":
                vocab = Path(__file__).resolve().parent / Path(VOCAB_PATH).name
                if not vocab.is_file():
                    raise FileNotFoundError(
                        f"Missing vocabulary tree: {vocab}. "
                        "Select exhaustive_matcher instead."
                    )
                pairing_options = {"vocab_tree_path": str(vocab)}

            database_path.parent.mkdir(parents=True, exist_ok=True)
            sparse_dir.parent.mkdir(parents=True, exist_ok=True)
            threads = min(4, os.cpu_count() or 1)

            # CPU execution works on macOS without a separate COLMAP install.
            # Stage the run so failures do not erase an existing reconstruction.
            with tempfile.TemporaryDirectory(
                prefix="sfm_run_", dir=database_path.parent
            ) as temporary:
                work = Path(temporary)
                new_database = work / "database.db"
                new_sparse = work / "sparse"
                new_sparse.mkdir()
                if quality is not None:
                    reconstruction = refine_from_baseline(new_database, new_sparse)
                else:
                    print(f"SfM: extracting features from {len(images)} images...")
                    pycolmap.extract_features(
                        database_path=new_database,
                        image_path=image_dir,
                        image_names=[row[0] for row in signature["images"]],
                        camera_mode=(pycolmap.CameraMode.PER_FOLDER if grouped_cameras
                                     else pycolmap.CameraMode.AUTO),
                        reader_options={"camera_model": self.camera_model},
                        extraction_options={
                            "max_image_size": 3200, "num_threads": threads,
                            "sift": {"max_num_features": 16384, "first_octave": 0},
                        },
                        device=pycolmap.Device.cpu,
                    )
                    print(f"SfM: matching with {self.matcher}...")
                    # CPU guided matching allocates large dense pairwise matrices.
                    # Geometric verification still rejects inconsistent matches.
                    matchers[self.matcher](
                        database_path=new_database,
                        matching_options={"num_threads": threads, "guided_matching": False},
                        pairing_options=pairing_options,
                        device=pycolmap.Device.cpu,
                    )
                    print("SfM: estimating cameras and triangulating points...")
                    models = pycolmap.incremental_mapping(
                        database_path=new_database,
                        image_path=image_dir,
                        output_path=new_sparse,
                        options={"num_threads": threads, "extract_colors": True},
                    )
                    valid = [m for m in models.values()
                             if m.num_reg_images() >= 2 and m.num_points3D() > 0]
                    if not valid:
                        raise RuntimeError(
                            "SfM could not reconstruct these images. Use sharp, "
                            "overlapping views of one stationary scene, captured "
                            "from different camera positions."
                        )
                    reconstruction = max(
                        valid, key=lambda m: (m.num_reg_images(), m.num_points3D())
                    )
                (new_sparse / metadata_name).write_text(json.dumps(signature))

                backup = work / "previous"
                backup.mkdir()
                moved, installed = [], []
                try:
                    for destination in (database_path, sparse_dir):
                        if destination.exists():
                            saved = backup / destination.name
                            destination.rename(saved)
                            moved.append((saved, destination))
                    for source, destination in (
                        (new_database, database_path), (new_sparse, sparse_dir)
                    ):
                        source.rename(destination)
                        installed.append(destination)
                except Exception:
                    for destination in reversed(installed):
                        if destination.is_dir():
                            shutil.rmtree(destination)
                        else:
                            destination.unlink()
                    for saved, destination in reversed(moved):
                        saved.rename(destination)
                    raise
        else:
            print("SfM: loaded the cached reconstruction.")

        points = list(reconstruction.points3D.values())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray([p.xyz for p in points], dtype=np.float64)
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray([p.color for p in points], dtype=np.float64) / 255.0
        )

        colmap_cameras = {}
        registered = [reconstruction.images[i]
                      for i in reconstruction.reg_image_ids()]
        for image in sorted(registered, key=lambda image: image.name):
            camera = reconstruction.cameras[image.camera_id]
            K = camera.calibration_matrix()
            pose = image.cam_from_world()  # World-to-camera; do not invert.
            colmap_cameras[image.name] = {
                "extrinsic": [pose.rotation.matrix(), pose.translation.copy()],
                "intrinsic": {
                    "width": int(camera.width), "height": int(camera.height),
                    "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                },
            }

        self._pcd = pcd
        self._cameras = colmap_cameras
        self.activate_camera_name = self.camera_names[0]
        print(
            f"SfM complete: {len(colmap_cameras)}/{len(images)} images registered, "
            f"{len(points):,} sparse points in the largest connected model."
        )
