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

        reconstruction = None
        metadata_path = sparse_dir / metadata_name
        if not recompute:
            try:
                saved = (json.loads(metadata_path.read_text())
                         if metadata_path.exists() else None)
                # Opening an existing result should not recompute merely
                # because the fitting dropdowns differ from the saved model.
                # Fit Colmap passes recompute=True to apply new settings.
                same_inputs = saved is None or (
                    saved.get("images") == signature["images"]
                    and saved.get("camera_mode", "AUTO")
                    == signature.get("camera_mode", "AUTO")
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
            if self.matcher not in matchers:
                raise ValueError(f"Unsupported matcher: {self.matcher}")
            pairing_options = {}
            if self.matcher == "sequential_matcher":
                # Requires filenames in capture order; no vocabulary download.
                pairing_options = {"overlap": 10, "loop_detection": False}
            elif self.matcher == "vocab_tree_matcher":
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
