import numpy as np
import open3d as o3d
import os.path as osp
import glob

from utils.thread_utils import run_on_thread


VOCAB_PATH = 'modules/colmap/vocab_tree_flickr100K_words32K.bin'


class ColmapAPI:
    def __init__(
        self,
        gpu_index,
        camera_model,
        matcher,
    ):
        self._data_path = None
        self._pcd = None
        self._thread = None
        self._active_camera_name = None
        self._cameras = dict()
        self._vis = None

        self._gpu_index = gpu_index
        self._camera_model = camera_model
        self._matcher = matcher
        if self._matcher not in ['exhaustive_matcher', 'vocab_tree_matcher', 'sequential_matcher']:
            raise ValueError(f'Only support exhaustive_matcher and vocab_tree_matcher, got {self._matcher}')

    @property
    def data_path(self):
        if self._data_path is None:
            raise ValueError(f'Data path was not set')
        return self._data_path

    @data_path.setter
    def data_path(self, new_data_path):
        self._data_path = new_data_path

    @property
    def image_dir(self):
        return osp.join(self.data_path, 'images')

    @property
    def database_path(self):
        return osp.join(self.data_path, 'colmap/database.db')

    @property
    def sparse_dir(self):
        return osp.join(self.data_path, 'colmap/sparse')

    @property
    def num_cameras(self):
        return len(self._cameras)

    @property
    def camera_names(self):
        return list(self._cameras.keys())

    @property
    def pcd(self):
        if self._pcd is None:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._pcd

    @property
    def activate_camera_name(self):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._active_camera_name

    @activate_camera_name.setter
    def activate_camera_name(self, new_value):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        self._active_camera_name = new_value

    @property
    def camera_model(self):
        return self._camera_model

    @camera_model.setter
    def camera_model(self, new_value):
        self._camera_model = new_value 

    @property
    def matcher(self):
        return self._matcher

    @matcher.setter
    def matcher(self, new_value):
        self._matcher = new_value

    def check_colmap_folder_valid(self):
        database_path = self.database_path
        image_dir = self.image_dir
        sparse_dir = self.sparse_dir

        print('Database file:', database_path)
        print('Image path:', image_dir)
        print('Bundle adjustment path:', sparse_dir)

        is_valid = \
            osp.isfile(database_path) and \
            osp.isdir(image_dir) and \
            osp.isdir(sparse_dir)

        return is_valid

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

        # Rebuild if the inputs or GUI settings changed.
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
                same_inputs = (
                    not metadata_path.exists()
                    or json.loads(metadata_path.read_text()) == signature
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
            threads = min(8, os.cpu_count() or 1)

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
                    camera_mode=pycolmap.CameraMode.AUTO,
                    reader_options={"camera_model": self.camera_model},
                    extraction_options={
                        "max_image_size": 2000, "num_threads": threads,
                        "sift": {"max_num_features": 8192},
                    },
                    device=pycolmap.Device.cpu,
                )
                print(f"SfM: matching with {self.matcher}...")
                matchers[self.matcher](
                    database_path=new_database,
                    matching_options={"num_threads": threads},
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


    @staticmethod
    def _list_images_in_folder(directory):
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'}
        files = sorted(glob.glob(osp.join(directory, '*')))
        files = list(filter(lambda x: osp.splitext(x)[1].lower() in image_extensions, files)) 
        return files

    def estimate_done(self):
        return not self._thread.is_alive()

    def estimate_cameras(self, recompute=False):
        self._thread = self._estimate_cameras(recompute)

    def extract_camera_parameters(self, camera_name):
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self._cameras[camera_name]['intrinsic']['width'],
            self._cameras[camera_name]['intrinsic']['height'],
            self._cameras[camera_name]['intrinsic']['fx'],
            self._cameras[camera_name]['intrinsic']['fy'],
            self._cameras[camera_name]['intrinsic']['cx'],
            self._cameras[camera_name]['intrinsic']['cy'],
        )

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = self._cameras[camera_name]['extrinsic'][0]
        extrinsics[:3, 3] = self._cameras[camera_name]['extrinsic'][1]
        extrinsics = extrinsics

        return intrinsics, extrinsics
