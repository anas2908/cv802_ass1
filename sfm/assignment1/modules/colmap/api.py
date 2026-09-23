import numpy as np
import open3d as o3d
import os
import os.path as osp
import glob
import shlex
import shutil
import subprocess
import tempfile

from modules.colmap.model_io import (
    camera_to_pinhole_intrinsics,
    qvec_to_rotation_matrix,
    read_text_model,
)
from utils.thread_utils import run_on_thread


VOCAB_PATH = osp.join(osp.dirname(__file__), 'vocab_tree_flickr100K_words32K.bin')


def _contains_sparse_model(directory):
    """Return whether ``directory`` contains a COLMAP sparse model."""
    if not osp.isdir(directory):
        return False

    stems = ('cameras', 'images', 'points3D')
    return any(
        all(osp.isfile(osp.join(directory, stem + extension)) for stem in stems)
        for extension in ('.bin', '.txt')
    )


def _find_sparse_models(sparse_dir):
    """Find sparse models written either directly or in numbered folders."""
    models = []
    if _contains_sparse_model(sparse_dir):
        models.append(sparse_dir)

    if osp.isdir(sparse_dir):
        for name in sorted(os.listdir(sparse_dir)):
            candidate = osp.join(sparse_dir, name)
            if _contains_sparse_model(candidate):
                models.append(candidate)

    return models


def _resolve_colmap_executable():
    configured = os.environ.get('COLMAP_EXECUTABLE')
    executable = shutil.which(configured) if configured else shutil.which('colmap')

    if executable is None and configured and osp.isfile(configured):
        executable = configured

    if executable is None:
        raise RuntimeError(
            'COLMAP was not found. Install COLMAP and make the `colmap` command '
            'available, or set COLMAP_EXECUTABLE to its full path.'
        )

    return executable


def _run_colmap(executable, arguments):
    command = [executable, *map(str, arguments)]
    print('Running:', shlex.join(command))
    subprocess.run(command, check=True)


def _run_colmap_with_gpu_fallback(executable, arguments):
    """Retry feature work on CPU if a requested GPU execution fails."""
    try:
        _run_colmap(executable, arguments)
        return
    except subprocess.CalledProcessError:
        cpu_arguments = list(arguments)
        changed = False

        index = 0
        while index < len(cpu_arguments):
            argument = str(cpu_arguments[index])
            if argument.endswith('.use_gpu') and index + 1 < len(cpu_arguments):
                if str(cpu_arguments[index + 1]) == '1':
                    cpu_arguments[index + 1] = '0'
                    changed = True
                index += 2
                continue

            if argument.endswith('.gpu_index') and index + 1 < len(cpu_arguments):
                del cpu_arguments[index:index + 2]
                changed = True
                continue

            index += 1

        if not changed:
            raise

        print('GPU execution failed; retrying this COLMAP stage on the CPU.')
        _run_colmap(executable, cpu_arguments)


def _colmap_help(executable, command):
    completed = subprocess.run(
        [executable, command, '-h'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.stdout or ''


def _gpu_arguments(help_text, prefixes, gpu_index):
    """Build GPU flags for both legacy and recent COLMAP option names."""
    for prefix in prefixes:
        use_gpu_option = f'--{prefix}.use_gpu'
        if use_gpu_option not in help_text:
            continue

        use_gpu = int(gpu_index is not None and int(gpu_index) >= 0)
        arguments = [use_gpu_option, str(use_gpu)]

        index_option = f'--{prefix}.gpu_index'
        if use_gpu and index_option in help_text:
            arguments.extend([index_option, str(int(gpu_index))])

        return arguments

    # If help output is unavailable, let COLMAP use its own defaults.
    return []


def _load_sparse_model(model_dir, executable=None):
    """Load one sparse model, converting COLMAP binary output when needed."""
    stems = ('cameras', 'images', 'points3D')
    has_binary_model = all(
        osp.isfile(osp.join(model_dir, stem + '.bin')) for stem in stems
    )
    has_text_model = all(
        osp.isfile(osp.join(model_dir, stem + '.txt')) for stem in stems
    )

    if has_binary_model:
        executable = executable or _resolve_colmap_executable()
        with tempfile.TemporaryDirectory(prefix='cv802_colmap_text_') as text_dir:
            _run_colmap(
                executable,
                [
                    'model_converter',
                    '--input_path', model_dir,
                    '--output_path', text_dir,
                    '--output_type', 'TXT',
                ],
            )
            return read_text_model(text_dir)

    if has_text_model:
        return read_text_model(model_dir)

    raise FileNotFoundError(f'Incomplete COLMAP sparse model: {model_dir}')


def _validate_sparse_model(model_dir, cameras, images, points3d):
    """Validate the subset of a sparse model consumed by the GUI."""
    if not images:
        raise RuntimeError(
            f'The sparse model contains no registered images: {model_dir}'
        )
    if not points3d:
        raise RuntimeError(
            f'The sparse model contains no reconstructed 3D points: {model_dir}'
        )

    image_names = set()
    for image in images.values():
        camera_id = image['camera_id']
        if camera_id not in cameras:
            raise KeyError(
                f'Registered image {image["name"]} refers to missing camera '
                f'{camera_id}'
            )
        if image['name'] in image_names:
            raise ValueError(
                f'Duplicate registered image name in {model_dir}: '
                f'{image["name"]}'
            )
        image_names.add(image['name'])

        rotation = qvec_to_rotation_matrix(image['qvec'])
        translation = np.asarray(image['tvec'], dtype=np.float64)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError(
                f'Invalid translation for registered image {image["name"]}'
            )
        if not np.all(np.isfinite(rotation)):
            raise ValueError(
                f'Invalid rotation for registered image {image["name"]}'
            )

        intrinsic = camera_to_pinhole_intrinsics(cameras[camera_id])
        if intrinsic['width'] <= 0 or intrinsic['height'] <= 0:
            raise ValueError(f'Invalid image dimensions for camera {camera_id}')
        numeric_intrinsics = np.asarray(
            [
                intrinsic['fx'], intrinsic['fy'],
                intrinsic['cx'], intrinsic['cy'],
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(numeric_intrinsics))
            or intrinsic['fx'] <= 0
            or intrinsic['fy'] <= 0
        ):
            raise ValueError(f'Invalid intrinsics for camera {camera_id}')

    for point_id, point in points3d.items():
        xyz = np.asarray(point['xyz'], dtype=np.float64)
        rgb = np.asarray(point['rgb'])
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f'Invalid coordinates for 3D point {point_id}')
        if rgb.shape != (3,):
            raise ValueError(f'Invalid color for 3D point {point_id}')


def _load_largest_sparse_model(model_dirs, executable=None):
    """Load the largest valid connected component from COLMAP's output."""
    loaded_models = []
    errors = []
    for model_dir in model_dirs:
        try:
            cameras, images, points3d = _load_sparse_model(
                model_dir,
                executable=executable,
            )
            _validate_sparse_model(model_dir, cameras, images, points3d)
            loaded_models.append((model_dir, cameras, images, points3d))
        except Exception as error:
            errors.append(f'{model_dir}: {error}')

    if not loaded_models:
        detail = '; '.join(errors) if errors else 'no model directories found'
        raise RuntimeError(f'No valid COLMAP sparse model was found ({detail})')

    return max(
        loaded_models,
        key=lambda model: (len(model[2]), len(model[3])),
    )


def _run_sparse_pipeline(
    executable,
    image_dir,
    database_path,
    sparse_dir,
    camera_model,
    matcher,
    gpu_index,
):
    """Run COLMAP feature extraction, matching, and incremental mapping."""
    if matcher == 'vocab_tree_matcher' and not osp.isfile(VOCAB_PATH):
        raise FileNotFoundError(
            'The vocabulary-tree matcher was selected, but its vocabulary '
            f'file is missing: {VOCAB_PATH}. Use exhaustive/sequential '
            'matching or download the vocabulary tree.'
        )

    os.makedirs(osp.dirname(database_path), exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    feature_help = _colmap_help(executable, 'feature_extractor')
    feature_arguments = [
        'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--ImageReader.camera_model', camera_model,
    ]
    feature_arguments.extend(
        _gpu_arguments(
            feature_help,
            ('FeatureExtraction', 'SiftExtraction'),
            gpu_index,
        )
    )
    _run_colmap_with_gpu_fallback(executable, feature_arguments)

    matcher_help = _colmap_help(executable, matcher)
    matcher_arguments = [matcher, '--database_path', database_path]
    matcher_arguments.extend(
        _gpu_arguments(
            matcher_help,
            ('FeatureMatching', 'SiftMatching'),
            gpu_index,
        )
    )

    if matcher == 'vocab_tree_matcher':
        matcher_arguments.extend(
            ['--VocabTreeMatching.vocab_tree_path', VOCAB_PATH]
        )

    _run_colmap_with_gpu_fallback(executable, matcher_arguments)
    _run_colmap(
        executable,
        [
            'mapper',
            '--database_path', database_path,
            '--image_path', image_dir,
            '--output_path', sparse_dir,
        ],
    )


def _replace_cached_reconstruction(
    staged_database,
    staged_sparse_dir,
    database_path,
    sparse_dir,
):
    """Install a validated recomputation while preserving the old cache on error."""
    cache_root = osp.dirname(database_path)
    os.makedirs(cache_root, exist_ok=True)
    backup_root = tempfile.mkdtemp(prefix='.cv802_previous_', dir=cache_root)
    backup_database = osp.join(backup_root, 'database.db')
    backup_sparse = osp.join(backup_root, 'sparse')

    previous_database_moved = False
    previous_sparse_moved = False
    staged_database_installed = False
    staged_sparse_installed = False

    try:
        if not osp.isfile(staged_database):
            raise FileNotFoundError(
                f'COLMAP did not create its database: {staged_database}'
            )
        if not osp.isdir(staged_sparse_dir):
            raise FileNotFoundError(
                f'COLMAP did not create its sparse output: {staged_sparse_dir}'
            )

        if osp.isfile(database_path):
            shutil.move(database_path, backup_database)
            previous_database_moved = True
        if osp.isdir(sparse_dir):
            shutil.move(sparse_dir, backup_sparse)
            previous_sparse_moved = True

        shutil.move(staged_database, database_path)
        staged_database_installed = True
        shutil.move(staged_sparse_dir, sparse_dir)
        staged_sparse_installed = True
    except Exception as install_error:
        recovery_errors = []

        try:
            if staged_database_installed and osp.isfile(database_path):
                os.remove(database_path)
        except Exception as error:
            recovery_errors.append(f'could not remove new database: {error}')

        try:
            if staged_sparse_installed and osp.isdir(sparse_dir):
                shutil.rmtree(sparse_dir)
        except Exception as error:
            recovery_errors.append(f'could not remove new sparse model: {error}')

        try:
            if previous_database_moved and osp.isfile(backup_database):
                shutil.move(backup_database, database_path)
        except Exception as error:
            recovery_errors.append(f'could not restore old database: {error}')

        try:
            if previous_sparse_moved and osp.isdir(backup_sparse):
                shutil.move(backup_sparse, sparse_dir)
        except Exception as error:
            recovery_errors.append(f'could not restore old sparse model: {error}')

        if recovery_errors:
            raise RuntimeError(
                'Installing the new COLMAP cache failed and rollback was '
                f'incomplete. Recovery files remain in {backup_root}. '
                + '; '.join(recovery_errors)
            ) from install_error

        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_root, ignore_errors=True)


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
            raise ValueError(
                'Only exhaustive_matcher, vocab_tree_matcher, and '
                f'sequential_matcher are supported; got {self._matcher}'
            )

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
            len(_find_sparse_models(sparse_dir)) > 0

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

        image_dir = Path(self.image_dir).resolve() # images hai ismai
        database_path = Path(self.database_path).resolve() # sift jaisai features ka database
        sparse_dir = Path(self.sparse_dir).resolve() # point cloud and camera poses ka folder
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
            "camera_model": self.camera_model, # OpenCV camera model name, e.g., PINHOLE, SIMPLE_RADIAL, etc.
            "matcher": self.matcher, # exhaustive/vocab/ or custome made selective 
            "pycolmap": pycolmap.__version__,
            "images": [ # relative to image_dir
                [p.relative_to(image_dir).as_posix(),
                 p.stat().st_size, p.stat().st_mtime_ns]
                for p in images
            ],
        }
        # Prepared photo/video datasets group each device, lens and clip in
        # its own subfolder, so video frames share one calibration per clip.
        grouped_cameras = (image_dir.parent / "camera_groups.json").is_file() #PER_FOLDER means images within each folder share one set of intrinsic parameters
        if grouped_cameras:
            signature["camera_mode"] = "PER_FOLDER"
        metadata_name = "sfm_inputs.json"

        def largest_model(folder):
            best = None
            if not folder.is_dir():
                return None
            for candidate in [folder] + sorted( #iterates through subfolders
                p for p in folder.iterdir() if p.is_dir()
            ):
                if not any(
                    all((candidate / (name + ext)).is_file()
                        for name in ("cameras", "images", "points3D"))
                    for ext in (".bin", ".txt")
                ):
                    continue # if no .bin/.txt then continue to next candidate folder
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
            return best # best is one with more 3d points and more registered images

        # Optional, dataset-local refinement reuses calibrated baseline poses.
        # All helpers remain nested so this entire method is paste-ready.
        quality = None # for our qualitative experiments
        dataset_dir = Path(self.data_path).resolve()
        refine_path = dataset_dir / "sfm_refine.json" 
        if refine_path.is_file(): #prepares and validates the higher-quality refinement configuration.
            import hashlib

            config = json.loads(refine_path.read_text()) #Read refinement instructions
            if not isinstance(config, dict): # checking its a dictionary
                raise ValueError("sfm_refine.json must contain a JSON object") 
            quality = dict(config)
            for key in ("baseline_dataset", "boxes", "matching_pairs", "guided_pairs"): # for loop, baseline dataset is E1
                if not isinstance(config.get(key), str) or not config[key]: # reject missing/incorrect/empty values
                    raise ValueError(f"sfm_refine.json requires a {key} path")
                quality[key] = (dataset_dir / config[key]).resolve() # convert into absolute path
            quality["guided_lock"] = (dataset_dir / config.get(
                "guided_lock", "../../work/quality-sfm/guided.lock"
            )).resolve() #bcs using mac initially i had to use guided matching only one experiment at a time
            cap = config.get("max_features", 18000) # max features per image
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
                raise ValueError("max_features must be a positive integer")
            quality["max_features"] = cap
            resume_matching = config.get("resume_matching", False) # if more images comes so u can continue from last checkpoint
            if not isinstance(resume_matching, bool):
                raise ValueError("resume_matching must be a boolean")
            quality["resume_matching"] = resume_matching
            for key, default in (("matching_batch_size", 128), ("matching_threads", 1)): # save chkpt adter 128 matching pair image
                value = config.get(key, default)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                quality[key] = value
            baseline_dir = quality["baseline_dataset"]
            baseline_database = baseline_dir / "colmap" / "database.db"
            baseline_sparse = baseline_dir / "colmap" / "sparse"
            if not baseline_database.is_file() or not baseline_sparse.is_dir():
                raise FileNotFoundError(f"Baseline reconstruction is missing: {baseline_dir}")
            quality["extraction_options"] = { # sift property, Define higher-detail SIFT
                "max_image_size": -1, "num_threads": 1,
                "sift": {
                    "first_octave": -1, "max_num_features": 16384, # feature extraction=16384, feature selection max is 18k
                    "peak_threshold": 0.004, "estimate_affine_shape": True,
                    "domain_size_pooling": True, "dsp_num_scales": 10, # hifher quality sift, but slower
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
                "boxes_sha256": hashlib.sha256(quality["boxes"].read_bytes()).hexdigest(), #bounding boxes through apple mac
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
                quality["resume_verification_options"] = verification # verifying whether two sift fatures are of same correspondence
                quality["resume_pairing_block_size"] = pairing["block_size"] # number of image pairs getting mathced in one batch
                quality["matching_recipe"] = json.loads(json.dumps({
                    "checkpoint_version": 1, "features": quality["feature_key"],
                    "pycolmap": pycolmap.__version__, "device": "cpu",
                    "matching_pairs_sha256": recipe["matching_pairs_sha256"],
                    "guided_pairs_sha256": recipe["guided_pairs_sha256"],
                    "matching_options": matching.todict(),
                    "verification_options": verification.todict(), #RANSAC
                    "pairing_options": pairing,
                    "matching_batch_size": quality["matching_batch_size"],
                }, default=str))
            signature["quality_refinement"] = hashlib.sha256(json.dumps(
                recipe, sort_keys=True
            ).encode()).hexdigest()

        def refine_from_baseline(new_database, new_sparse): #e3
            import fcntl
            from PIL import Image as PILImage

            baseline = largest_model(quality["baseline_dataset"] / "colmap" / "sparse") #It loads the whole model, including cameras and points—not only cameras.
            if baseline is None:
                raise ValueError("No usable baseline model for quality refinement")
            by_name = {baseline.images[i].name: baseline.images[i]
                       for i in baseline.reg_image_ids()} #create dict (preprocessing step)
            if set(by_name) != {row[0] for row in signature["images"]}: #just matching names
                raise ValueError("Quality images must match the registered baseline image names")
            sizes = {}
            for path in images: #Get its relative filename and (width, height)
                name = path.relative_to(image_dir).as_posix()
                with PILImage.open(path) as photo:
                    size = photo.size
                camera_id = by_name[name].camera_id
                if camera_id in sizes and sizes[camera_id] != size: # verifying the grouping of camera's images
                    raise ValueError(f"Mixed image dimensions for baseline camera {camera_id}")
                sizes[camera_id] = size
            for camera_id, size in sizes.items(): #rescale groups images
                camera = baseline.cameras[camera_id]
                camera.rescale(*size)
                camera.has_prior_focal_length = True

            def read_pairs(path):
                result = []
                seen = set() #duplicate tracker
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

            pairs = read_pairs(quality["matching_pairs"]) # selective matching (matching_pairs.txt)
            guided_pairs = read_pairs(quality["guided_pairs"]) #[("imageA.jpg", "imageB.jpg"), ("imageA.jpg", "imageC.jpg")]
            
#             In [prepare_and_run.py, line 66 (line 66)](/home/anas.khan/cv802_project/project1/ass1/sfm/legacy_tools/work/matching-ablation/E6/prepare_and_run.py:66):
# (dataset / 'guided_pairs.txt').write_text('') --> For E6, guided matching is disabled by creating an empty guided-pair file.
            
            
            if not pairs or not set(guided_pairs).issubset(set(pairs)):
                raise ValueError("Matching pairs must be nonempty and contain every guided pair")
            if quality["resume_matching"] and (
                set(guided_pairs) != set(pairs)
                or len(pairs) != len(by_name) * (len(by_name) - 1) // 2 #N × (N − 1) / 2
            ):
                raise ValueError("Resumable matching requires every unique image pair, all guided")
            boxes = json.loads(quality["boxes"].read_text()) #prepared person-box/crop records.
            if isinstance(boxes, list):
                boxes = {row["image_name"]: row for row in boxes} #If records arrive as a list, reorganize them into a dictionary:
            if not isinstance(boxes, dict):
                raise ValueError("Quality boxes must be a dictionary or image-record list")

            cache_dir = database_path.parent / "quality_feature_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_db = cache_dir / (quality["feature_key"] + ".db") #database containing extracted features
            cache_metadata = cache_dir / (quality["feature_key"] + ".json") #notes describing that database and its inputs/settings
            cached = False
            try:
                metadata = json.loads(cache_metadata.read_text())
                cached = (metadata.get("signature") == quality["feature_signature"]
                          and cache_db.stat().st_size == metadata.get("database_size"))
                #Both conditions must hold:
                # - The saved feature signature matches our current one.
                # - The database’s file size matches its recorded size.
            except (OSError, ValueError, AttributeError):
                pass
            if cached:
                shutil.copy2(cache_db, new_database)
                print("SfM quality: reused the completed feature checkpoint.", flush=True)
            else:
                shutil.copy2(quality["baseline_dataset"] / "colmap" / "database.db", new_database) 
                extractor = pycolmap.FeatureExtractor.create(#Extract person-focused features
                    pycolmap.FeatureExtractionOptions(quality["extraction_options"]),
                    pycolmap.Device.cpu,
                )
                total_features = 0
                with pycolmap.Database.open(new_database) as database:
                    database.clear_two_view_geometries()
                    database.clear_matches()
                    database.clear_keypoints()
                    database.clear_descriptors() #Why? We are about to extract a different feature set.
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
                        keypoints, descriptors = extractor.extract_from_uint8_array(pixels) #sift
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
                        if len(matrix) > cap: #Distribute retained features, 8x16 grid feature selection
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

            def match_with_checkpoints(): #guided matching in batches and saves verified checkpoints.
                import sqlite3 #After an interruption, it can restart from the last good checkpoint. It does not perform triangulation.
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
                                    pycolmap.match_image_pairs( #Run COLMAP matching for the current batch
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
                match_with_checkpoints() #E10 
            else:
                print(f"SfM quality: matching {len(pairs):,} overlapping image pairs...", flush=True)
                pycolmap.match_image_pairs( #Match selected pairs
                    database_path=new_database,
                    matching_options={"num_threads": threads, "guided_matching": False},
                    pairing_options={"match_list_path": str(quality["matching_pairs"])},
                    device=pycolmap.Device.cpu,
                )
            if guided_pairs and not quality["resume_matching"]: #guided mathcing on/off #e6
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
                        pycolmap.match_image_pairs( #Apply guided matching
                            database_path=new_database,
                            matching_options={"num_threads": 1, "guided_matching": True},
                            pairing_options={"match_list_path": str(quality["guided_pairs"])},
                            device=pycolmap.Device.cpu,
                        )
                    finally:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            print("SfM quality: triangulating with the calibrated camera poses fixed...", flush=True)
            result = pycolmap.triangulate_points( #Reconstruct new points
                reconstruction=baseline, database_path=new_database,
                image_path=image_dir, output_path=new_sparse / "0",
                clear_points=True, refine_intrinsics=False,
                options={
                    "num_threads": threads, "extract_colors": True,
                    "mapper": {"filter_max_reproj_error": 3.5},
                    "triangulation": {"ignore_two_view_tracks": True, "min_angle": 1.5}, #equire sufficient separation between viewing rays
                },
            )
            if result.num_reg_images() != len(by_name) or result.num_points3D() == 0:
                raise RuntimeError("Quality triangulation did not retain the calibrated image set with valid points")
            return result

        reconstruction = None
        metadata_path = sparse_dir / metadata_name #Metadata: its filename, file size and modification time.
        if not recompute:
            try:
                saved = (json.loads(metadata_path.read_text())
                         if metadata_path.exists() else None)
                # Opening an existing result should not recompute merely
                # because the fitting dropdowns differ from the saved model.
                # Fit Colmap passes recompute=True to apply new settings.
                same_inputs = (saved is None and quality is None) or (isinstance(saved, dict) and #“There are no saved input notes, and this is not quality refinement. Try loading it as an older baseline result.”
                    saved.get("images") == signature["images"] 
                    and saved.get("camera_mode", "AUTO")
                    == signature.get("camera_mode", "AUTO")
                    and saved.get("quality_refinement") == signature.get("quality_refinement") #Saved notes exist and match
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
            elif quality is None and self.matcher == "vocab_tree_matcher": #clustering.  
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
                    reconstruction = refine_from_baseline(new_database, new_sparse) # higher reso 
                else:
                    print(f"SfM: extracting features from {len(images)} images...")
                    pycolmap.extract_features( #sift features #e1
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
                    #PyCOLMAP uses the verified correspondences to:
                    # - Initialize a reconstruction.
                    # - Register additional cameras.
                    # - Triangulate 3D points.
                    # - Refine cameras and points through bundle adjustment.
                    # - Extract point colors from the images.
                    # models can contain multiple disconnected reconstructions.
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

                backup = work / "previous"  #saving files its fine not to read this
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
        pcd = o3d.geometry.PointCloud() #Create an Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray([p.xyz for p in points], dtype=np.float64)
        ) #Add the XYZ coordinates
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray([p.color for p in points], dtype=np.float64) / 255.0
        ) #Add the point colors

        colmap_cameras = {}
        registered = [reconstruction.images[i]
                      for i in reconstruction.reg_image_ids()] #Collect successfully registered images
        for image in sorted(registered, key=lambda image: image.name): #Read each image’s calibration and pose
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
            }#Store the camera information

        self._pcd = pcd
        self._cameras = colmap_cameras
        self.activate_camera_name = self.camera_names[0] #Give the results to the application
        print(
            f"SfM complete: {len(colmap_cameras)}/{len(images)} images registered, "
            f"{len(points):,} sparse points in the largest connected model."
        ) # summary 






    @staticmethod
    def _list_images_in_folder(directory):
        image_extensions = {
            '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff'
        }
        files = glob.glob(osp.join(directory, '**', '*'), recursive=True)
        return sorted(
            path
            for path in files
            if osp.isfile(path)
            and osp.splitext(path)[1].lower() in image_extensions
        )

    def estimate_done(self):
        return not self._thread.is_alive()

    @property
    def estimate_error(self):
        if self._thread is None:
            return None
        return getattr(self._thread, 'exception', None)

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



# Person/background cleanup is outside api.py. It is a separate operation on a completed reconstruction.
# - Initial rectangle cleanup: [make_subject_preview.py (line 192)](/home/anas.khan/cv802_project/project1/ass1/sfm/legacy_tools/work/full-capture/make_subject_preview.py:192).
# - Light-shirt mask cleanup: [filter_person_masks.py (line 61)](/home/anas.khan/cv802_project/project1/ass1/sfm/legacy_tools/reconstructions/experiments/light_person_mask_cleanup/filter_person_masks.py:61).
# - Dark-shirt cleanup with crutch protection: [filter_person_crutches.py (line 38)](/home/anas.khan/cv802_project/project1/ass1/sfm/legacy_tools/reconstructions/experiments/black_person_crutches_cleanup/filter_person_crutches.py:38).


# run_e10.py
# ├── run_quality.py → _estimate_cameras() → save reconstruction
# ├── make_subject_preview.py → rectangle-based filtering #e2
# └── filter_person_masks.py → 90% mask cleanup #e4,e5


#run_cleanup.py - e7,e9


# prepare_vocab_pairs.py - e8