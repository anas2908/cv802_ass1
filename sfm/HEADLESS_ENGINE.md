# Headless SfM engine: implementation and viva guide

This document explains the source-only server engine in `sfm_engine/` at the
level expected in a viva. It describes a **fresh-camera sparse
Structure-from-Motion (SfM)** pipeline. It is not MVS, a mesh generator, or the
historical fixed-pose E3–E10 refinement code.

## Storage contract

The server deployment has one permanent heavy-data root:

```text
/l/users/anas.khan/cv_802_ass1/
└── sfm/
    ├── inputs/<dataset>/images/       # source images for this method
    └── experiments/<experiment>/     # DBs, attempts, models, PLYs, receipts
```

The code stays in:

```text
/home/anas.khan/cv802_project/project1/ass1/sfm/
```

`CV802_DATA_ROOT` is accepted only as an explicit absolute path. Every input,
experiment, model, point cloud, and generated destination is resolved and
checked against `DATA_ROOT/sfm`. Symlinked destinations that resolve outside it
are rejected. On the server, leave `CV802_DATA_ROOT` set to the permanent path
above. An explicit override exists for a Mac/external-disk installation; there
is no implicit current-directory or home-directory fallback.

## Commands

The environment itself belongs in the data root. The production command must
run on the allocated compute node, not the login node:

```bash
export CODE_ROOT=/home/anas.khan/cv802_project/project1/ass1/sfm
export CV802_DATA_ROOT=/l/users/anas.khan/cv_802_ass1
export SFM_DATA_ROOT="$CV802_DATA_ROOT/sfm"
export TMPDIR="$SFM_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$SFM_DATA_ROOT/cache/xdg"
export PYTHONPYCACHEPREFIX="$SFM_DATA_ROOT/cache/python"
export PIP_CACHE_DIR="$SFM_DATA_ROOT/cache/pip"
export PYTHONNOUSERSITE=1

cd "$CODE_ROOT"
./scripts/setup_linux_cuda.sh

"$SFM_DATA_ROOT/envs/headless-cuda/bin/python" -B run_headless.py run \
  --experiment E11_linux_cuda_fresh_light_1600_v1 \
  --images "$SFM_DATA_ROOT/inputs/light_shirt/images" \
  --camera-model SIMPLE_RADIAL \
  --camera-mode per_folder \
  --matcher exhaustive \
  --device cuda \
  --max-image-size 1600 \
  --max-num-features 8192 \
  --num-threads 16 \
  --random-seed 0
```

`setup_linux_cuda.sh` is idempotent and writes its environment and package
receipts below `SFM_DATA_ROOT`; it must not be redirected to a home-directory
environment. The command shown above is the exact validated E11 configuration.
An identical later invocation is also the saved-model/cache-reload check.

Run a storage/input check without importing PyCOLMAP or touching an experiment:

```bash
cd /home/anas.khan/cv802_project/project1/ass1/sfm
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python -B \
  run_headless.py run \
  --experiment E11_linux_cuda_fresh_light_1600_v1 \
  --images /l/users/anas.khan/cv_802_ass1/sfm/inputs/light_shirt/images \
  --dry-run
```

Validate a saved sparse model without reconstructing:

```bash
cd /home/anas.khan/cv802_project/project1/ass1/sfm
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python -B \
  run_headless.py validate \
  --model /l/users/anas.khan/cv_802_ass1/sfm/experiments/E11_linux_cuda_fresh_light_1600_v1/outputs/colmap/sparse/0
```

The CLI prints its final machine-readable receipt as JSON. The Slurm wrapper
should send stdout/stderr to a log below `DATA_ROOT/logs`; no log belongs in the
code repository. The per-attempt `status.json` is also safe to inspect while a
run is active.

## What the algorithm does

### 1. Freeze the experiment identity

`discover_images()` recursively selects ordinary JPG, JPEG, PNG, BMP, TIFF, or
TIF files. It ignores file symlinks and requires at least two views.

`image_manifest()` reads every image in chunks and stores its relative path,
byte count, and SHA-256. Device/inode/size/mtime are checked immediately before
and after hashing, so a file being copied or edited at that moment is rejected.
The portable cache identity uses content hashes, not absolute Mac paths or
modification times. The images are hashed again after reconstruction; a result
is never published when an input changed during the run.

The identity also covers:

- the exact engine source SHA-256 and recipe version;
- PyCOLMAP version and CUDA-build flag;
- selected CPU/CUDA device class;
- camera model and camera-grouping mode;
- matcher and all controlled matching constants;
- resize/feature/thread limits and random seed;
- all relative image names, sizes, and hashes.

Changing any of those requires a new experiment name. That rule prevents an
old result from being silently presented as a new configuration.

### 2. Infer cameras and extract SIFT features

`pycolmap.extract_features()` imports the selected images into a new COLMAP
SQLite database and extracts SIFT keypoints and 128-dimensional descriptors.
The controlled options are:

- maximum image dimension: 3200 by default;
- maximum keypoints: 16,384 per image;
- SIFT `first_octave=0`, so this fresh recipe does not upsample the image;
- explicit CPU/CUDA device and worker count.

For the default `SIMPLE_RADIAL` camera, the unknown intrinsics are focal length
`f`, principal point `(cx, cy)`, and one radial-distortion value `k`. For a
camera-space point, normalized coordinates are `x=X/Z`, `y=Y/Z`, with
`r²=x²+y²`; the simplified projection is

```text
u = f (1 + k r²) x + cx
v = f (1 + k r²) y + cy
```

PyCOLMAP can initialize focal length from image metadata and later refines the
camera parameters during bundle adjustment.

Camera grouping matters:

- `auto`: COLMAP uses metadata to decide camera sharing;
- `single`: all images share one intrinsic calibration;
- `per_folder`: images in the same immediate capture folder share calibration.

The prepared light dataset uses different phone/lens or video capture groups,
so `per_folder` preserves that intended grouping. `single` should be used only
when every view truly comes from the same unchanged camera calibration.

### 3. Match features and verify geometry

The matcher proposes descriptor correspondences and COLMAP performs two-view
geometric verification, rejecting correspondences inconsistent with a robust
epipolar model.

- `exhaustive` considers every unordered pair: `N(N-1)/2`. For 125 images this
  is 7,750 pairs. It is the safer default for a small unordered photo set.
- `sequential` considers nearby names with overlap 10. Loop detection is
  explicitly off, so this mode is appropriate only when filenames preserve
  capture order and neighbouring frames overlap.

Guided matching is explicitly off in this fresh recipe. This keeps its meaning
distinct from historical E10, whose verified E10 database attempted guided
matching across all 7,750 light-image pairs. A CUDA device accelerates SIFT
feature extraction and matching; this code does not claim that every mapper or
bundle-adjustment operation runs on the GPU.

### 4. Incrementally estimate poses and triangulate points

`pycolmap.incremental_mapping()` performs the standard incremental SfM loop:

1. choose a geometrically strong initial image pair;
2. recover their relative poses and triangulate seed 3D points;
3. register another image from its 2D–3D correspondences;
4. triangulate new tracks seen by multiple cameras;
5. run local/global bundle adjustment and remove inconsistent geometry;
6. repeat until no additional image can be registered.

Conceptually, bundle adjustment jointly refines camera parameters `C_i` and 3D
points `X_j` to minimize robust reprojection error:

```text
min over {C_i, X_j}  Σ_(i,j observed)  ρ( || x_ij - project(C_i, X_j) ||² )
```

The seed is fixed to zero through `pycolmap.set_random_seed(0)` and the
incremental pipeline option. Scale is still arbitrary: ordinary monocular SfM
recovers geometry only up to a similarity transform unless an external metric
constraint is supplied.

Disconnected view graphs can produce several component folders. The engine
loads every component and selects the one with the most registered images,
breaking a tie by the number of 3D points. It does not delete the smaller
components from the attempt record.

### 5. Save colours and prove the result reloads

The mapper is asked to extract point colours. Before publication, the chosen
model is written to COLMAP binary files and then loaded again from disk. The
reload checks:

- PyCOLMAP's internal reconstruction consistency;
- at least two registered images and at least one 3D point;
- unique registered names, all drawn from the input manifest;
- every image references an existing camera;
- finite rotations, translations, camera parameters, point coordinates, and
  point reprojection errors;
- valid camera dimensions/parameter counts and RGB values in `[0,255]`;
- finite mean reprojection error and positive mean track length.

`export_PLY()` creates the interoperable coloured sparse point cloud. The PLY
validator checks the magic/header, exact vertex count, and uint8 red, green,
and blue properties. This is why the output is correctly described as coloured
3D points, not merely an unverified file with a `.ply` suffix.

## Crash safety, cache, and resume

Each experiment owns an OS `flock`; a second process cannot mutate the same
database or outputs concurrently. A run has this layout:

```text
experiments/<name>/
├── .run.lock
├── manifest.json
├── attempts/
│   ├── 0001/
│   │   ├── status.json
│   │   └── work/
│   │       ├── database.db
│   │       └── sparse_components/
│   └── 0002/                         # created only after a retry
├── outputs/
│   ├── colmap/sparse/0/{cameras,images,points3D}.bin
│   └── sparse_colored.ply
└── receipt.json
```

An interrupted COLMAP database is not a trustworthy checkpoint. Therefore
`--resume` means **retain the failed attempt and start a clean numbered
attempt**. It never appends to a partly written SQLite database or component
folder. `--no-resume` instead rejects an incomplete existing experiment.

Outputs are first produced under that attempt. Only after model reload, PLY
validation, and the second input-hash pass does one same-filesystem rename
publish the complete `outputs/` tree. `receipt.json` is written atomically. If
termination happens after the output rename but before the receipt, the next
invocation validates those outputs and reconstructs the missing receipt without
rerunning SfM.

A normal cache hit never invokes extraction, matching, or mapping. It reloads
the fixed expected output paths, rechecks the registered names/key counts and
PLY, and compares them with the receipt. It does not trust output paths merely
because they appear in JSON.

## CUDA checks and exact dependency contract

The Linux environment pins `pycolmap-cuda12==4.2.0`. The official COLMAP Python
documentation distinguishes the ordinary `pycolmap` wheel from the Linux CUDA
wheel. This engine is deliberately strict about the validated 4.2.0 API.

`pycolmap.has_cuda` proves only that CUDA support was compiled in. Before an
actual CUDA stage, the engine additionally requires:

1. `get_num_cuda_devices()` reports exactly one visible device;
2. the permanent cluster data root is being used from an active Slurm job;
3. `nvidia-smi` succeeds and its GPU UUID, model, driver, and memory are saved;
4. Slurm/CUDA visibility variables are captured in the receipt.

Exactly one visible GPU prevents the method from accidentally consuming GPUs
belonging to another project. `--device cuda` never silently falls back to CPU.
`--device auto` chooses CUDA only when the build and runtime report a device;
`--device cpu` keeps the original Mac/headless CPU path available.

## Source-code map

- `run_headless.py`: minimal repository entry point; imports no GUI.
- `sfm_engine/cli.py`: parses `run` and `validate`, builds `SfMConfig`, prints
  JSON, and lets failures produce a nonzero process exit.
- `SfMConfig.validate()`: validates names, numeric bounds, methods, and input
  containment before PyCOLMAP is imported.
- `discover_images()` / `image_manifest()`: immutable input discovery and
  portable content identity.
- `resolve_device()` / `cuda_runtime_identity()`: no-silent-fallback device
  choice and runtime GPU provenance.
- `reconstruction_metrics()`: geometric and colour invariants.
- `largest_valid_model()`: component reload/selection.
- `_call_matcher()`: the two supported pairing policies and their exact 4.2
  keyword dictionaries.
- `_experiment_lock()`: one writer per experiment, released automatically when
  the process exits or crashes.
- `run_reconstruction()`: stage state machine, clean retry, atomic publication,
  cache validation, and receipt generation.

## Tests

The fresh-engine tests use the Python standard library and a recording fake of
the PyCOLMAP 4.2 surface. The complete current directory also includes fixed-pose
tests requiring installed NumPy/PyCOLMAP and historical-browser tests. None
downloads packages or runs GPU reconstruction:

```bash
cd /home/anas.khan/cv802_project/project1/ass1/sfm
TMPDIR=/l/users/anas.khan/cv_802_ass1/sfm/tmp \
XDG_CACHE_HOME=/l/users/anas.khan/cv_802_ass1/sfm/cache/xdg \
PIP_CACHE_DIR=/l/users/anas.khan/cv_802_ass1/sfm/cache/pip \
PYTHONPYCACHEPREFIX=/l/users/anas.khan/cv_802_ass1/sfm/cache/python \
PYTHONNOUSERSITE=1 \
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python -B \
  -m unittest discover -s tests -v
```

The suite covers path escape, symlink escape, recursive image discovery, exact
content hashing, version/device checks, CUDA identity, both matcher dictionaries,
full save/reload/cache flow, clean attempt retry, `--no-resume`, post-rename
recovery, tampered receipts, missing PLYs, input mutation, lock exclusivity, and
invalid reconstruction/camera geometry. It also discovers the fixed-pose and
historical-browser tests now present under `sfm/tests`; report the count printed
by the command instead of relying on a hard-coded total in this document.

The separate preserved `assignment1/tests` backend suite also passed in this
environment after adding Pillow 11.3.0; it mocks Open3D. See
[`DEPENDENCIES.md`](DEPENDENCIES.md) for the complete dependency audit, both
test commands, exact verified counts and durable evidence.

## Viva distinctions and limitations

- **SfM versus MVS:** SfM estimates camera poses and a sparse feature-track
  cloud. MVS consumes calibrated cameras to estimate dense depth and fuse a
  dense cloud. This module performs SfM only.
- **Fresh versus fixed poses:** this CLI estimates cameras from scratch. The
  preserved E3–E10 code instead inherits E1 poses, rescales intrinsics, changes
  features/pairs, and retriangulates. Those are not independent camera
  estimates and are not silently relabelled here.
- **Sparse versus mesh:** a coloured PLY is a set of vertices; it has no faces,
  surface, watertightness, or texture atlas.
- **Reprojection error versus ground-truth accuracy:** a low reprojection error
  measures consistency with the same observations. It does not establish true
  metric scale or human-body accuracy.
- **Dynamic subject limitation:** SfM assumes a static scene. Human motion,
  plain clothing, blur, repeated texture, and occlusion can reduce matches or
  create inconsistent geometry.
- **90% mask meaning:** the historical 90% result means foreground agreement
  across usable camera projections, not 90% accuracy and not 90% point
  retention. This fresh engine does not apply that historical cleanup.
- **Why not reuse a partial database:** SQLite may be structurally readable
  while containing only some features/matches. Mixing it with a restarted stage
  makes provenance ambiguous; numbered clean attempts are safer and auditable.
- **Why use content hashes:** server paths and mtimes differ from the Mac. A
  relative-name/size/SHA-256 identity survives a verified copy while still
  detecting changed pixels.

Official implementation references:

- [COLMAP PyCOLMAP documentation](https://colmap.github.io/pycolmap/pycolmap.html)
- [COLMAP Python bindings and CUDA-wheel notes](https://github.com/colmap/colmap/blob/4.2.0/python/README.md)
- [COLMAP feature matching tutorial](https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification)
