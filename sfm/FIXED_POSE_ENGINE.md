# Portable fixed-pose quality refinement

`run_fixed_pose.py` closes a different requirement from `run_headless.py`.
The fresh engine estimates camera poses. This command reuses historical E1
poses, the quality images, retained affine/domain-size-pooled SIFT features,
and already geometrically verified pairs, then triangulates new colored sparse
points on Linux or macOS without Open3D or a display. It does not rerun matching.

The original `assignment1/modules/colmap/api.py`, `estimate_cameras.py`, Mac
launcher, and historical E1–E10 algorithms are unchanged. A new result is called
E12, not a replacement for E3 or E10. The sample configuration uses E3's
published light-shirt quality database, not an E10 working/checkpoint database.

## Storage and invocation

Source and this configuration are below
`/home/anas.khan/cv802_project/project1/ass1/sfm`. All data, private databases,
models, generated manifests, logs and outputs are below
`/l/users/anas.khan/cv_802_ass1/sfm`. Relative paths inside the config resolve
against that SfM data directory. `CV802_DATA_ROOT` can explicitly override the
parent data root for another machine; there is no home-storage fallback.

The already installed Linux environment is:

`/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python`

It contains pinned PyCOLMAP 4.2.0 and NumPy. Its CUDA-capable build also supports
this CPU-only triangulation. No GPU allocation is required for the algorithm,
but reconstruction on this cluster requires a Slurm compute allocation.
Use an existing approved job; this command never requests an allocation.
Run the following inside that compute job:

```sh
export CV802_DATA_ROOT=/l/users/anas.khan/cv_802_ass1
export TMPDIR="$CV802_DATA_ROOT/sfm/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
cd /home/anas.khan/cv802_project/project1/ass1/sfm
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python \
  run_fixed_pose.py --config configs/fixed_pose_light_e3.json --dry-run
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python \
  run_fixed_pose.py --config configs/fixed_pose_light_e3.json
```

Dry-run hashes and validates the full input, including decoded image dimensions
and SQLite arrays, but creates no experiment. It is a real validation rather
than a printout of intended paths. The execution creates the configured
experiment ID exactly once. A completed identical experiment is loaded and
validated without triangulation or writes; it returns `cached_reuse: true`.
Changed identities, corrupted outputs, and an incomplete/failed directory are
rejected. To deliberately recompute, copy the small config and give it a new
experiment name. Failed artifacts stay available with `failure.json` for diagnosis.

The sample creates:

```text
DATA_ROOT/sfm/experiments/E12_linux_fixed_pose_light_e3_v1/
  manifest.json                  frozen config/input hashes and scientific plan
  database.db                    private copy used by COLMAP
  baseline_scaled/                original poses, intrinsics scaled for images
  outputs/colmap/sparse/0/        retriangulated COLMAP model
  outputs/sparse_colored.ply      colored sparse points
  receipt.json                   success only after all validations pass
```

Images are read from the existing physical SfM data copy. They are neither
duplicated for this same-method experiment nor linked across method roots.

## What each engine function proves

`FixedPoseConfig.validate` rejects paths outside the configured SfM data root,
symlinks, nonexistent inputs, `.working` databases, and IDs that reuse E1–E10
labels. The frozen dataclass and content identity hash describe one run.

`check_quiet_database` rejects nonempty WAL/journal files and visible same-user
writable database handles on Linux. Empty transferred WAL and stale SHM files
are left untouched. `readonly_database` uses SQLite `mode=ro&immutable=1` and
`query_only`; this avoids SQLite changing source sidecars. It is deliberately
for **published inactive input**. It is not a cluster-wide distributed lock;
concurrent writing from another host is unsupported. Pre/post full-file hashes
detect changes and prevent a success receipt if any source changes.

`frozen_inputs` fingerprints every input image, source database, feature
checkpoint, feature provenance JSON, and the complete COLMAP model file set.
The model includes `rigs` and `frames` when present, not just the old three-file
COLMAP layout. Each SHA-256 read also checks size, inode and modification time
before/after. Original Mac path strings in provenance remain unchanged.

`validate_database` checks SQLite integrity and image/camera references against
the registered baseline. All input image names must equal the baseline set.
Images sharing a camera must have identical dimensions. The baseline intrinsics
are scaled with `Camera.rescale`, exactly as in the historical quality path,
and must agree with those stored in the quality database within 1e-12 relative
and absolute tolerance. These scaled values become the frozen intrinsics.

It checks six-column affine keypoints (`x, y, a11, a12, a21, a22`), finite coordinates inside the image,
matching descriptor counts and 128-byte SIFT descriptors. It decodes COLMAP
pair IDs as `first * 2147483647 + second`, checks referenced images, and checks
every raw and verified match index against its image's feature count. At least
one verified nonempty pair is required. The feature provenance must explicitly
record both affine shape and domain-size pooling and match image names/sizes.

`table_digest` hashes ordered SQLite table contents independently of SQLite page
packing. Keypoint and descriptor tables in the matched database must exactly
match the feature checkpoint. This preserves the expensive native-resolution
affine/DSP work; it is not replaced with a different CUDA SIFT extractor.

`triangulation_options` sets both fixed-frame flags, disables intrinsic and rig
calibration refinement, disables GPU bundle adjustment, keeps the historical
3.5-pixel reprojection filter and 1.5-degree minimum triangulation angle, and
ignores two-view-only tracks. The call to `pycolmap.triangulate_points` uses
`clear_points=True` and `refine_intrinsics=False`. Thus previous 3D coordinates
are discarded and points are solved from existing 2D correspondences while the
camera solution stays fixed. The RGB values are extracted from the images.

`calibration` records every registered world-to-camera matrix and camera model,
size and parameters before triangulation. `verify_fixed_calibration` compares
these against the result and against the reloaded saved model; a difference
over 1e-12 is an error. `reconstruction_metrics` also checks valid finite
geometry, nonempty points and proper colors. The PLY vertex count must equal
the model point count. Hashes of features and matches in the private database
must still equal the original, and all historical files are rehashed before a
success receipt is written.

The source is never opened for writing. Only the new experiment's private
database can be upgraded or touched by COLMAP. No old PID files, matching
locks, guided locks or resume records are interpreted or modified.

## Tests and limits

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/l/users/anas.khan/cv_802_ass1/sfm/tmp \
  /l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python \
  -m unittest discover -s tests -p test_fixed_pose.py -v
```

The tests exercise valid feature accounting, camera-reference and intrinsic
disagreement, out-of-range match indices, feature-content tampering, pending
WAL and active writer rejection, read-only source preservation, fixed-pose
invariance, the actual installed PyCOLMAP options, and historical-ID/working-DB
rejection. The Linux `/proc` writer check is skipped on macOS.

## Verified Linux execution

E12 completed on 2026-09-19 using the existing job 264198, CPU-only step .16,
with four threads. The executed command was:

```sh
srun --jobid=264198 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --gres=none --export=ALL \
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 TMPDIR=/l/users/anas.khan/cv_802_ass1/sfm/tmp \
  PYTHONDONTWRITEBYTECODE=1 \
  /l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python \
  /home/anas.khan/cv802_project/project1/ass1/sfm/run_fixed_pose.py \
  --config /home/anas.khan/cv802_project/project1/ass1/sfm/configs/fixed_pose_light_e3.json
```

The job ID is an execution record, not a permanent usable allocation. For future
execution use the compute allocation approved for that session.

The success receipt records 125/125 registered images, five cameras, 41,675
colored sparse points, 1.3721696057 px mean reprojection error and 4.0993881224
mean track length. It reused 1,741,593 feature rows and 1,344 nonempty verified
pairs. The largest camera/pose difference was 6.6613381478e-16 (required
tolerance 1e-12). Engine execution after validation took 16.0903 seconds; this
does **not** include initial hashing/database/image validation. Features,
matches and frozen source contents were verified unchanged. The full E3 point
count was reproduced; this is not the E3 person-cropped preview or E10 cleanup.

The result PLY SHA-256 is
`7989f54870d02078a6bfd2d294eb2f3b84da84c64e2e27d7cc813e665e1a4e7b`.
Evidence lives under the SfM data root:

- `experiments/E12_linux_fixed_pose_light_e3_v1/receipt.json`
- `fixed-pose-E12.log`
- `fixed-pose-E12-cache-check.json`
- `fixed-pose-E12-cache-check.log`

An instrumented second invocation passed with zero triangulation calls and all
14 experiment files' sizes, modification times and SHA-256 hashes unchanged.
`tests/check_fixed_pose_cache.py` patches the actual `pycolmap.triangulate_points`
to fail if invoked, snapshots the real experiment, and then executes normal
cached validation. Its report is generated outside the experiment. Reproduce
that read-only check using the same CPU environment:

```sh
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python -B \
  tests/check_fixed_pose_cache.py \
  --config configs/fixed_pose_light_e3.json \
  --report /l/users/anas.khan/cv_802_ass1/sfm/fixed-pose-E12-cache-check-new.json
```

Choose a new report filename directly in `DATA_ROOT/sfm`; the checker rejects
existing reports and paths inside any source/experiment directory.

The fixed-pose checks are discovered as part of the complete portable SfM
suite. Use the measured suite count and log hash in the final
`run_validation_suite.py` receipt linked from
[`docs/EXPERIMENT_RESULTS.md`](../docs/EXPERIMENT_RESULTS.md), rather than a
hard-coded total that becomes stale when viewer or portability checks are added.

This is intentionally the **reuse and retriangulate** quality mode. It requires
a compatible finished feature/matching database. It does not provide a fresh
affine/DSP extraction or CPU/CUDA rematching implementation, and does not
claim to regenerate E1–E10, their person crops, or mask-consensus filtering.
Those remain separate preserved stages. A fixed-pose result's registered camera
count is inherited, not evidence of improved pose estimation. Sparse points on
a moving person or uniform clothing can still have missing or duplicated body
surfaces. The result requires visual inspection in addition to numerical checks.
