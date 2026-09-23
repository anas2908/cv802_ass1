# VGGSfM implementation notes for the viva

## One-sentence description

This method independently estimates cameras and colored 3D points from raw
multi-view images by running the pinned official VGGSfM v2 neural SfM engine;
our integration makes that engine headless, reproducible, storage-safe and
comparable with the other assignment methods.

## End-to-end data flow

1. `scripts/prepare_input.py` copies the raw image set into
   `DATA_ROOT/vggsfm/inputs/<dataset>/images`. It hashes source and destination
   with SHA-256 before atomically publishing the dataset. It does not delete the
   source.
2. `InferenceProfile.from_json` accepts only reviewed fields. An unknown key
   such as `load_gt` is an error, so a profile cannot smuggle ground-truth poses
   or turn on a server UI.
3. `VGGSfMEngine._prepare_request` hashes every input image/mask and combines
   those hashes with the full profile, official Git SHA and model identity. The
   canonical JSON SHA-256 is the request fingerprint.
4. `_request_state` binds a run ID to exactly one fingerprint. A verified
   complete result is reused; an incomplete attempt needs explicit `--resume`;
   a different request needs a new run ID.
5. `_stage_inputs` creates a clean attempt-local `scene/images` containing
   symlinks only to already-copied inputs inside the VGGSfM data root. The real
   capture remains grouped in camera subdirectories, so each official staging
   filename is `six_digit_sequence__first_16_hex(SHA256(relative_path))__basename`.
   The sequence prefix preserves capture order when the official loader sorts
   filenames, while the digest prevents collisions between camera folders.
   `request.json` records the complete reversible mapping
   used later to match cameras for post-hoc alignment. The stage does not copy
   `sparse`, a COLMAP database, cameras, or any E10 artifact.
6. `_preflight_command` imports the environment's PyTorch and asserts CUDA.
   `SubprocessExecutor` invokes argv directly with `shell=False` and tees merged
   stdout/stderr to the attempt log.
7. The official `demo.py` reads the images, downloads the official checkpoint
   to redirected Hugging Face/Torch cache if necessary, reconstructs the scene,
   and writes a `pycolmap.Reconstruction` under `scene/sparse`.
8. `discover_model` accepts either `sparse/*.bin` (the documented VGGSfM
   layout) or `sparse/0/*.bin` (common COLMAP layout), but fails if output is
   missing or ambiguous.
9. `colmap.py` fully parses all three binary files. It checks camera model
   layouts, finite intrinsics, positive dimensions/focal lengths, finite
   normalized poses, unique IDs/names, camera references, file bounds and
   trailing bytes. It validates finite point geometry and reciprocal links
   between image observations and point tracks. It then streams the points to
   a binary little-endian PLY, preserving RGB and converting XYZ doubles to
   floats.
10. `_publish` hard-links the official COLMAP files when possible (copy fallback),
    creates the PLY and hashes every normalized file in a hidden staging
    directory. A final atomic rename publishes the output only after all checks
    pass.
11. The attempt receipt records command, timings, Slurm/GPU context, sampled
    process-tree RAM/VRAM and failure details. The output manifest records the
    same successful-inference resource summary plus geometry metrics, upstream
    identity, profile, checksums and the explicit no-external-pose claim.

The manifest also hashes the primary VGGSfM checkpoint and the auxiliary DINO
and ALIKED weights. `auxiliary_source_snapshots` hashes the exact cached DINO
Torch Hub source files. Upstream requests DINO's `main` branch, and that cache
does not contain Git metadata: the integration records this limitation rather
than inventing a DINO commit. The verified official VGGSfM checkout itself is
pinned to the commit in `constants.py`; all cached source and weights remain in
the method's data directory.

## What VGGSfM contributes

The official engine contains three learned/geometric blocks:

- The camera predictor provides an initial camera estimate from global image
  evidence. This removes the fragile incremental bootstrapping requirement of
  classic SfM.
- The tracker predicts consistent 2D feature trajectories from selected query
  pixels across views. Query frames are ranked, and missing/weakly visible
  frames can trigger complementary queries.
- The triangulator combines predicted cameras/tracks with robust preliminary
  geometry, triangulates 3D points, filters outliers and performs bundle
  adjustment through the geometric backend.

The loss/training of these networks is not executed in this assignment. This
is inference plus geometric optimization. The downloaded `vggsfm_v2_0_0.bin`
contains pretrained weights.

## Important configuration answers

`SIMPLE_RADIAL` estimates focal length and one radial distortion coefficient;
it is more suitable than an ideal pinhole for phone imagery. `shared_camera`
ties intrinsics across frames and is valid only for one camera without material
zoom/focal changes. Extrinsics remain per image. Our profile sets it to `False`
because the capture mixes iPhone 11/13, 13/14/26 mm and video groups; forcing a
single intrinsic calibration would violate the data acquisition model.

`query_frame_num=6` uses more anchor views than the upstream default of three,
helping broad coverage in the light-shirt capture. `max_query_pts=2048` is the
per-query feature budget. Increasing either can improve difficult coverage but
costs memory/time; it does not guarantee accuracy.

`fine_tracking=True` refines coarse tracks at higher spatial precision.
`mixed_precision=fp16` reduces activation memory on the A100 while retaining
the official supported path. The official geometry/BA code promotes the
appropriate portions to float32.

`extra_pt_pixel_interval=10` samples an image grid every ten pixels. Accepted
extra tracks are triangulated and concatenated because
`concat_extra_points=True`. These points are colored but are not jointly bundle
adjusted with the original sparse set. `extra_by_neighbor=16` limits each grid's
tracking to 16 temporally adjacent frames instead of all 125. At 1024 px and a
10-pixel interval this keeps each tracking call near the upstream A100 chunk
budget while preserving enough views for robust triangulation. For a strictly
BA-optimized sparse model, use `--sparse-only` and a distinct run ID.

All visualization flags are false. This is why no X display, OpenGL window,
Visdom server or Gradio service is needed. PyTorch3D is optional upstream for
Visdom and is skipped in our headless environment.

The environment uses `opencv-python-headless`, which provides the working
`cv2` module without Qt/X GUI libraries. LightGlue 0.0 declares
`opencv-python` by distribution name. Python package metadata has no standard
"alternative provider" mechanism, so `pip check` cannot know the headless
distribution satisfies the same import and prints one warning. The environment
audit proves `cv2 4.10.0` imports from the fixed data-root environment and that
this is the only `pip check` finding. Installing both distributions would be a
bad fix because they can overwrite the same `cv2` package files.

## Binary `points3D.bin` layout

All integers are little-endian. The file begins with `uint64 number_of_points`.
Each point then stores:

```text
uint64 point_id
float64 x, y, z
uint8 red, green, blue
float64 reprojection_error
uint64 track_length
track_length × (uint32 image_id, uint32 point2D_index)
```

Our PLY contains a declared vertex count followed by
`float32 x,y,z + uint8 r,g,b`. Converting XYZ from float64 to the conventional
PLY float32 representation changes numerical precision slightly but not the
coordinate frame, scale, ordering or colors. The original full-precision
COLMAP binary remains available beside the PLY.

VGGSfM's optional concatenated additional points have empty tracks and were not
part of bundle adjustment. The manifest therefore reports tracked and
trackless point counts separately. Reprojection error and track-length quality
means use tracked points only. Explicitly named `raw_*` fields retain the former
all-point arithmetic for receipt compatibility, but are not BA quality metrics.

## Resource measurements

The wrapper samples the official inference process and all descendants every
two seconds. Each sample records summed `/proc` resident memory and NVIDIA's
per-process GPU memory in the attempt-local
`resources-<pid>.jsonl`. `resource_metrics` in both the successful receipt and
published manifest contains the sample path/count and observed peak
`peak_process_tree_rss_kib` and `peak_process_tree_gpu_mib`. The CUDA preflight
is sampled separately; its short-lived file is not the inference result.

These values are defensible sampled peaks, not exact allocator high-water
marks. Sampling can miss short spikes, summed RSS may double-count shared
pages, and NVIDIA accounting is restricted to the inference process tree so
unrelated users' GPU memory is deliberately excluded.

## Correct interpretation of results

- `point_count` is density, not accuracy.
- `trackless_point_count` is official extra-point density, not evidence of BA
  support; use the tracked-only error and track-length means for quality.
- A stored reprojection error of `-1` is an uncomputed PyCOLMAP/VGGSfM
  sentinel, not a negative pixel error. The converter excludes it, reports an
  unavailable count, and returns `null` for the tracked error mean if no
  measured error exists. The explicitly labelled legacy raw mean is provenance
  only.
- Registered-image count measures coverage, not pose correctness.
- Reprojection error measures self-consistency with fitted observations; a low
  value can coexist with wrong geometry in degenerate captures.
- Monocular reconstruction has no metric scale without an external measurement.
- More points from the additional grid are not equivalent to dense MVS depth.
- Visual comparisons across SfM/MVS/VGGSfM require post-hoc similarity
  alignment. Feeding another method's poses into VGGSfM would invalidate the
  independence of the comparison.

## Why there is a derived COLMAP review export

The official loader pads every image to a square and resizes that square to
`img_size=1024`. For original width `w`, height `h` and `L=max(w,h)`, its crop
origin is `left=(w-L)//2`, `top=(h-L)//2`. The runner rescales each camera back
to `w x h`. Its optional matching observation conversion is:

```text
top_left_abs = abs(float32([left, top] / L * 1024))
resize_ratio = float32(L / 1024)
xy_original = (xy_padded - top_left_abs) * resize_ratio
```

Our official run resolved that optional switch to false, as recorded verbatim
in `official.log`; therefore the published camera and `point2D` coordinate
frames differ. `scripts/review_export.py` reproduces the exact upstream
float32 formula after inference in a new immutable output. It projects each
tracked XYZ through `image.cam_from_world` and `camera.img_from_cam`, computes
Euclidean pixel residuals, replaces each tracked point's stored error with the
mean of its actual observations, and uses `-1` for trackless official extra
points. This fixes interoperability metadata; it does not refine a camera or
3D point and does not improve the reconstruction itself.

The audit compares two independently loaded reconstructions field by field.
Only `point2D.xy` and `point3D.error` may differ. Camera IDs/models/sizes/
intrinsics, image IDs/names/camera references/poses, point IDs/XYZ/RGB and
ordered tracks must be identical. The source PLY is copied byte-for-byte. A
few exact converted observations can lie outside the original rectangle
because VGGSfM retained tracks in square padding; they are honestly counted,
not clamped or removed. This preserves the official tracks and formula.

Useful validation includes registered-image ratio, point count, reprojection
error, track length, visual outlier inspection, camera trajectory plausibility,
and (only after reconstruction) aligned geometric distances to a reference.

## Failure and recovery behavior

A failed attempt is never deleted or reused silently. Its `receipt.json` becomes
`status: failed`, while the log and any official partial output remain. Running
without `--resume` stops. Running with `--resume` creates the next attempt
number and restages the same checksum-bound input. Published output is immutable:
if its manifest/file checksums fail, the engine raises an error instead of
claiming a cache hit.

This is "safe resume," not checkpoint-level neural continuation. Official
VGGSfM does not expose a general mid-scene resume contract; a new attempt reruns
inference while preserving forensic evidence from the previous attempt.

## Likely viva questions

**Did you implement VGGSfM itself?** No. We integrate the official pinned model
and can identify exactly which code is upstream versus ours. Reimplementing it
would be both unnecessary and hard to validate.

**Is it training?** No. The assignment run loads pretrained weights and performs
inference followed by robust geometry and bundle adjustment.

**Why keep COLMAP and PLY?** COLMAP retains cameras, image poses, tracks, errors
and full XYZ precision; PLY gives MeshLab a simple colored point cloud. PLY
alone loses observations and camera parameters.

**Why can you compare it with E10 but not use E10?** A fair comparison can align
outputs after both are independently estimated. Initialization from E10 would
leak baseline information into VGGSfM and destroy method independence.

**What makes the run reproducible?** Pinned source SHAs, model identity,
captured package inventory, fixed seed, checksummed inputs/config, immutable
attempts, logged argv/environment, and checksummed outputs. GPU kernels may
still have small platform-dependent numerical differences.

**Why force a new run ID after changing a knob?** The run ID is a human label;
the fingerprint is the true experiment identity. Refusing mismatches prevents
an old result from being mislabeled as a new configuration.

**Where can large files appear?** Only below
`/l/users/anas.khan/cv_802_ass1/vggsfm`. The environment redirects Conda, pip,
Hugging Face, Torch, CUDA, Triton, Python bytecode, compiler extensions,
matplotlib, XDG data/config and temporary directories there.
