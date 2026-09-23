# MVS viva guide: algorithm, implementation and defensible claims

## One-sentence description

This method takes fixed calibrated cameras from E10, estimates a dense depth and
normal map for each undistorted reference image with CUDA PatchMatch Stereo,
checks those estimates across neighbouring views, and fuses consistent samples
into one colored 3D point cloud.

## SfM versus MVS

Structure from Motion (SfM) estimates cameras and a sparse set of 3D landmarks
from feature correspondences. Multi-View Stereo (MVS) assumes those cameras are
already known and estimates far more surface samples by matching image patches.
In this implementation:

- E10 supplies intrinsics, distortion, rotations, translations, registered image
  names and sparse points.
- MVS does not run feature extraction, matching, essential-matrix estimation,
  incremental mapping or bundle adjustment.
- The sparse points help COLMAP choose depth ranges and overlapping source views.
- The new output is dense geometry, not evidence that camera estimation was
  repeated.

E3 through E10 inherited E1 poses, rescaled intrinsics and retriangulated. That
history must be mentioned when interpreting an E10-initialized dense result.

## Camera geometry used by the engine

COLMAP stores an image pose as a unit quaternion `qvec` and translation `tvec`
for the world-to-camera transform:

```text
X_camera = R(qvec) X_world + tvec
```

Perspective projection first normalizes a camera-space point:

```text
u_normalized = X_camera.x / X_camera.z
v_normalized = X_camera.y / X_camera.z
```

The camera model then applies lens distortion and intrinsics. E10 contains five
`SIMPLE_RADIAL` calibration groups. `image_undistorter` resamples the original
distorted pixels into pinhole images and writes camera parameters consistent
with those resampled pixels. PatchMatch consumes that new workspace; it does not
incorrectly apply a distorted camera to an undistorted image.

The controller parses both COLMAP binary and text formats before running. For a
binary `images.bin`, it reads the image ID, seven doubles (`qvec`, `tvec`), camera
ID, null-terminated UTF-8 image name, point count, and then skips the fixed-size
2D observation records. It checks unique IDs/names, finite poses, quaternion norm
near one, valid camera references, positive dimensions/focal length and actual
image existence. Path traversal in an image name is rejected.

## What PatchMatch Stereo does

For each reference image, COLMAP selects nearby/overlapping source images using
the sparse reconstruction. Each reference pixel needs a depth and surface-normal
hypothesis. PatchMatch repeatedly improves hypotheses using:

1. **Initialization**: create candidate depth/normal hypotheses within a range
   inferred from the sparse model.
2. **Propagation**: good hypotheses found at neighbouring pixels are proposed at
   the current pixel; surfaces are often locally coherent.
3. **Random refinement**: perturb depth and normal to escape a poor local choice.
4. **Photometric scoring**: warp the hypothesized patch into source views and
   compare appearance, commonly with normalized cross-correlation-like costs.
5. **Filtering**: reject pixels that lack enough agreement across source views.

The work is massively parallel over pixels and hypotheses, which is why COLMAP
PatchMatch is the GPU-intensive step. The 1600-pixel cap reduces the number of
pixels roughly quadratically compared with native 3024x4032 stills, while the
10-source cap bounds per-reference matching work and memory. `window_step=1`
samples every pixel in a window; `num_iterations=5` and `num_samples=15` retain
the COLMAP-quality default style rather than using an aggressive speed-only
preset.

## Photometric and geometric consistency

Photometric consistency asks whether warped patches look similar. Repeated
texture, plain clothing, illumination changes and motion can still yield wrong
depth. With `geom_consistency=1`, COLMAP first estimates photometric depth maps,
then evaluates a hypothesis using depth/normal predictions in the other views.
A reference point projected into a source view should agree with that source's
depth and project back consistently.

Geometric consistency costs more time and VRAM but usually removes unstable
depths. The command passes numeric `1`, not the word `true`, because some COLMAP
CLI versions interpreted textual booleans ambiguously. Fusion consequently uses
`--input_type geometric`; choosing geometric fusion without generating geometric
maps would be an implementation error.

## Source-view configuration

After undistortion, COLMAP creates `stereo/patch-match.cfg` as alternating
reference-image and source-selection lines. The controller retains every
reference image but replaces each source line with:

```text
__auto__, 10
```

This asks COLMAP to choose the ten images with greatest useful overlap per
reference. It is not “use only ten input images”; all 125 images remain reference
views. The engine verifies that the config contains exactly 125 reference/source
pairs before accepting undistortion.

## Stereo fusion

PatchMatch produces one depth and normal estimate per accepted reference pixel,
so the same physical surface can appear many times. `stereo_fusion` back-projects
depth pixels to 3D and merges samples supported by compatible observations.
Configured checks include:

- at least five consistent pixels/observations (`min_num_pixels=5`);
- reprojection error at most two pixels;
- relative depth disagreement at most `0.01`;
- normal disagreement at most ten degrees;
- up to fifty images checked for consistency.

Tighter thresholds can remove real thin structures; looser thresholds retain
floaters. These are reconstruction hyperparameters, not accuracy measurements.

The fused point color comes from the source images during COLMAP fusion. The
validator parses the PLY header, requires `x y z red green blue`, scans every
vertex for finite coordinates and legal color values, records the 3D bounds and
computes SHA-256. “Colored point cloud” is therefore verified, not inferred from
the `.ply` filename.

## Foreground masking

The historical Apple Vision masks describe the original distorted pixel grid.
The implementation applies them to those original images, then gives the masked
images to `image_undistorter`. Applying an original mask directly to an already
undistorted image would misalign at the edges because lens distortion moves
pixels.

Values at least 128 are treated as person foreground. A three-pixel native-grid
dilation tolerates small matte errors, and background pixels are set to black.
The implementation rejects masks with changed hashes, mismatched dimensions or
implausibly empty/full foreground. The mask polarity is explicit: high values
mean keep, not exclude.

This preprocessing encourages dense matching on the person but does not make a
hard proof that every fused point is foreground; black regions and silhouettes
can still produce artifacts. It also is not E10's sparse 90% consensus filter.
That filter counted foreground agreement over usable feature-track views. Dense
points lack those tracks, so the same denominator and visibility cannot be
silently reused.

## Why the final cleaned E10 sparse model is acceptable

The final E10 model retains all calibrated images/cameras and 21,744 foreground-
consistent sparse points. It is a reasonable initialization for subject-focused
MVS because these points provide subject depth ranges and overlap evidence. A
possible limitation is that cleaning can reduce overlap evidence used for MVS
neighbour selection. If this causes poor neighbours, the scientifically honest
response is a separately named experiment initialized by a copied unfiltered
E10 model, with the change recorded—not mutation of the published E10 result.

## Code map

| File | Responsibility |
|---|---|
| `paths.py` | Canonical data-root containment and experiment-name validation |
| `config.py` | Strict schema, range checks, one-GPU rule, native/container backend |
| `colmap_model.py` | Dependency-free binary/text camera and registered-image audit |
| `staging.py` | Stable-source copy, symlink resolution, SHA-256 and mask-manifest rebase |
| `masking.py` | Distorted-grid foreground preprocessing without modifying inputs |
| `commands.py` | Shell-free exact CLI/PyCOLMAP-worker argv construction and source-view cap |
| `pycolmap_backend.py` | Explicit PyCOLMAP 4.2 CUDA fallback and option translation |
| `validation.py` | Allocation/CUDA/container preflight, input checks, stage checks, PLY scan |
| `runner.py` | Stage order, logging, receipts, resume, metrics and recoverable overwrite |
| `cli.py` | User-facing `stage-inputs`, `plan`, `validate`, `run`, `status` commands |

## Reproducibility and provenance

An experiment is identified by more than its folder name. The controller stores:

- SHA-256, byte size and stable source timestamps for copied inputs;
- original and rebased mask manifests;
- exact JSON config and its canonical digest;
- exact argv for each COLMAP subprocess;
- Slurm job/node, driver, GPU/VRAM, visible device and COLMAP help header;
- registry OCI digest, local SIF SHA-256 and Python package versions;
- per-stage start/end status, elapsed time, output identity and sampled resource
  peaks;
- final point count, finite-coordinate result, RGB presence, bounds and PLY hash.

The source files stay small and GitHub-ready; all runtime evidence is generated
under the method data root.

## Resume logic in detail

Each successful stage gets a receipt containing a command digest and an output
digest. During `--resume`:

1. The previous run's config digest must equal the current config digest.
2. A completed stage's command digest must match.
3. Its current output is revalidated and rehashed to the saved identity.
4. Only then is the stage skipped.

PatchMatch has special partial-resume behavior documented by COLMAP: complete
per-view maps are skipped. Undistortion, fusion and meshing are not trusted as
incremental here, so partial products are atomically moved to `archive/` before
retry. A changed completed artifact causes a fail-closed error instead of being
silently reused.

## Failure modes and responses

| Symptom | Likely cause | Correct response |
|---|---|---|
| `SLURM_JOB_ID is unset` | Running on login node | Enter the one allocated compute job |
| PatchMatch help probe fails | COLMAP lacks CUDA/MVS or container cannot see GPU | Verify `singularity exec --nv`, host driver and SIF |
| CUDA out of memory | Resolution/sources/geometric pass too costly | New experiment: reduce 1600 or sources 10; do not relabel settings |
| Few/no depth maps | Weak overlap, depth range or plain/moving surface | Inspect neighbour config and copied model; pilot unmasked/unfiltered separately |
| Fusion has few points | Consistency thresholds too strict or depths poor | Inspect depth completion first; change thresholds only in a new config |
| Many floaters | Ambiguous texture/background survives matching | Keep geometric consistency; tune fusion or masking as a new experiment |
| Masked person has cut edges | Matte threshold/dilation too strict | Inspect masks and create a new experiment with documented values |
| Receipt mismatch | Artifact/config changed | Investigate provenance; use new name or recoverable `--overwrite` |

## Quality statements that are safe after receipts confirm them

The following forms are safe only after substituting values from the completed
run's receipts and final-validation manifest:

- “The output contains N validated finite colored vertices.”
- “All N validated calibrated views were passed to undistortion/PatchMatch.”
- “The run used an A100 and the recorded peak sampled VRAM was X MiB.”
- “Geometric consistency and the listed fusion thresholds were enabled.”
- “The mask preprocessing retained foreground in distorted coordinates.”

Unsafe statements without ground truth:

- “The reconstruction is 90% accurate.” The 90% number was mask agreement,
  not metric accuracy.
- “More points means better geometry.” Density can include more outliers.
- “The MVS camera poses were estimated independently.” They came from E10.
- “The sparse consensus filter was applied unchanged to dense points.” It was
  not; the preprocessing has different semantics.
- “The run is GPU accelerated” merely because a CUDA package is installed. The
  actual runtime/container/GPU evidence must be present.

## Short viva questions

**Why undistort first?** Patch matching assumes a consistent pinhole projection.
Undistortion resamples pixels and updates intrinsics together, so epipolar/
multi-view warps use the correct geometry.

**Why not use the sparse cloud as the final output?** Sparse points occur only at
repeatable keypoints. MVS estimates surface samples at many more pixels after
cameras are known.

**Why does an A100 help?** PatchMatch evaluates many independent pixel/depth/
normal hypotheses and source-view warps; these operations parallelize well on a
GPU. Fusion is more CPU/I/O oriented.

**Why cap image size and source views?** Pixel count grows approximately with
the square of linear resolution, and matching work grows with source-view count.
The caps control VRAM, RAM and runtime for a bounded pilot.

**What is the difference between a depth map and a point cloud?** A depth map is
camera-centric: each valid pixel stores distance for one view. Fusion transforms
consistent depths from many cameras into a shared world-coordinate point cloud.

**Why is meshing optional?** Meshing adds a surface prior and can fill holes or
smooth detail. The assignment's primary MVS evidence is the dense colored point
cloud; a mesh must be labelled as a derived product.

**What makes the run headless?** Only COLMAP CLI subcommands or the equivalent
PyCOLMAP dense APIs are called. There is no Qt/OpenGL viewer and no X display.
CUDA is passed into Singularity with `--nv`, or verified by the pinned CUDA
wheel's runtime probe.

**Does the PyCOLMAP fallback change the MVS algorithm?** No. The pinned
`pycolmap-cuda12==4.2.0` wheel binds the same COLMAP 4.2 undistortion,
PatchMatch Stereo and fusion implementations. The controller maps every recorded
setting into the corresponding typed option object and requires CUDA at runtime.
The backend changes the invocation boundary because this cluster could not mount
the otherwise valid SquashFS image; it does not replace PatchMatch with another
method or silently run a CPU approximation.

**How do you know the copy is independent?** Staging copies only model-referenced
images and required calibration/masks, dereferences symlinks, verifies hashes,
and every configured dense path is contained in the MVS root. The container
backend additionally mounts only that root. Dense writes cannot flow back to SfM.
