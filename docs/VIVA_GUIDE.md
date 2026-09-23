# CV802 Project 1 viva guide

This is the cross-method guide to the non-UI implementation. It explains what
the code does, what each method estimates, how results are validated, and which
claims are defensible. The method-specific guides remain the source for
file-by-file details:

- [headless sparse SfM](../sfm/HEADLESS_ENGINE.md)
- [read-only historical E1-E10 catalog and WebGL browser](../sfm/HISTORICAL_BROWSER.md)
- [dense COLMAP MVS](../mvs/docs/VIVA_GUIDE.md)
- [official VGGSfM integration](../vggsfm/VIVA_NOTES.md)
- [post-hoc VGGSfM-to-E10 alignment](../scripts/evaluation/VIVA_GUIDE.md)
- [storage relocation and verification](../scripts/storage/README.md)

The original Open3D GUI is intentionally outside this guide. It is a viewer and
launcher; it is not one of the reconstruction algorithms.

## The project in one answer

The project reconstructs the same light-shirt capture in three different ways:

1. **Classic sparse SfM** estimates camera intrinsics, camera poses and a sparse
   colored point cloud from local-feature correspondences.
2. **Pixel-wise multi-view stereo** keeps E10's calibrated cameras fixed, uses
   CUDA PatchMatch to estimate depth and normal hypotheses at image pixels, and
   fuses geometrically consistent samples into a dense colored point cloud.
3. **VGGSfM** independently estimates cameras and points from the copied raw
   images with the pinned official pretrained model and geometric refinement.

E10 is the verified historical reference. It registered all 125 light-shirt
images and contains 21,744 cleaned colored sparse points. It is not ground
truth. Fresh Linux/CUDA SfM E11 is now also verified complete: it independently
registered 125/125 images with five cameras and 10,204 colored sparse points.
MVS uses E10 calibration, and VGGSfM is not allowed to consume E10 poses.
VGGSfM is aligned to E10 only after inference for comparison.

### Current evidence status

| Branch | Evidence-backed status |
|---|---|
| Historical E10 | Verified: 125 registered images and 21,744 cleaned colored points |
| Fresh SfM E11 | Verified complete: 125/125 images, 5 cameras, 10,204 colored points, 1.3273668003 px mean reprojection error, 5.5246961976 mean track length, 84.476150969 s engine time |
| Dense MVS | Verified complete: official PyCOLMAP 4.2 CUDA fallback, 125 reference views, 487,451 finite colored points, 3,834.510 s successful-controller time; CPU-only saved-result reload passed |
| VGGSfM | Verified official inference: 125/125 registered, 414,208 RGB points (11,362 tracked + 402,846 trackless extras), 1,519.284 s; read-only reload and a separate coordinate-consistent COLMAP export passed |
| VGGSfM-to-E10 alignment | Verified post-hoc similarity fit: 125 matched cameras, 95 RANSAC inliers, all-camera normalized RMSE 0.053810; not metric ground truth |

Dark-shirt extensions are separate experiments and do not replace these
light-shirt results. The complete
machine-readable identities and failure evidence are indexed in
[the experiment report](EXPERIMENT_RESULTS.md).

### Explain the VGGSfM export correction in the viva

Inference uses square-padded, resized images. The pinned official exporter
restores camera intrinsics to each original image size, but its default leaves
2-D observations in the padded inference frame. Camera parameters and
observations must use the same pixel frame before interpreting reprojection
errors. The separate review export applies the upstream inverse padding/resize
formula to observations and recomputes errors; it changes no pose, intrinsic,
XYZ, RGB, point identity or track. The original output remains immutable.

For all 438,655 linked observations in the full-light run, the corrected mean
Euclidean residual is 5.628313 original-image pixels (RMSE 6.768393). The
1,219.277-pixel uncorrected mean is a coordinate-frame mismatch, not the
reconstruction's actual reprojection quality. There are 2,396 tracks mapped
outside the original image rectangle because upstream retained padding
observations; those are reported, not clamped or deleted. The 402,846 trackless
extras have no observation-based reprojection metric and retain the `-1`
uncomputed sentinel. None of this is an accuracy percentage.

The raw and corrected PLYs are byte-identical because only COLMAP observation
coordinates and error metadata change. Even after post-hoc alignment, the
full-bounds point preview contains substantial scattered/radiating geometry;
more points are not proof of a cleaner person model.

### Explain the separate VGGSfM person cleanup

The user-requested cleanup is a CPU-only **subset selection**, not another
inference or reconstruction. It retains 138,677 of the 414,208 raw points
(4,026 tracked and 134,651 trackless), preserving every selected float32 XYZ
and RGB record byte-for-byte. The original full output remains immutable.

For each raw world point `X`, use VGGSfM's own pose to compute `Xc = R X + t`.
Require finite coordinates and `Xc.z > 0`, divide by depth, then apply that
camera's native-resolution projection including radial distortion. Sample
the matching independently copied person mask at the nearest in-bounds pixel.
A uint8 mask value `>=128` is foreground. If `n` cameras give usable projections
and `f` vote foreground, keep the point when `n>=6` and `f>=ceil(0.9*n)`.

Native mask dimensions must equal the camera dimensions; padding-frame 2-D
feature coordinates are not used for this 3-D projection. The corrected model
has the same VGGSfM poses/intrinsics and supplies consistent original-image
metadata. No E10 pose or old point ID is used. The 125 masks are segmentation
evidence copied into `DATA_ROOT/vggsfm/inputs/light_shirt_cleanup_masks`, not
a hidden cross-method runtime dependency.

An in-bounds projection is not proof that a point is visible: the filter has
no depth-occlusion test, and most retained points are trackless grid additions.
Thus 90% is **mask agreement**, not accuracy and not the original sparse-track
filter unchanged. It can remove real boundary/limb points when masks or camera
estimates disagree. The preview shows a much cleaner isolated person but
still has gaps; no coordinates are optimized or missing surfaces generated.
The implementation is `vggsfm/scripts/clean_light_with_masks.py`, with mask
copying in `prepare_light_cleanup_masks.py`; counts, source checks and the
exact selection rule are in the separate cleanup receipt.
See [the cleanup guide](../vggsfm/CLEANUP.md) for file-by-file responsibilities,
auditable commands and question-and-answer preparation.

## Boundaries that must stay clear

| Question | Classic SfM | Dense MVS | VGGSfM |
|---|---|---|---|
| Main input | raw images | images plus calibrated E10 model | raw images only |
| Estimates cameras? | yes | no | yes |
| Main correspondence | local SIFT features | pixel patches/depth hypotheses | learned long-range tracks |
| Main output | sparse cameras and colored landmarks | dense colored fused points | learned-SfM cameras and colored landmarks |
| Uses E10 poses? | historical E3-E10 did; fresh server SfM does not | yes, as fixed calibration | no |
| Primary GPU work | SIFT extraction/matching | PatchMatch Stereo | neural inference/tracking |
| Metric scale? | no | inherits E10's arbitrary scale | no |

Do not call MVS a fresh camera reconstruction. Do not call VGGSfM's optional
extra triangulated grid points PatchMatch MVS. Do not call any point cloud a
mesh: a mesh requires faces, while these primary PLY files contain colored
vertices.

If a presentation or future UI shows five buttons, do not describe that as five
algorithms. MeshLab is an external viewer. “COLMAP baseline” refers to the
classic COLMAP/PyCOLMAP sparse pipeline, while the pixel-wise method uses
COLMAP's different dense PatchMatch/fusion modules. The three scientific
branches implemented here are sparse SfM, dense MVS and VGGSfM.

## Permanent storage and method isolation

The source root is fixed at:

```text
/home/anas.khan/cv802_project/project1/ass1
```

It may contain only source code, tests, small configuration and documentation.
The data root is fixed at:

```text
/l/users/anas.khan/cv_802_ass1
```

Images, masks, databases, models, PLY files, environments, downloaded upstream
repositories, model weights, caches, logs, temporary files and generated
reports all belong in the data root. Each engine reads and writes only its own
method directory. Reused inputs are physical, checksum-verified copies; a
method never writes into another method's experiment.

This is more than housekeeping. It prevents:

- a dense run from mutating its sparse initialization;
- VGGSfM from accidentally seeing E10 poses;
- a cache under home from hiding an unrecorded model download;
- Git from absorbing datasets, environments or generated results; and
- a changed source path from being mistaken for changed image content.

The relocation code inventories files, copies them, independently verifies
relative names, logical sizes and SHA-256 values, extracts Python/source files
from mixed historical directories, prepares each method's input copy, and only
then permits deletion of verified heavy duplicates. Cleanup is explicitly
gated; it never deletes a mixed directory merely because its name looks heavy.

### Supporting orchestration code

The batch orchestration is reproducibility/safety code, not a fourth
reconstruction method:

| File | Responsibility |
|---|---|
| `scripts/hpc/cv802_ass1_3h.sbatch` | fixed one-node/one-GPU resource envelope, log destination, timeout warning and non-requeue policy |
| `scripts/hpc/run_handoff_3h.sh` | validates the canonical roots/runtime and redirects shared Conda, pip, XDG, Hugging Face, Torch, bytecode and temporary state to the data root |
| `scripts/hpc/write_allocation_receipt.py` | records the actual scheduler, node, GPU, driver, mount and space identity |
| `scripts/hpc/run_storage_stages.py` | invokes explicit copy/verify/input/finalize relocation phases |
| `scripts/hpc/run_engine_stages.sh` | runs MVS, VGGSfM and fresh SfM as separately reported methods, retries only the verified cleanup gate, and writes one final status receipt |

The method wrapper does not let one engine's failure erase another engine's
evidence: each status is captured and later stages are attempted. Signals are
forwarded through the child process tree so a timed-out wrapper does not leave
an orphan GPU process. The five-minute scheduler warning creates a stop receipt
and gives resumable stages time to preserve their state. The orchestration never
turns a failed method into success merely because later methods ran.

## Inputs and historical experiment meaning

The light-shirt capture has 125 views: 91 still photographs and 34 selected
video frames. The dark-shirt-with-crutches capture has 290 views: 183 stills
and 107 video frames. They are separate subjects and must never be mixed.

The historical sequence is:

- **E1:** fresh-camera exhaustive sparse SfM.
- **E2:** rectangle filtering of E1 points; no new reconstruction.
- **E3:** native-resolution quality features, selected pairs, guided matching
  on a subset, fixed E1 poses and retriangulation.
- **E4/E5:** E3 point filtering at 90%/97% foreground-mask agreement.
- **E6/E7:** E3 recipe with guided matching off, then 90% filtering.
- **E8/E9:** top-20 vocabulary retrieval with selected guided matching, then
  90% filtering.
- **E10:** light only; the E3 feature set, all 7,750 unique light-image pairs,
  guided matching offered to every pair, fixed E1 poses, retriangulation and
  the same 90% cleanup policy.

E3-E10 are refinements in the E1 camera frame, not independent camera
estimations. The new headless Linux SfM engine deliberately provides a fresh
camera mode so this distinction is not hidden.

For `N` images, exhaustive matching considers
`N(N-1)/2` unordered non-self pairs. With 125 images that is 7,750 pairs.
Pair selection chooses which two images to compare. Guided matching acts later,
inside a selected pair, using estimated two-view geometry to search for more
correspondences. A pair offered to guided matching may still fail to produce
valid geometry.

The historical 90% value is foreground-mask agreement over a point's usable
projected views. It is neither reconstruction accuracy nor the percentage of
points retained. A 97% threshold is stricter and removed additional real-looking
leg and footwear detail in this capture, so 90% was retained as the better
visual trade-off.

## Method 1: fresh-camera headless SfM

### Execution flow

`sfm/run_headless.py` enters `sfm_engine/cli.py`, converts the CLI arguments to
an immutable `SfMConfig`, and calls `run_reconstruction()` in
`sfm_engine/pipeline.py`.

The implementation then:

1. resolves the configured data root and rejects paths outside
   `DATA_ROOT/sfm`;
2. recursively discovers ordinary image files and rejects image symlinks;
3. hashes each image while checking that device, inode, size and modification
   time remain stable during hashing;
4. imports the pinned PyCOLMAP 4.2.0 API and resolves CPU/CUDA without silently
   downgrading an explicit CUDA request;
5. combines input hashes, engine-source hash, dependency/runtime identity and
   every reconstruction option into the experiment identity;
6. acquires an OS `flock` so two processes cannot mutate one experiment;
7. extracts SIFT features into a fresh attempt-local COLMAP database;
8. performs exhaustive or sequential matching and COLMAP geometric
   verification;
9. performs incremental mapping, triangulation and bundle adjustment;
10. selects the valid connected component with the most registered images,
    breaking a tie by point count;
11. writes and reloads the COLMAP model, validates it, exports a colored PLY,
    and validates the PLY; and
12. re-hashes the inputs before atomically publishing outputs and a receipt.

### Camera and projection model

The normal light-shirt configuration uses `SIMPLE_RADIAL`. For a camera-space
point, normalized coordinates are `x=X/Z`, `y=Y/Z` and `r^2=x^2+y^2`. The
simplified projection is:

```text
u = f (1 + k r^2) x + cx
v = f (1 + k r^2) y + cy
```

The unknowns are focal length `f`, principal point `(cx,cy)`, and radial
distortion `k`. `per_folder` camera grouping shares intrinsics only within a
prepared capture group; `single` would incorrectly force one calibration if
different phones/lenses or zoom settings were used.

### Incremental geometry

Classic SfM begins with a strong two-view seed, recovers relative pose,
triangulates initial tracks, registers more images from 2D-to-3D
correspondences, triangulates new tracks, and repeatedly refines the model.
Bundle adjustment minimizes robust reprojection error:

```text
min over cameras C_i and points X_j
    sum over observations (i,j) rho(||x_ij - project(C_i, X_j)||^2)
```

This objective measures internal image consistency. It does not supply metric
scale, and a low error does not prove that a moving human was reconstructed
anatomically correctly.

### CUDA proof and scope

The environment pins `pycolmap-cuda12==4.2.0`. A CUDA-labelled run requires the
wheel to report CUDA support, exactly one PyCOLMAP-visible device, an active
Slurm allocation, a successful `nvidia-smi` identity query, and consistent
Slurm/CUDA visibility. CUDA accelerates SIFT extraction and matching. The code
does not claim that incremental mapping and every bundle-adjustment operation
run on the GPU.

### Cache, retry and publication

An interrupted SQLite database is not treated as a scientific checkpoint.
`--resume` preserves the failed attempt and starts a new numbered attempt with
a clean database. It never appends to an ambiguous partial database.

Outputs are created inside the attempt, reloaded and validated, and then moved
into the public output directory with a same-filesystem atomic rename. A cache
hit is accepted only after current paths, model contents, registered names,
point/color values, PLY structure, configuration and receipt all agree.

### SfM code map

| File | Responsibility |
|---|---|
| `sfm/run_headless.py` | minimal executable entry point |
| `sfm/sfm_engine/cli.py` | CLI parsing and `SfMConfig` construction |
| `sfm/sfm_engine/pipeline.py` | path safety, manifests, PyCOLMAP stages, locking, validation, retry and atomic publication |
| `sfm/scripts/setup_linux_cuda.sh` | data-root environment creation and CUDA-wheel receipt |
| `sfm/assignment1/modules/colmap/api.py` | preserved historical starter/Mac implementation |
| `sfm/estimate_cameras.py` | paste-ready synchronized copy of the historical method |

The fresh server experiment and E1-E10 must be discussed separately. The
historical code remains important evidence but is not silently reused as the
fresh-camera headless implementation.

### Verified E11 result and the correct comparison to E10

E11 attempt `0001` was a cache miss and completed without recovering an
interrupted attempt. It hashed 125 inputs, registered all 125, selected mapper
component `0`, and published a model with five cameras and 10,204 colored
points. The mean reprojection error was 1.3273668003 pixels and the mean track
length was 5.5246961976 observations. The engine receipt reports 84.476150969
seconds; environment installation is outside that time scope. The output PLY
has 10,204 vertices and its SHA-256 is
`c48c7abb0123c22fe4f0f959337c58298f178526fa654973843c14d6e57d4f6a`.

The viva-safe conclusion is that E11 achieved complete view coverage and
passed structural validation. It is not valid to say it is geometrically more
or less accurate than E10 from point count or reprojection error alone. E10 and
E11 use different camera-estimation histories, image scales, feature recipes
and cleanup policies, and neither is physical ground truth. Visual assessment
of the E11 trajectory, subject completeness and floaters remains explicitly
pending.

### Preserved starter backend and fixed-pose quality path

The original non-UI backend remains in
`sfm/assignment1/modules/colmap/api.py::_estimate_cameras`. Its ordinary path
recursively identifies supported images, builds an input/configuration
signature, chooses `PER_FOLDER` camera sharing when `camera_groups.json` is
present, and either reloads the largest valid cached component or stages a new
database/model in a temporary directory. A fresh ordinary run extracts SIFT,
runs the selected exhaustive/sequential/vocabulary matcher with geometric
verification, performs incremental mapping, selects the largest valid
component, and atomically replaces the old database/model only after success.
If installation fails midway, the previous result is renamed back.

When a dataset contains `sfm_refine.json`, the backend takes the separate
fixed-pose quality path used by E3-E10:

1. load the valid baseline reconstruction and require the quality image names
   to equal the baseline registered names;
2. rescale each baseline camera to its quality-image dimensions while retaining
   the baseline extrinsic poses;
3. freeze a feature identity from image metadata, baseline artifacts, box-file
   hash and extraction recipe;
4. extract affine/domain-size-pooled SIFT on padded person crops, then retain a
   spatially distributed feature budget;
5. validate the ordinary and guided pair lists, requiring every guided pair to
   be present in the ordinary list;
6. run ordinary matching first, then delete/re-estimate two-view geometry for
   the guided subset so guided refinement is actually applied;
7. for E10's opt-in resumable all-pairs mode, commit verified batches to durable
   checkpoints under an exclusive file lock; and
8. call `triangulate_points` with the baseline reconstruction, fixed poses,
   no intrinsic refinement, color extraction, reprojection filtering and a
   minimum triangulation angle.

This explains why E3-E10 can generate new sparse points and matching results
without re-estimating camera poses. The mask cleanup scripts operate afterward:
they project an existing point into usable cameras, read the corresponding
foreground mask, count positive/usable votes, enforce minimum track/view
support, and save a filtered copy. They do not move cameras or create new 3D
points. Dark-subject cleanup has an additional reviewed crutch-protection rule;
light E10 does not contain a dark-subject run.

The historical cache includes absolute paths/mtimes and must remain preserved
as provenance. The new server engine instead uses portable content hashes. Do
not edit old manifests merely to make them look Linux-native.

## Method 2: pixel-wise COLMAP MVS

### Why this is the requested pixel-wise method

The MVS engine is COLMAP's CUDA PatchMatch Stereo and stereo fusion. It works
at image pixels rather than only repeatable SIFT landmarks. It implements the
assignment's pixel-wise view-selection/multi-view stereo branch; MeshLab is
only a viewer for its result.

### Execution flow

`mvs/run_mvs.py` enters `cv802_mvs.cli`. The workflow is:

1. `stage-inputs` parses the E10 COLMAP model and copies only its registered
   images, calibration/model files and approved masks into the MVS experiment;
2. `plan` constructs and displays exact shell-free COLMAP argv without running
   them;
3. `validate --runtime` proves the storage, one-GPU allocation, pinned
   container, checksums and configured COLMAP options;
4. optional preprocessing thresholds/dilates each distorted foreground mask
   and sets the distorted-image background to black;
5. `image_undistorter` resamples images and cameras into a consistent pinhole
   workspace;
6. the controller edits `stereo/patch-match.cfg` so every image remains a
   reference and COLMAP automatically chooses up to ten source views for it;
7. CUDA PatchMatch estimates per-reference depth and surface-normal maps with
   photometric and geometric consistency; and
8. `stereo_fusion` back-projects and merges mutually consistent samples into a
   colored `fused.ply`, which is completely scanned by the validator.

The moderate light configuration caps the longest image dimension at 1600,
uses ten source views per reference, five PatchMatch iterations, fifteen random
samples, a five-pixel window radius, one-pixel window step, a minimum of two
consistent source estimates during filtering, geometric consistency, 16 GB
cache settings and 16 controller threads. Fusion requires five supporting
pixels/observations, at most two-pixel reprojection error, at most 0.01 relative
depth error, at most ten-degree normal error, and checks up to fifty images.
These are hyperparameters, not measured accuracy.

### PatchMatch reasoning

For each reference pixel, PatchMatch maintains a depth and normal hypothesis.
It improves hypotheses through initialization, propagation from neighbouring
pixels, random refinement and multi-view photometric scoring. Geometric
consistency then checks whether the hypothesized 3D sample agrees with depth
predictions in other views and projects back consistently.

Pixel count grows approximately quadratically with linear resolution, and
matching work also grows with the number of source images. The 1600-pixel and
ten-source caps therefore control VRAM and runtime without dropping any of the
125 reference images.

Fusion is necessary because depth maps are camera-centric and duplicate the
same physical surface. It transforms valid depths to the common E10 world
frame and merges samples that pass reprojection, depth and normal checks.

### Mask semantics

The saved foreground masks are in the original distorted pixel coordinates.
They are applied before undistortion so COLMAP transforms the masked pixels and
camera model together. Applying an original mask directly to an undistorted
image would misalign boundaries.

This preprocessing is not the historical sparse 90% filter. Dense points do
not inherit the same feature tracks or usable-view denominator. The report must
say that masking encourages subject-focused matching; it does not prove every
fused point belongs to the subject.

### MVS resume and code map

A stage receipt binds exact argv and configuration to output hashes. Resume
skips a stage only after revalidating its output. Complete PatchMatch maps may
be reused because COLMAP has per-view resume behavior. Interrupted
undistortion, fusion and meshing products are archived and restarted. An
intentional overwrite also archives old artifacts instead of recursively
deleting them.

A completed-run reload takes a separate read-only branch before directory
creation, runtime preflight or metadata writes. It verifies complete state,
configuration and original input identity, checks all expected stage receipts
(including masking), and rescans/hashes the final PLY. It returns the original
final report with `cache_hit: true`; it does not overwrite reconstruction time
with cache-check time or require another CUDA allocation. Corrupt or missing
completion evidence fails closed. The actual light-result check recorded zero
executor, runtime-probe and metadata-write calls with 1,038 file metadata records
and 14 JSON hashes unchanged.

| File | Responsibility |
|---|---|
| `mvs/src/cv802_mvs/paths.py` | canonical root and experiment-name safety |
| `config.py` | strict JSON schema and range/one-GPU checks |
| `colmap_model.py` | independent COLMAP binary/text model parsing |
| `staging.py` | stable, checksummed, model-referenced input copying |
| `masking.py` | distorted-grid foreground preprocessing |
| `commands.py` | exact shell-free COLMAP/container argv |
| `validation.py` | allocation, CUDA, container, model and PLY checks |
| `lifecycle.py` | receipts and recoverable archive transitions |
| `runner.py` | pipeline ordering, subprocess groups, logs, resume and metrics |
| `cli.py` / `run_mvs.py` | user commands and entry point |

The primary output is the dense colored point cloud. Meshing is deliberately
disabled in the first configuration so surface interpolation is not confused
with measured/fused points.

The first MVS container launch did not reach PatchMatch: the generated,
hashed 3.0-GB SIF failed to mount on `gpu-04` with a SquashFS bad-superblock
error. Recovery switched to the official PyCOLMAP 4.2 CUDA API in the same
Slurm allocation and reused only verified method-local inputs/environment, not
an accepted dense result. That fallback completed on 2026-09-19 at 21:18:18
Asia/Dubai. Its final receipt validates all 125 reference images, 125 geometric
depth maps, 125 normal maps and a 487,451-vertex RGB PLY. Meshing remains disabled.
The 3,834.510-second successful-controller timer excludes installation, the
failed container launch and the previously completed 153.151-second mask stage.

The first controller's GPU monitor queried the entire node: its 17,137 MiB peak
must not be claimed as MVS's own memory usage. A later process-specific sampler
observed 990 MiB GPU memory and 4,602,696 KiB summed RSS, but only for the final
1,250.153 seconds of PatchMatch. It is a partial-run sampled peak, not a full-run
memory requirement. See [`RESOURCE_MEASUREMENTS.md`](RESOURCE_MEASUREMENTS.md)
and the exact receipt links in [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).

## Method 3: independent official VGGSfM

### Ownership boundary

The pinned official VGGSfM repository owns the neural camera predictor,
learned tracker, triangulator and geometric optimization. This repository owns
safe input staging, a reviewed configuration interface, headless CUDA
invocation, provenance, recovery, binary COLMAP validation and output
normalization. The viva answer is “we integrated and validated the official
method,” not “we reimplemented the VGGSfM network.”

This is inference with pretrained weights, not training. The pinned upstream
source and model identity are recorded in
[PROVENANCE.md](../vggsfm/PROVENANCE.md).

### Execution flow

1. `scripts/prepare_input.py` physically copies and hashes the method's raw
   images below `DATA_ROOT/vggsfm/inputs`.
2. `InferenceProfile` accepts only reviewed options. Unknown keys fail instead
   of silently enabling ground-truth loading or visualization.
3. The engine hashes the input set, profile, official Git SHA and model
   identity into one request fingerprint.
4. A run ID is bound to that fingerprint. A changed request requires a new run
   ID; a failed identical request needs explicit `--resume`.
5. The attempt stages a flat official scene using deterministic names of the
   form `six_digit_sequence__sha256(relative_path)[:16]__basename`. The
   sequence prefix preserves basename/capture order after upstream sorts the
   flat directory. The reversible name map is
   stored in `request.json`; no E10 database, camera or point file is staged.
6. CUDA preflight imports PyTorch, verifies device visibility, and records the
   runtime identity.
7. The wrapper invokes official `demo.py` with an argv list and every UI/video
   visualization option disabled.
8. The official code predicts cameras, tracks features, triangulates, rejects
   inconsistent observations and performs geometric refinement/bundle
   adjustment.
9. The wrapper discovers exactly one supported COLMAP model layout, validates
   its cameras/images/points, converts colored `points3D.bin` to binary PLY,
   hashes every output and atomically publishes it.

### Profile meaning

The light profile uses:

- `SIMPLE_RADIAL` cameras;
- `shared_camera=false`, allowing independent per-view intrinsics because
  the capture includes multiple camera/calibration groups;
- six query frames and 2,048 ALIKED query points;
- 1024-pixel official input sizing;
- fine tracking;
- FP16 mixed precision;
- deterministic seed zero; and
- an extra-point grid interval of ten pixels, tracked through 16 temporal
  neighbours and concatenated to the main points.

The extra grid points are tracked and triangulated, but they are not part of
the earlier joint bundle adjustment. They make a denser VGGSfM point output;
they do not turn VGGSfM into the independent PatchMatch MVS method.

### Binary normalization

COLMAP `points3D.bin` begins with a little-endian point count. Each point stores
an ID, float64 XYZ, uint8 RGB, float64 reprojection error, and a variable-length
track of `(image_id, point2D_index)` pairs. The converter streams this file and
writes float32 XYZ plus the same RGB to a binary little-endian PLY. The original
COLMAP binary remains the full-precision, track-bearing output.

### Failure, cache and code map

A failed attempt keeps its log, partial official output and failed receipt.
`--resume` creates the next clean attempt; it does not pretend that official
neural inference supports checkpoint-level continuation. Published output is
immutable. A later cache hit re-hashes every normalized output before returning
`cached`.

The first VGGSfM launch failed before cloning or installation. The install
script had set `CONDARC=/dev/null`; on this cluster, Conda 4.11's
`conda shell.bash hook` returned status 1 with `KeyError: 8192`, and `set -e`
stopped the script. Recovery replaced it with `vggsfm/configs/condarc`, which
keeps environments and caches in the permanent data root, then restarted the
pinned-source installation inside the same allocation. Installation and its
actual A100 CUDA probe subsequently passed. A 16-view pilot's official
inference succeeded in 122.063 s but the wrapper initially rejected valid
zero-based camera ID 0. The corrected validator accepts legitimate zero IDs
while retaining reserved-ID, finite-value and reciprocal-track checks. A
separate CPU review recovered the pilot export without rerunning inference
or replacing the failed receipt. The full 125-view run remains a distinct
experiment; pilot success is not a claim of full-run completion.

COLMAP/VGGSfM can store `-1` when a point's reprojection error has not been
computed. It is not a negative measured error or a quality score. Updated
metrics exclude such sentinels and report unavailable counts explicitly;
immutable earlier receipts retain their raw values with this interpretation.

| File | Responsibility |
|---|---|
| `vggsfm/vggsfm_engine/paths.py` | fixed data layout and containment |
| `profile.py` | allow-listed profile parsing and validation |
| `engine.py` | request identity, attempt lifecycle, invocation and publication |
| `colmap.py` | native model validation and streaming colored PLY conversion |
| `io_utils.py` | hashing and atomic JSON/file helpers |
| `cli.py` / `run_vggsfm.py` | doctor, plan and run commands |
| `scripts/prepare_input.py` | independent image/mask copy |
| `scripts/install_linux.sh` | pinned upstream/environment installation below the data root |
| `scripts/write_install_receipt.py` | installed Git/package/GPU provenance |

## Post-hoc VGGSfM-to-E10 evaluation

Independent monocular reconstructions have arbitrary origin, orientation and
scale. Raw coordinates cannot be compared directly. The evaluator uses the
same registered image as a camera-center correspondence and computes a proper
similarity transform only after both methods have finished.

For COLMAP's world-to-camera pose:

```text
x_camera = R X_world + t
```

the camera center is:

```text
C_world = -R^T t
```

For matched VGGSfM centers `x_i` and E10 centers `y_i`, robust Umeyama fitting
estimates:

```text
y_i approximately equals s R x_i + t,
s > 0 and det(R) = +1
```

The determinant constraint rejects a mirror as a valid rotation. Deterministic
RANSAC samples non-collinear triplets, scores all cameras, refits its consensus,
and reports both all-camera and inlier errors. E10 is an arbitrary-scale
reference, so normalized errors divide by the RMS radius of matched E10 camera
centers.

The aligned PLY is published only if minimum-match, consensus, rank, finiteness
and relative-error gates all pass. Otherwise the JSON report remains available
and the PLY is honestly marked `withheld`. The evaluator never guesses image
correspondence from basename; it reverses the immutable digest-name map from
VGGSfM's request receipt and matches exact nested input names.

Alignment supports comparison; it does not turn E10 into physical ground truth
and does not prove point-level surface accuracy.

## Validation layers

The implementation separates five different questions:

1. **Storage validity:** are all mutable/heavy files in the data root, with
   physical method-specific copies and verified relocation receipts?
2. **Runtime validity:** was exactly the intended GPU visible, and were the
   pinned executable/container/model identities recorded?
3. **Model validity:** are cameras, poses, points and colors finite and
   internally consistent, with legal image references?
4. **Artifact validity:** can the saved model be reloaded, does its PLY header
   and vertex stream agree, and do output hashes match the receipt?
5. **Scientific interpretation:** what does the method actually estimate, and
   what limitations remain without external ground truth?

Passing the first four does not automatically establish the fifth. A point
cloud can be numerically valid yet incomplete or visually noisy.

## Why the human capture is difficult

SfM and MVS normally assume a static scene. A person may move between views;
plain clothing supplies few repeatable features; specular/soft fabric changes
appearance; self-occlusion hides surfaces; video frames may blur; rolling
shutter can distort geometry; and background texture can dominate matching.
Masking suppresses some background evidence but may also remove boundaries or
valid body detail. These are expected limitations, not evidence that a file
validator failed.

## Reading the final measurements

- **Registered-image count** measures coverage, not pose accuracy.
- **Point count** measures retained density, not correctness.
- **Mean reprojection error** measures consistency with fitted observations,
  not physical error.
- **Mean track length** indicates how many observations support a typical
  sparse point, but does not alone establish good geometry.
- **MVS fused vertices** can include outliers; density is not surface accuracy.
- **Runtime/VRAM** are hardware-and-configuration measurements, not quality.
- **Aligned camera-center error** measures trajectory agreement after removing
  similarity gauge freedom; E10 is still not ground truth.
- **A MeshLab screenshot** is a qualitative visualization, not a numerical
  evaluation.

The final measured values belong in
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), copied from generated receipts
rather than memory or terminal scrollback.

## Common viva questions

**Why are there three methods if COLMAP appears in more than one?**

COLMAP supplies reusable formats and geometry components. Classic SfM estimates
cameras from sparse features. COLMAP MVS keeps cameras fixed and estimates
dense pixel depths. VGGSfM independently predicts cameras/tracks with a learned
model and may serialize its result in COLMAP format. Sharing a file format does
not make the algorithms the same.

**Is MeshLab a baseline reconstruction method?**

No. MeshLab loads, colors, inspects and optionally processes an existing point
cloud/mesh. It does not replace SfM, PatchMatch MVS or VGGSfM in this pipeline.

**Why initialize MVS from E10?**

MVS requires calibrated, registered cameras. E10 supplies all 125 camera poses
and subject-focused sparse points for overlap/depth-range initialization. The
dependency is disclosed, so MVS is a dense extension of E10, not an independent
camera baseline.

**Why may VGGSfM be aligned to E10 but not initialized from it?**

Post-hoc alignment removes unavoidable coordinate gauge differences after
independent inference. Supplying E10 poses before or during inference would
leak the reference solution and invalidate the independence claim.

**Why use content hashes?**

Absolute paths and modification times change during a Mac-to-Linux transfer.
Relative names, sizes and SHA-256 values preserve portable content identity and
still detect changed pixels or models.

**Why not reuse any partial database or neural attempt?**

A readable SQLite file can contain only some features or matches, and the
official neural runner has no general mid-scene continuation contract. Clean
numbered attempts preserve forensic evidence without mixing partial state into
a result presented as reproducible.

**Why atomic publication?**

Validation happens before a rename makes the result visible at its final path.
Therefore consumers see either the previous complete result or the new complete
result, not a half-written model.

**What does headless mean?**

The engines call Python/COLMAP command-line code with visualization disabled.
They do not open Open3D, Qt, OpenGL, Visdom or Gradio windows and do not require
an X display.

**What is the most important honesty statement?**

The project validates computation and internal geometry but has no physical
ground-truth body scan. Results must be described as reconstructed, finite,
colored points with measured consistency and coverage—not as a percentage of
real-world anatomical accuracy.

## Why dark cleanup must protect the crutches separately

A person-segmentation mask may label thin metal crutch shafts as background.
Taking only points with 90% body-mask agreement can therefore delete real
supporting objects. The dark MVS derivative uses the **union** of body-consensus
points and separately protected crutch candidates. In the measured output,
68,112 protected candidates fail body consensus but survive that union.

Protection is geometric, not reuse of historical point IDs: new dense points
are checked against bounded capsules and multiple annotated image corridors
with separated viewing rays, using calibration in the same coordinate frame.
The union retains exact original vertices; it does not move, synthesize or
repair them. Visual overlay review is required because foreground projection
alone has no occlusion reasoning and can preserve points behind the person.

VGGSfM independently estimates a different world frame. Applying the MVS/E1
world-space capsules directly to its raw points would be invalid. Its dark
cleanup must instead use its own camera projections and independently
validated support, or an explicitly verified post-hoc frame transformation.
Reusing native-resolution 2-D annotations is valid only when image identity,
pixel dimensions, distortion and projection conventions are checked.

For the implemented dark VGGSfM rule, let `h_i(X)` mean that a raw point `X`
projects inside a reviewed corridor in annotated view `i`. A corridor is a
polyline expanded by a documented half-width in native pixels. For each
camera centre `C_i`, define the unit ray `r_i = (X-C_i)/||X-C_i||`. Crutch
support requires a triple `(i,j,k)` with all three `h` values true and
`r_a · r_b <= cos(15 degrees)` for **every pair** in that triple. Testing only
three hits, or only one well-separated pair, would not implement this rule.
The body/crutch union is then applied without changing any retained coordinate.
These are geometrically supported **candidates**, not certified visible metal
surfaces: the rule still lacks an occlusion model and cannot create missing
geometry.

## Evidence to inspect before the viva

Use generated data-root receipts, not guessed values:

- `provenance/gpu-allocation-<job>.json` for allocation/GPU identity;
- `runtime/jobs/<job>/engine-status.json` for orchestration stage status;
- `sfm/experiments/<run>/receipt.json` and its output model/PLY;
- `mvs/experiments/<run>/receipts/` and `manifests/final_validation.json`;
- `vggsfm/install_receipt.json`, experiment request/attempt receipts and output
  `manifest.json`;
- `evaluation/<id>/report.json` for post-hoc alignment; and
- `relocation/receipts/` for copy, verification, extraction, input-copy and
  home-cleanup evidence.

If any expected receipt is absent or says `failed`, report that status and the
preserved failure; do not fill the final report from intended configuration.
