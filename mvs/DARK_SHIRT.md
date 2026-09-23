# Dark-shirt MVS: raw reconstruction and separate crutch-preserving cleanup

`configs/dark_e3_1024_raw_pycolmap.json` defines a **new**, initially unmasked
dense experiment, `dark_e3_colmap_mvs_1024_raw_v1`. It does not alter or rename
E1–E10; dark E10 never ran. Generated files are exclusively under
`/l/users/anas.khan/cv_802_ass1/mvs`.

## Why E3 and why unmasked?

The source is E3 `black_shirt_crutches_quality`, with all 290 registered views
(183 photos + 107 video frames), four cameras, and 101,150 sparse points.
E6 has 78,888 points; E8 has 95,896. E3 supplies the richest available
unfiltered quality correspondence graph for MVS neighbour selection. This is
not an accuracy ranking based on point count: these variants inherit the same
E1 camera poses and have no ground-truth geometry. E3 reprojection error is
1.4075001083 px and mean track length 4.6803559071 in the historical report.

Saved **body masks omit crutch shafts**. Consequently the new config explicitly
uses `masking.mode=none`: raw background remains and no claim of 90%/97%
foreground cleanup is made. A future cleaned result must be a separate output.
Do not enable the light-shirt black-background mask recipe for this subject.

The reduced-resource profile uses all 290 images at maximum dimension 1024,
five source views per reference, three PatchMatch iterations, geometric
consistency, four CPU threads, and eight-GB host caches. Fusion requires five
supporting pixels and the same reprojection/depth/normal gates as light.
This is a resolution/settings pilot, not an image subset and not equivalent
quality to the light 1600/10-source/5-iteration run. Duration is measured, not
guaranteed to fit a new allocation.

The actual server retry shares the same authorized A100 with other CV802
VGGSfM runs for part of its lifetime. Its elapsed time is therefore an
operational measurement under concurrent GPU/CPU load, not an isolated speed
benchmark. The coordinator also observed overlapping CPU affinity in the
Slurm steps near the end; no unrelated project's allocation was used.

## Verified server result

The corrected retry completed on 2026-09-19 at **22:28:44 Asia/Dubai**, within
job `264198`, without a new allocation. It produced **616,827 finite RGB
points** from all 290 views, with 290 geometric depth maps and 290 normal maps.
The raw PLY is 16,654,563 bytes; SHA-256 is
`6a61c61382692de34a6a6836f6d6e5ccf2354679b68dd885690dc7b660e6199c`.

| Measured item | Value |
|---|---:|
| Successful-controller elapsed time | 3,392.944895 s |
| Complete successful command lifecycle | 3,395.090494 s |
| Undistortion | 101.416439 s |
| PatchMatch, both passes | 3,186.888232 s |
| Fusion | 99.559768 s |
| Sampled process-tree peak GPU memory | 584 MiB |
| Sampled process-tree peak RSS | 4,646,764 KiB |
| Resource samples | 1,641 |

These timings exclude the preserved first failed launch. Resource figures are
attributable process-tree samples, not exact allocator maxima; summed RSS can
double-count shared pages. Do not replace them with the old stage monitor's
GPU-wide values, which can include concurrent VGGSfM.

Authoritative evidence, all below `DATA_ROOT/mvs`:

```text
experiments/dark_e3_colmap_mvs_1024_raw_v1/manifests/final_validation.json
experiments/dark_e3_colmap_mvs_1024_raw_v1/run_state.json
experiments/dark_e3_colmap_mvs_1024_raw_v1/outputs/fused.ply
telemetry/dark_e3_colmap_mvs_1024_raw_v1-command-v2/receipt.json
runtime/cache-validation/dark1024-readonly-v1.json
reviews/dark-e3-1024-crutch-projections-v1/review.json
reviews/dark-e3-1024-crutch-projections-v1/visual_review.json
```

The CPU-only cached reload passed in 5.387587 s: **zero** reconstruction,
runtime-probe or metadata-write calls; 2,080 file metadata records and 21 JSON
hashes remained unchanged. Original reconstruction timing was preserved.
The complete MVS source suite passed 29/29 tests in 8.895 s; the log is
`DATA_ROOT/mvs/runtime/dark-review-suite-20260919.log`.

### Actual crutch review and limitations

All 616,827 new XYZ rows were checked. Undistortion changed no camera-pose
elements. The two bounded capsule regions contained 43,998 and 66,906 dense
points; 31,243 A candidates and 41,152 B candidates also passed the three
separated corridor-projection test. These are **candidate support counts**,
not numbers of proven crutch-surface points or an accuracy percentage.

All six newly generated photo overlays were visually inspected. Frontal and
oblique overlays broadly align candidate bands with both shafts/forks, but
bands are thicker than the metal, gaps remain, and side/back projections can
overlap occluding arms, torso or legs. This demonstrates why corridor votes
alone cannot certify dense visibility or replace the old sparse quality gates.
`visual_review.json` accepts only raw publication with these limitations and
explicitly does **not** approve a 90%/97%-cleaned dense cloud.

The raw orthographic preview is
`DATA_ROOT/evaluation/preview-mvs-dark-raw-v1/preview.png` with its own receipt.
It displays 205,609 deterministically sampled points without coordinate
clipping or alignment. Background and scattered outliers remain visible;
the stored 616,827-point raw PLY is unchanged.

## Independent copies and verification

`scripts/prepare_dark_inputs.py` runs on the existing allocated compute node.
It uses the standard checksum-verified stager for the 290 JPEGs, complete E3
COLMAP model and 290 body masks; input symlinks are resolved into physical copies.
It checks image dimensions and calibration references and retains historical
experiment labels/source hashes. It refuses an existing destination.

`inputs/provenance/crutches` additionally preserves the original body manifest,
approved crutch-protection JSON, its reviewed historical overlay, E3 report,
and the exact hash-verified E1 reference model. A separately rebased protection
JSON points only into this MVS experiment; original JSON bytes remain unchanged.
Preparation verifies equal registered names, fixed camera poses and rescaled
intrinsics against the approved reference. All copy/validation evidence is in
`manifests/input_provenance.json` and `manifests/dark_preparation.json`.

Original sparse helpers are copied byte-for-byte into `mvs/legacy_tools`:
`filter_person_crutches.py`, `crutch_protection/build_protection.py`, and
`work/quality-sfm/compare_models.py`. Their historical path assumptions and
provenance are retained; they are algorithm references, not a runnable dense
cleanup command and not imports from the live SfM engine.

## Separate CPU-only crutch-preserving cleanup

`scripts/clean_dark_person_crutches.py` implements the separately named
`DATA_ROOT/mvs/outputs/dark-e3-mvs-body90-crutches-v1` derivative. It never
overwrites the 616,827-point raw reconstruction. A `validated_pending_visual_review`
receipt is not publication approval; the six new overlays and full-bounds
preview must be reviewed first, with a separate visual-review receipt.

For each original dense point, the body test counts positive-depth,
in-bounds projections through all 290 original E3 cameras. Camera distortion is
applied before calibrated pixel coordinates are scaled to each native mask's
width/height. Calibration and mask aspect ratios must agree within 0.2%;
per-view dimensions, scale factors and vote totals are recorded. Mask lookup
uses the containing pixel (`floor`); grayscale values >=128 are foreground.
Body points require at least six usable projections and at least 90%
foreground agreement. This is a **silhouette agreement threshold, not a 90%
accuracy claim**. Out-of-bounds/behind-camera views are not negative votes.

Crutches are protected independently: recompute bounded approved A/B capsule
membership on the **new dense XYZ**, then require a confirming triple among
the six annotated camera views with >=15-degree pairwise camera-to-point ray
separation. Corridor membership is the historical per-view union of A/B
normalized polylines and their annotated half-widths. The final selection is
`body_consensus OR protected_crutch_candidate`. Thus a shaft can survive even
with zero body-mask votes. Historical sparse point IDs are never transferred.

Important limitations: projections do not prove that a point is visible in
the source image; no depth-buffer/occlusion test is added. Capsule/corridor
points can include nearby background, occluded body surfaces and ambiguous
A/B support. The separate raw result remains available for comparison.
Sparse track-error/angle quality gates cannot be claimed for dense vertices.

The binary PLY writer copies each selected original vertex record verbatim,
including XYZ, RGB and normals, in original order. It reloads the output and
checks byte identity against the selection. `selection_audit.npz` retains
source vertex-row indices, all pointwise body foreground/usable vote counts,
and body/protected flags; these row indices are not historical SfM point IDs.
`projection_rows.json` records every camera/mask mapping. Before/after SHA-256
records cover raw PLY/receipts, calibration, masks, approved protection and
reference model. New overlays show body in green and protected candidates in
red over the six annotated photographs. No models, images or environments are
created in the source tree.

Run only inside an existing explicitly authorized CPU Slurm step, with the
method environment and all caches/temp paths below `DATA_ROOT/mvs`:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
TMPDIR=/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests \
XDG_CACHE_HOME=/l/users/anas.khan/cv_802_ass1/mvs/runtime/cache \
PYTHONPYCACHEPREFIX=/l/users/anas.khan/cv_802_ass1/mvs/runtime/cache/pycache \
/l/users/anas.khan/cv_802_ass1/mvs/envs/mvs-engine/bin/python -B \
  /home/anas.khan/cv802_project/project1/ass1/mvs/scripts/clean_dark_person_crutches.py \
  --config /home/anas.khan/cv802_project/project1/ass1/mvs/configs/dark_e3_1024_raw_pycolmap.json \
  --derived-id dark-e3-mvs-body90-crutches-v1
```

Existing output folders are refused. Interrupted attempts keep their state and
artifacts for diagnosis; use a new derived ID rather than overwriting them.

### Actual cleaned result, 2026-09-19

The CPU-only cleanup completed in **65.783392 seconds** inside existing job
`264198`, without a new allocation or inference/reconstruction rerun:

| Selection | Points |
|---|---:|
| Original raw cloud, unchanged | 616,827 |
| Body-mask consensus | 387,069 |
| Protected crutch candidates | 72,395 |
| Protected candidates that body masks would remove | 68,112 |
| Final union, retained | **455,181** |
| Removed from this derivative | 161,646 |

The two protected regions retain 31,243 A and 41,152 B candidates. This agrees
with the earlier all-new-XYZ geometric review; no historical IDs were used.
The cleaned PLY has 12,290,121 bytes and SHA-256
`80664efaafbd2cd2605f40100c794cf03ad7827f16f3227d3c134f460d41dd51`.

All six new overlays and the unaligned, full-bounds orthographic preview were
visually inspected. The isolated person and both crutch shafts/forks are
clearly retained. Some gaps, oversized candidate bands and far-side crutch
projections over occluding body regions remain. This is approved as a
**separate labelled cleaned viewing result**, not an accuracy certification.
The preview displays 227,591 deterministically sampled points without clipping.

Publication evidence under `DATA_ROOT/mvs/outputs/dark-e3-mvs-body90-crutches-v1`:

```text
point_cloud.ply
cleanup_receipt.json          # Original computation receipt; never rewritten
visual_review.json            # Explicit six-overlay/preview observations
publication.json              # Complete, hashes and subset reverified
selection_audit.npz           # Per-point votes, decisions and source-row indices
projection_rows.json          # All 290 calibrated-to-native mask mappings
overlay-00.png ... overlay-05.png
state.json
```

`scripts/publish_dark_cleanup_review.py` is a separate CPU-only publication
check. After explicit visual approval, it rechecks raw-source hashes, PLY
identity, every selected vertex's bytes, pointwise body/union decision,
overlay/audit hashes and preview lineage before creating `publication.json`.
The original computation receipt intentionally retains
`validated_pending_visual_review`; publication is established by the separate
complete publication receipt and visual review, not by rewriting history.

The preview and its receipt are under
`DATA_ROOT/evaluation/preview-mvs-dark-clean-v1`. The full MVS suite passed
**32/32 tests in 9.146 s**, including three new protection/subset tests;
`DATA_ROOT/mvs/runtime/dark-cleanup-tests-20260919.json` records the actual
command, environment and result. The full-suite test run occurred after
cleanup implementation and before the separate publication helper was added;
the helper itself was exercised on the real completed derivative.

### Original crutch-review requirements

Recompute the approved capsules and multi-view corridor support on **new dense
XYZ**, in the verified world frame, never by transferring historical point IDs.
The original protection requires three exposed annotated views with at least
15-degree pairwise ray separation. Its sparse feature-track error/angle gates
cannot be claimed unchanged for dense points, which do not carry those tracks.
Define dense support/depth-visibility rules explicitly, project masks in their
correct distorted coordinates (or consistently transformed undistorted ones),
create new photo overlays, and review both crutches before publishing a cleaned
result. Historical overlay approval is not approval of a new dense model.

## Execution and live tracking

After preparation, the coordinator must confirm GPU availability within the
already-authorized job. This script never requests an allocation:

```bash
srun --jobid=264198 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --gres=gpu:1 --export=ALL \
  bash /home/anas.khan/cv802_project/project1/ass1/mvs/scripts/run_dark_shirt_pycolmap.sh
```

The outer process-tree sampler starts with the controller, so its sampled RSS
and NVIDIA per-process peaks are attributable to this command and descendants.
Evidence is under `DATA_ROOT/mvs/telemetry/dark_e3_colmap_mvs_1024_raw_v1-command-v1`.
The runner's older node-wide GPU fields must not override these process-scoped
measurements. Sampled peaks may miss brief spikes and RSS can double-count
shared pages. The telemetry command refuses to overwrite an existing run;
for an interrupted later resume, choose a new telemetry output identifier.

The first real launch failed before depth estimation because the unmasked
undistorter lacked its `work/` parent. The runner now creates that parent
explicitly; its regression test passes. Original failed metadata/logs are
preserved under `archive/failed-first-undistort-v1` and telemetry `command-v1`.
The corrected same-experiment resume uses the script argument `command-v2`:

```bash
bash /home/anas.khan/cv802_project/project1/ass1/mvs/scripts/run_dark_shirt_pycolmap.sh command-v2
```

This inner command still must run inside the existing GPU step shown above.

```bash
tail -F /l/users/anas.khan/cv_802_ass1/mvs/telemetry/dark_e3_colmap_mvs_1024_raw_v1-command-v2/command.log
```

Only a successful `manifests/final_validation.json` establishes completion.
Preparation alone is not a completed dense reconstruction.

After completion, `scripts/review_dark_crutches.py` scans every new dense XYZ,
rechecks that undistortion preserved the original world frame, computes bounded
capsule membership plus three-separated-corridor candidate support, and exports
six photo overlays into a new `DATA_ROOT/mvs/reviews/<review-id>` directory.
This is a read-only geometric plausibility review, **not dense cleanup**:
projections do not by themselves establish pixel visibility or true crutch
surface accuracy, and no old point ID or sparse-track quality metric is reused.

### Explaining the review in a viva

`read_dense_xyz` parses the PLY vertex schema and memory-maps every serialized
vertex, then extracts finite XYZ as float64. A row index identifies only that
new PLY row; it is never treated as an old COLMAP point ID. The model reload
checks identical registered names and unchanged camera extrinsics after
undistortion, establishing the coordinate frame for the historical capsules.

For capsule endpoints `a,b` and a new point `p`, `capsule_gate` computes
`t = clip(dot(p-a,b-a)/dot(b-a,b-a), 0, 1)` and tests the distance from `p` to
`a+t*(b-a)` against the saved radius. Capsules are bounded selection regions,
not reconstructed cylinders and not newly inserted points.

`confirmation_gate` projects each candidate through the **original distorted
camera**, matching the original photo/annotation coordinates. Only finite,
positive-depth, in-frame projections vote. A vote must fall inside an annotated
polyline corridor. For a confirming triple, every pair of normalized
point-to-camera rays must have dot product at most `cos(15 degrees)`, rejecting
three nearly identical viewpoints. This computes fresh geometric support on
new XYZ; it does not test depth-map occlusion or recreate missing metal-shaft
texture. Overlay colors identify candidates, not a certified segmentation.

The report preserves full point/candidate counts, source hashes, frame-check
differences, per-view evidence and overlay hashes. It never edits the fused PLY
or experiment receipts. A separate visual review is required after inspecting
the exported photo overlays; a generated `review.json` alone is not approval.
