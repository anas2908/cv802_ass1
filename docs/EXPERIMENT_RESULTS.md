# CV802 Project 1 experiment results

This document is the measured experiment report and evidence index. It intentionally
separates verified historical facts from server measurements copied from
generated receipts. The historical E10 result, fresh Linux/CUDA E11 result and
fixed-pose Linux portability validation E12 and dense MVS result below are
verified. E14 separately measures a fresh full-light CUDA run's resources.
Official full-light VGGSfM and its post-hoc camera alignment now have
completed, separately verified artifacts. Do not replace
a `PENDING` field with an expected value, a config value, or a value remembered
from terminal output.
E13 is separately reported only as a measured CPU smoke test with weak 2/16
registration, not as a full-capture result.

## Status legend

- **VERIFIED HISTORICAL:** validated before the server handoff and preserved.
- **VERIFIED COMPLETE:** a generated final receipt and its saved outputs passed
  the method's validation gates.
- **PENDING RECEIPT:** fill only after the named generated artifact validates.
- **FAILED:** retain the failure reason, attempt number and log path; do not
  present intended output as completed.
- **NOT RUN:** no reconstruction was attempted.

## Scope and inputs

The first server target is the light-shirt capture: 125 views comprising 91
still photographs and 34 selected video frames. All methods receive their own
physical input copy. The user subsequently requested separate MVS and VGGSfM
runs for the 290-view dark-shirt-with-crutches capture. Those extensions do not
replace the light-shirt validation or create a dark E10 result; their profiles,
results and current status are recorded separately.

The permanent roots are:

```text
CODE_ROOT=/home/anas.khan/cv802_project/project1/ass1
DATA_ROOT=/l/users/anas.khan/cv_802_ass1
```

## Historical reference: E10 light shirt

Status: **VERIFIED HISTORICAL**

| Field | Verified value |
|---|---:|
| Submitted/registered images | 125 / 125 |
| Calibration groups | 5 `SIMPLE_RADIAL` cameras |
| Unique exhaustive pairs offered | 7,750 |
| Saved cleaned colored points | 21,744 |
| Comparable E10 points in E4 comparison | 21,744 |
| Mean comparison reprojection error | 1.2224 E1 pixels |
| Occupied comparison voxels | 5,041 |
| Full reconstruction wall time | 4,299.237 s |

E10 reused E3 native features, fixed E1 poses, offered guided matching to all
unique pairs, retriangulated, and applied the established 90% mask-consensus
cleanup. The recorded worker sequence was 1 to 2 to 4, so the total wall time
is not a single-worker benchmark. The 90% value is mask agreement, not accuracy.

Historical sources:

- [E10 result explanation](../sfm/E10_RESULTS.md)
- [E1-E10 comparison table](../sfm/E1_E10_COMPARISON.md)
- [complete experiment record](../sfm/EXPERIMENTS.md)

The relocated machine-readable model and reports live under
`DATA_ROOT/sfm/historical/transferred/sfm`; this Git document does not duplicate
them.

The relocated E10 `analysis.json` independently records 28,924 source points,
21,744 retained points, 7,180 removed points, 125 registered images and 125
reliable mask views. Its serialized-subset validation says all retained
coordinates, colors and tracks were preserved exactly, every camera calibration
was preserved, and the maximum pose-element difference was zero. The companion
`provenance.json` records `pycolmap` 4.2.0 and the SHA-256 identities of the
source model files. These facts make E10 a verified historical reference; they
do not make it physical ground truth.

Machine-readable E10 evidence:

```text
DATA_ROOT/sfm/historical/transferred/sfm/reconstructions/experiments/
  E10_quality_exhaustive_guided_consensus90/light_shirt/analysis.json
DATA_ROOT/sfm/historical/transferred/sfm/reconstructions/experiments/
  E10_quality_exhaustive_guided_consensus90/light_shirt/provenance.json
DATA_ROOT/relocation/receipts/
historical-verify-20260919T155724.450069Z-578779-verification.json
```

The separate Linux read-only reload
`DATA_ROOT/sfm/e10-model-validation.json` used PyCOLMAP 4.2.0 and confirmed
125 registered names, five cameras and 21,744 points without reconstruction.
Its model-native mean reprojection error is **1.5086835047 px** and mean
track length is **5.1156640912**. This native-resolution model statistic is
not the 1.2224 **E1-pixel** comparison statistic above; the measurement frames
and report definitions must not be interchanged.

## Server execution identity

Status: **INITIAL WRAPPER INTERRUPTED; INDIVIDUAL RECOVERIES VERIFIED**

The original allocation is documented by
`DATA_ROOT/provenance/gpu-allocation-264198.json`; its terminal evidence is
`DATA_ROOT/runtime/jobs/264198/terminal-accounting.json`. No canonical
`engine-status.json` was emitted, and none was reconstructed or fabricated.

| Field | Actual value |
|---|---|
| Slurm job ID | `264198` |
| Node | `gpu-04` |
| Partition / QoS | `cscc-gpu-p` / `gpu-debug-qos` |
| Start / expiry | 2026-09-19 19:50:06 / 22:50:06 Asia/Dubai (15:50:06Z / 18:50:06Z) |
| GPU name / UUID | NVIDIA A100-SXM4-40GB / `GPU-cc1e9750-ccc1-0856-f782-92ad464796ea` |
| GPU memory | 40,960 MiB |
| NVIDIA driver | 570.195.03 |
| Slurm allocated / PyCOLMAP-visible devices | 1 / 1; `CUDA_VISIBLE_DEVICES=0` (raw host `nvidia-smi` listed four physical GPUs) |
| CPU / RAM allocation | 16 CPUs / 96 GiB |
| Final original orchestration status | FAILED, exit `75:0`, after USR1 advance-warning handler at 22:44:46 Dubai; original MVS rc255 and VGGSfM rc1 failures preserved, subsequent per-method recoveries verified separately |

The authentic stop receipt, captured active-wrapper identity and immutable
Slurm log prefix are bound by the terminal record. Slurm's retained `scontrol`
record supplies the terminal state; `sacct` was unavailable (database connection
refused), explicitly recorded as unavailable rather than invented accounting.

The user then authorized exactly one additional **one-hour** short allocation,
job **265977**, on the same `gpu-04`/A100-40GB device and short partition/QoS,
22:49:47–23:49:47 Dubai on 2026-09-19. Its receipt
`DATA_ROOT/provenance/gpu-allocation-265977.json` verifies active Slurm state,
the one-hour limit, real Lustre mount, a writable/readable storage probe and
407,060,284,416 bytes of user-quota headroom at the check. This continuation
does CPU cleanup, browser validation and tests, not reconstruction or relocation.
It does not overlap the previous CV802 allocation or use unrelated project jobs.
The subsequent scoped receipt
`DATA_ROOT/provenance/gpu-allocation-265977-scoped-v1.json` explicitly resolves
Slurm's physical GRES index and queries only GPU 0, UUID
`GPU-cc1e9750-ccc1-0856-f782-92ad464796ea`. It verifies 40,960 MiB and the same
storage gates, without overwriting the initial node-inventory receipt.

Do not infer successful GPU execution merely from an installed CUDA package.
The receipt must identify the visible allocated device and the method receipt
must prove that the actual stage completed.

## Storage publication and relocation

Status: **VERIFIED COMPLETE; CODE-ROOT POLICY AUDIT CLEAN**

Populate from `DATA_ROOT/relocation/receipts/`.

| Gate | Receipt | Result |
|---|---|---|
| Historical copy | `historical-copy-20260919T155138.071398Z-578779-copy.json` | Copied 28,850 files / 31,928 entries / 31,992,834,320 logical bytes; awaiting the separate verifier at this stage |
| Independent copy verification | `historical-verify-20260919T155724.450069Z-578779-verification.json` | VERIFIED: all paths, types, logical sizes and file SHA-256 values equal |
| Source extraction from mixed folders | `legacy-extract-20260919T155729.314958Z-578779-legacy.json` | VERIFIED: 140 selected source paths, with sizes and SHA-256 values equal |
| Per-method physical input copies | `method-inputs-20260919T155732.797644Z-578779-inputs.json` | VERIFIED: 125 images / 335,892,874 logical bytes per method; independent-inode checks passed |
| Verified heavy-home cleanup | `finalize-20260919T180155.647318Z-803724-finalization.json` | VERIFIED: 18 explicit duplicate targets removed after fresh source/destination checksums and active-writer checks; recovery copy retained |
| Final code-root audit | finalization receipt `post_cleanup_policy_violations: []`; fresh CLI audit after restoring tracked empty `.gitkeep` | PASS: zero violations; source, docs and Git history preserved |

Record counts, logical bytes and manifest/checksum identities from the receipts.
Do not claim cleanup succeeded solely because the data-root copy exists.

The first manual cleanup attempt stopped before deleting anything because
Lustre's allocated-block counts (`st_blocks`) changed after copying. A fresh
comparison proved identical paths, types, modes, modification times, logical
sizes and SHA-256 contents for all 125 input images in all three methods, with
independent inodes. The corrected verifier preserves original allocation
metadata as forensic evidence and records its drift separately; it still
rejects content/timestamp changes and shared physical files. The successful
retry rehashed both the 31,992,834,320-byte historical source and recovery copy,
then removed only the 18 frozen targets. The empty tracked dataset `.gitkeep`
was restored; the checked-out historical GIF remains only in the data root
and immutable Git history, as required by the storage policy.

Recovery is available from
`DATA_ROOT/sfm/historical/transferred`; 140 mixed-folder helper source paths
also remain under `CODE_ROOT/sfm/legacy_tools`. No source directory was deleted
without the prior extraction-and-verification evidence.

An independent capacity inspection found approximately 121 GiB allocated in
the data root before cleanup of any download cache. About 51 GiB was the MVS
Singularity download cache. Its largest cached SIF has 3,148,517,568 logical
bytes but 50,523,760,640 allocated bytes; it is byte-identical to the retained
3,148,517,568-byte dependency SIF. Cache reclamation was proposed separately
and awaits explicit approval; this verified **home** cleanup did not delete
the data-root cache, inputs, environments or results.

## Method 1: fresh Linux/CUDA SfM

### Intended immutable recipe

This subsection records the planned configuration, not successful execution.

| Field | Configuration |
|---|---|
| Experiment | `E11_linux_cuda_fresh_light_1600_v1` |
| Input | method-local copy of all 125 light-shirt images |
| Camera model / grouping | `SIMPLE_RADIAL` / `per_folder` |
| Matcher | exhaustive |
| Device | explicit CUDA |
| Longest image dimension | 1600 px |
| Maximum features per image | 8,192 |
| Threads / seed | 16 / 0 |
| Dependency | `pycolmap-cuda12==4.2.0` |

### Actual result

Status: **VERIFIED COMPLETE**

Evidence comes from
`DATA_ROOT/sfm/experiments/E11_linux_cuda_fresh_light_1600_v1/receipt.json`, its
attempt `status.json`, reloaded COLMAP model and validated PLY.

| Field | Actual value |
|---|---|
| Attempt / status | `0001` / complete; cache miss; no interrupted-attempt recovery |
| Input images hashed | 125 |
| Registered images | 125 / 125 |
| Cameras | 5 |
| Colored sparse points | 10,204 |
| Mean reprojection error | 1.3273668003 px |
| Mean track length | 5.5246961976 observations per point |
| Connected components | Count not recorded in the receipt; do not infer it |
| Selected component | Mapper component `0` |
| Wall time | 84.476150969 s, engine receipt scope; environment setup excluded |
| GPU identity / peak usage if sampled | A100-SXM4-40GB, UUID `GPU-cc1e9750-ccc1-0856-f782-92ad464796ea`; peak usage not sampled |
| COLMAP model hash identity | `cameras.bin` `b6987880a323ff0ff164bf04d82bf19f852b641ba84a3d89f8d3ea092e74ab0d`; `images.bin` `6af901840a8853361781530f959fdc136ed28dcc4b527ab15ede8f23048720f1`; `points3D.bin` `3c34cff05628cbdc72d96566691244317a15c1d0d569928599273ea4e1255a29` |
| PLY bytes / SHA-256 | 153,239 / `c48c7abb0123c22fe4f0f959337c58298f178526fa654973843c14d6e57d4f6a` |
| Reload validation | PASS: the published model reloads as 5 cameras, 125 images and 10,204 finite colored points; PLY header declares the same 10,204 vertices |
| Cache replay | PASS in 1.2915452132 s: cache hit; 16 artifacts unchanged by bytes, mtime and SHA-256; zero extraction, matching or incremental-mapping calls |
| Deterministic preview | 10,204 / 10,204 points displayed with no clipping and no coordinate transform; PNG 72,916 bytes, SHA-256 `d84acb04d3b6201ba076f8d8d5a37cbbc521e75de6561b098563c848977978ea` |

Interpretation:

- coverage is 125 / 125 submitted views (100%);
- physical plausibility of the fresh camera trajectory is not independently
  established by the saved-point preview;
- the deterministic three-projection preview displays all 10,204 points and
  visibly shows sparse full-scene/background and scattered coverage, but does
  not establish accuracy or isolate subject completeness; and
- an important difference from E10 is that E11 estimates cameras from scratch at the
  listed server settings, while E10 is a fixed-E1-pose quality refinement.

The original E11 receipt did **not** sample peak RAM or VRAM. The listed 40,960
MiB is the selected GPU's capacity, not observed usage, and must not be reported
as a peak. Do not substitute a measurement from a later cache check or another
method. See [resource measurement interpretation](RESOURCE_MEASUREMENTS.md).

The E11 receipt is:

```text
DATA_ROOT/sfm/experiments/E11_linux_cuda_fresh_light_1600_v1/receipt.json
```

Its request/engine identity is
`6f66fa18242900377ec4bbf7e98b43ce9cb1eb910c9d490ed8bcb07daaa58a3f`.
The attempt status completed at `2026-09-19T16:08:13Z`. Point totals must not be
ranked directly against E10: E11 uses a 1600-pixel cap and fresh poses, whereas
E10 uses native-resolution quality features, fixed E1 poses and a later 90%
mask-consensus cleanup.

The cache proof and qualitative-preview evidence are:

```text
DATA_ROOT/sfm/e11-cache-check-v1.json
DATA_ROOT/evaluation/preview-e11-light-v1/preview.png
DATA_ROOT/evaluation/preview-e11-light-v1/preview_receipt.json
```

The preview receipt explicitly says that it is qualitative only and does not
establish geometric accuracy. It uses model coordinates with no alignment
implied.

## Fixed-pose Linux portability validation: E12

Status: **VERIFIED COMPLETE**

E12 is a portability/reproducibility validation of the historical E3-style
fixed-pose branch, not a fourth scientific reconstruction method and not an
alias for E11. It uses the published E3 `light_shirt_quality` images and matched
database, verifies the features byte-for-byte against the E3 affine/DSP
checkpoint, scales the original E1 light-shirt calibration to the native
quality-image dimensions, fixes those poses/intrinsics, and performs fresh
triangulation only. It is not E10 and applies no rectangle, mask or person
cleanup.

| Field | Actual value |
|---|---|
| Experiment | `E12_linux_fixed_pose_light_e3_v1` |
| Source checkpoint accounting | 125 images; 1,741,593 features; 1,373 nonempty match pairs; 1,344 nonempty two-view geometries |
| Feature extraction / matching | reused unchanged; no extraction or new matching |
| Device / threads / Slurm job | CPU / 4 / `264198` |
| Fixed-camera controls | existing frames fixed; focal length, principal point, extra parameters and sensor-from-rig refinement all disabled; GPU bundle adjustment disabled |
| Triangulation controls | random seed 0; minimum angle 1.5 degrees; maximum filter reprojection error 3.5 px; two-view-only tracks ignored; colors extracted |
| Registered images / cameras | 125 / 5 |
| Colored sparse points | 41,675 |
| Mean reprojection error | 1.3721696057 px |
| Mean track length | 4.0993881224 observations per point |
| Fixed-calibration validation | PASS: intrinsics and extrinsics fixed; maximum absolute difference `6.661338147750939e-16`, below tolerance `1e-12` |
| Source immutability | PASS: feature/match tables preserved and transferred source unchanged |
| Measured wall time | 16.090282443 s; starts after initial hashing, image decoding and database validation, and includes private database copy, triangulation, color extraction, saving, final validation and rehash |
| PLY vertices / bytes | 41,675 / 625,304 |
| PLY SHA-256 | `7989f54870d02078a6bfd2d294eb2f3b84da84c64e2e27d7cc813e665e1a4e7b` |
| Recipe identity | `e40a96916377c259bebbbfa582627802098a7c374297cec1d5125113ad19eb95` |
| Cache replay | PASS: 14 files verified unchanged, zero triangulation calls, experiment files unchanged |
| Deterministic preview | 41,675 / 41,675 points displayed with no clipping and no coordinate transform; PNG 79,059 bytes, SHA-256 `1366776d7fce510788d9165c02bcb618222f7fbf8301c91c581f3f15b2937377` |

The camera-invariance test proves that the saved calibration was not optimized
away; it does not prove physical accuracy. Likewise, E12's larger point count
than E11 reflects different features, pair data, camera policy and image scale,
not a controlled accuracy ranking. E12's preview is a full-scene sparse cloud,
not the later E3 rectangle or mask cleanup. Its preview receipt is explicitly
qualitative and uses the saved model coordinates without alignment.

Machine-readable evidence:

```text
DATA_ROOT/sfm/experiments/E12_linux_fixed_pose_light_e3_v1/receipt.json
DATA_ROOT/sfm/fixed-pose-E12-cache-check-v2.json
DATA_ROOT/evaluation/preview-e12-light-v1/preview.png
DATA_ROOT/evaluation/preview-e12-light-v1/preview_receipt.json
```

## CPU execution and telemetry smoke test: E13

Status: **VERIFIED COMPLETE SMOKE TEST; NOT A FULL-SHIRT RESULT**

E13 exercised the fresh-camera headless SfM engine on CPU and measured the
complete command lifecycle. Its 16 inputs are checksum-verified physical copies
at evenly spaced indices in the preserved 125-image manifest, with endpoints
included and capture subfolders retained. It is deliberately a small execution
pilot, not a substitute for E11 and not evidence of good full-shirt coverage.

| Field | Actual value |
|---|---|
| Experiment | `E13_linux_cpu_smoke_light16_1024_v1` |
| Input selection | 16 / 125 evenly spaced manifest images; physical-copy hashes verified |
| Recipe | CPU, exhaustive matcher, `SIMPLE_RADIAL`, per-folder cameras, 1024-pixel cap, 4,096 features, 2 threads, seed 0 |
| Registered coverage | 2 / 16 images; weak reconstruction |
| Cameras / colored points | 2 / 40 |
| Mean reprojection error / track length | 0.6014058110 px / 2.0 observations per point |
| Engine time | 11.781557339 s |
| Observed command lifecycle | 12.719813824 s |
| Resource sampling | 6 samples at 2-second intervals; complete command lifecycle |
| Peak sampled process-tree RSS | 306,668 KiB (299.48 MiB) |
| Peak sampled process-tree GPU memory | 0 MiB; CPU execution |
| PLY bytes / SHA-256 | 776 / `b1d926212fed838300253660e5276f5e341298ab0da4b61c33fb78cc8ad88541` |

The low 2/16 registration is a negative quality result and must be stated, not
hidden by the successful exit code. The telemetry receipt also warns that
two-second sampling can miss brief spikes and summed process-tree RSS can
double-count shared pages. This measured E13 peak must not be retroactively
reported as E11 usage.

```text
DATA_ROOT/sfm/experiments/E13_linux_cpu_smoke_light16_1024_v1/receipt.json
DATA_ROOT/sfm/telemetry/E13_linux_cpu_smoke_light16_1024_v1/receipt.json
DATA_ROOT/sfm/inputs/light_shirt_cpu_smoke16/copy_provenance.json
```

## Measured full-light CUDA SfM validation: E14

Status: **VERIFIED COMPLETE**

E14 is a new full-light run to measure resources missing from the original E11
receipt. E11 is preserved. E14 uses the same 1600-pixel/8,192-feature,
exhaustive, fresh-camera CUDA recipe but four CPU threads rather than sixteen.
Dark MVS shared this project's allocated GPU during the measurement; this is
not an uncontended speed comparison or a replacement for E11's timing.

| Field | Actual value |
|---|---|
| Experiment | `E14_linux_cuda_measured_light_1600_v1` |
| Input / registered images | 125 / 125 |
| Cameras / RGB points | 5 / 10,090 |
| Mean reprojection error / track length | 1.3316889441 px / 5.5374628345 observations |
| Engine / complete command wall time | 248.834432691 / 251.279022455 s |
| Sampled peak process-tree GPU / summed RSS | 836 MiB / 398,464 KiB (389.125 MiB) |
| Sampling scope | 118 samples, nominal two-second interval, complete command lifecycle, exit 0 |
| PLY bytes / SHA-256 | 151,529 / `6aaacab8ece92b7d39945da4f24349f6247b67b3592eb8769b529a489f63c543` |
| Read-only cache replay | PASS: 16 files unchanged by bytes, SHA-256 and mtime; zero extraction, matching or mapping calls; 2.007211 s |

Evidence is `DATA_ROOT/sfm/experiments/E14_linux_cuda_measured_light_1600_v1/receipt.json`
and `DATA_ROOT/sfm/telemetry/E14_linux_cuda_measured_light_1600_v1/receipt.json`.
The separate cache proof is `DATA_ROOT/sfm/e14-cache-check-v1.json`.
Sampled peaks can miss brief spikes; RSS can double-count shared pages. CUDA
here accelerates SIFT extraction/matching, not every mapping/optimization step.
Do not retroactively assign these peaks to E11.

## Read-only historical E1-E10 browser validation

Status: **VERIFIED COMPLETE**

The historical browser is a viewer, not another reconstruction method. Its
production validator checked all 19 available experiment/subject PLYs against
the catalog's documented vertex counts; E10 dark was correctly exposed as
not run rather than as a missing result. A final real Chromium sweep then
selected and rendered all 19 clouds with SwiftShader software WebGL. Every
selection reported WebGL error code zero and non-background rendered pixels;
rotation/reset passed and the page-error list was empty. `CUDA_VISIBLE_DEVICES`
was null, so this validation did not consume a GPU.

The final screenshots of E10 light, E4 dark and the compact mobile layout were
visually inspected after the display-only axis correction. That correction
changes only the viewer orientation; it never rewrites a preserved PLY. The
receipt and screenshots are:

```text
DATA_ROOT/sfm/logs/browser-checks/20260919T165700Z/receipt.json
DATA_ROOT/sfm/logs/browser-checks/20260919T165700Z/E10-light.png
DATA_ROOT/sfm/logs/browser-checks/20260919T165700Z/E4-dark.png
DATA_ROOT/sfm/logs/browser-checks/20260919T165700Z/mobile-layout.png
```

During the active allocation the viewer ran loopback-only on `gpu-04`; a second
loopback SSH tunnel in tmux window `sfm-tunnel` exposed it only at port 8765 on
the login node. From the user's Mac, the verified final hop was:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8765:127.0.0.1:8765 anas.khan@10.127.79.236
```

The browser URL was `http://127.0.0.1:8765`. This is ephemeral access evidence,
not a promise that the service remains live after Slurm job 264198 ends.

The user's preferred lightweight layout is retained. Its camera-distance
range was expanded from 1.25–12 to 0.03–120 and the near clipping plane reduced
to 0.002, with reset unchanged. The later real-browser receipt
`DATA_ROOT/sfm/logs/browser-checks/final-mvs-v1/receipt.json` verifies all 19
historical clouds plus the 487,451-point light MVS result, actual wheel zoom
beyond both old limits, reset, and no WebGL/page errors. Dark MVS is only
enabled once its final result exists and validates.

The subsequent `final-vgg-light-v1/receipt.json` in the same browser-checks
directory verifies **12 experiment options and 21 available clouds**, adding
the exact 414,208-point full raw `VGGSFM/light` output. The route is restricted
to the configured data-root PLY and a completed manifest; inference is not
called. Actual software-WebGL rendering, expanded wheel zoom, rotation and
reset passed with no page/WebGL errors. Its `VGGSFM-light.png` was visually
inspected: the expected scattered full-scene geometry is shown honestly, not
replaced with a cleaned-person subset. The compute endpoint and login-node
SSH relay both returned HTTP 200 for the live catalog and validation route.

The later `final-vgg-clean-light-v2/receipt.json` adds the separately labelled
**VGGCLEAN** derivative: 13 options, 22 rendered available clouds, and the exact
138,677-point clean result with no page/WebGL errors. Root independently
checked HTTP 200 and the manifest-bound count through the login relay and
inspected the upright `VGGCLEAN-light.png`. The original raw entry remains.

## Method 2: dense COLMAP PatchMatch MVS

### Intended immutable recipe

| Field | Configuration |
|---|---|
| Experiment | `light_e10_colmap_mvs_1600` |
| Initialization | copied E10 light cameras/model |
| Input references | all 125 registered images |
| Foreground preprocessing | threshold 128, three-pixel dilation, black background before undistortion |
| Longest image dimension | 1600 px |
| Source views per reference | automatic top 10 |
| PatchMatch | 5 iterations, 15 samples, radius 5, step 1, min 2 consistent |
| Consistency | photometric plus geometric |
| Fusion | min 5 pixels, max 2 px reprojection, 0.01 relative depth, 10 degree normal error |
| Backend | executed `pycolmap-cuda12==4.2.0` dense APIs via `light_e10_1600_pycolmap.json`; explicit fallback after the initial SIF mount failure |
| Cache / threads | 16 GB PatchMatch and fusion caches / 16 threads |
| Meshing | disabled for primary result |

### Actual result

Status: **VERIFIED COMPLETE; CPU-ONLY READ-ONLY RELOAD PASSED**

The original controller completed at `2026-09-19T17:18:18+00:00` (21:18:18
Asia/Dubai). Evidence is under
`DATA_ROOT/mvs/experiments/light_e10_colmap_mvs_1600/`: the four
`receipts/*.json`, `manifests/input_validation.json`,
`manifests/runtime_validation.json`, `manifests/final_validation.json`, and
`run_state.json`. The independently revalidated output is `outputs/fused.ply`.
The executed configuration digest is
`547e1f8554901645aad4e1484e18373b407a7c265a2c0a587eed8eaf0624124d`.

| Field | Actual value |
|---|---|
| Staged image/model/mask verification | 125 images, 5 cameras, 125 masks, finite calibration/poses, no input symlinks; input-provenance digest `676704726fe6c2df362a6aa814394a3f9e9994efed3bc6dfdec16964a752cce7` |
| Original dimensions | 91 images at 3024×4032; 34 at 1080×1920; dense cap 1600 |
| Runtime/backend preflight | PyCOLMAP 4.2.0, CUDA and all three dense APIs verified; `gpu-04`, job 264198, A100-SXM4-40GB, visible device 0; no container used by the successful run |
| Undistorted reference views | 125 / 125 |
| Completed geometric depth/normal maps | 125 depth maps / 125 normal maps; 894,721,466 / 2,684,161,466 bytes |
| Fused colored vertices | 487,451 |
| Finite XYZ / valid RGB scan | PASS; every exported vertex scanned |
| 3D bounds | min `[-0.8970543742, -1.1223711967, -0.0315687656]`; max `[0.7843689919, 3.9136238098, 2.6455316544]`, E10 arbitrary units |
| Undistortion time | 16.2029558411 s |
| PatchMatch time | 3,763.2387794498 s |
| Fusion time | 51.6665056916 s |
| Successful-controller elapsed time | 3,834.5104364119 s (~63.91 min); scope caveat below |
| Attributable observed resource sample | PatchMatch tail only: peak process-tree RSS 4,602,696 KiB; GPU memory 990 MiB; 604 samples over 1,250.153 s; not a full-run peak |
| Original stage GPU peak fields | Node-wide 17,137 MiB / up to 100% utilization; **not attributable to MVS** |
| `fused.ply` bytes / SHA-256 | 13,161,411 bytes / `151f4b2cdcf9cbc590740267c5b726d27492f0f0bc36ae25d5ced4f63da8746f` |
| Mesh | Not generated (`meshing=none`) |
| Resume/cache behavior exercised | All four stage receipts revalidated; original final report/time preserved; no executor, CUDA-probe or metadata-write calls |

The 3,834.510-second figure is the successful controller's final-receipt scope,
not installation-to-completion time. Its mask-input stage was reused from the
earlier verified 153.150674-second mask stage, so that prior work and the failed
container/retry history are outside this timer. Stage `child_peak_rss_kib`
values are cumulative child-process high-water marks, not isolated stage peaks.
The original GPU monitor saw all physical cards; preserve its receipts but do
not treat their 17,137 MiB readings as MVS usage. The attributable partial-run
sample is
`telemetry/patchmatch-process-tail-v1/receipt.json`; see
[`RESOURCE_MEASUREMENTS.md`](RESOURCE_MEASUREMENTS.md).

The instrumented reload receipt is
`DATA_ROOT/mvs/runtime/cache-validation/light1600-readonly-v1.json`. A CPU-only
step in the same allocation revalidated the full PLY, inputs and every stage
without probing CUDA or invoking COLMAP. It preserved 1,038 files' metadata and
14 JSON hashes, returned the original final report plus `cache_hit: true`, and
finished validation in 3.208165 seconds. That cache-check time is **not** the
reconstruction runtime. The completed-run branch rejects missing/corrupt
receipts or artifacts instead of silently reconstructing.

The numerical result contains more samples than the sparse references, but
point count alone does not prove better geometry or complete body coverage.
Qualitative inspection remains separate. Cameras were inherited from E10,
and black-background preprocessing is not the historical 90% sparse-point
consensus rule.

## Dark-shirt MVS extension (separate from light)

Status: **VERIFIED COMPLETE; CPU-ONLY READ-ONLY RELOAD PASSED**

The user-requested dark run is
`DATA_ROOT/mvs/experiments/dark_e3_colmap_mvs_1024_raw_v1`. It uses a physically
copied E3 unfiltered sparse model in the inherited E1 frame, not a nonexistent
dark E10 result. All 290 views are used, with four camera groups. The raw,
unmasked 1024-pixel profile uses five source views and three PatchMatch
iterations, unlike the masked light profile's 1600/ten/five settings. Body
masks were deliberately not applied because they can remove crutch shafts.

| Field | Actual value |
|---|---|
| Completion | 2026-09-19 22:28:42 Asia/Dubai (final validation); outer command exit 0 |
| Reference views / geometric depth maps / normal maps | 290 / 290 / 290 |
| Finite colored fused points | 616,827 |
| PLY bytes / SHA-256 | 16,654,563 / `6a61c61382692de34a6a6836f6d6e5ccf2354679b68dd885690dc7b660e6199c` |
| Controller / outer-command wall time | 3,392.944895 / 3,395.090494 s |
| Undistortion / PatchMatch / fusion | 101.416439 / 3,186.888232 / 99.559768 s |
| Full-lifecycle process-scoped samples | 1,641; sampled peak GPU 584 MiB; summed RSS 4,646,764 KiB |
| Read-only cache proof | 5.387587 s; 2,080 file-metadata records and 21 JSON hashes unchanged; zero executor, runtime-probe or metadata-write calls |

The final validation receipt is `manifests/final_validation.json` in that
experiment. Attributable telemetry is
`DATA_ROOT/mvs/telemetry/dark_e3_colmap_mvs_1024_raw_v1-command-v2/receipt.json`;
the separate cache receipt is
`DATA_ROOT/mvs/runtime/cache-validation/dark1024-readonly-v1.json`.
Stage-level GPU-memory counters include other processes sharing this project's
assigned card and are **not** this MVS process's usage; use the process-tree
sampler above. Concurrent VGGSfM and overlapping CPU placement also make this
an operational runtime, not an uncontended benchmark.

The original missing-parent undistortion failure is retained under
`archive/failed-first-undistort-v1` and telemetry `command-v1`; the successful
retry did not overwrite that evidence. Raw background is expected. New-geometry
crutch-capsule/corridor overlays and their separate visual review are explained
in [the dark MVS guide](../mvs/DARK_SHIRT.md); old sparse point IDs are never
treated as identities of new dense points. A raw output does not imply perfect
crutch coverage or certify thin-structure accuracy.

The subsequent full-bounds preview is
`DATA_ROOT/evaluation/preview-mvs-dark-raw-v1/preview.png`, displaying 205,609
deterministic samples from 616,827 points. It visibly retains the person/crutch
structure together with substantial room/background points. Six new-geometry
overlays and `visual_review.json` under
`DATA_ROOT/mvs/reviews/dark-e3-1024-crutch-projections-v1/` were inspected:
candidate bands broadly follow both shafts/forks, but are thicker than the
metal, contain gaps, and project onto body regions where far crutches are
occluded. Capsule/corridor candidates number 31,243 for A and 41,152 for B;
these counts are **not** certified crutch surface points or completeness
percentages. Only raw publication was accepted, with no 90%/97% dense cleanup.

The browser receipt
`DATA_ROOT/sfm/logs/browser-checks/final-dark-mvs-vgg-clean-v1/receipt.json`
verifies 13 options and all 23 then-available clouds, including the exact
616,827-point `MVS/dark` result, raw light VGGSfM and cleaned light VGGSfM.
Actual software-WebGL rendering, deep zoom, rotation/reset and page-error
checks passed. The optional already-open native viewer was not restarted;
the primary browser contains the updated catalog.

## Method 3: official VGGSfM v2

### Intended immutable recipe

| Field | Configuration |
|---|---|
| Run ID | `light-shirt-vggsfm-v1` |
| Input | independent method-local raw-image copy; no E10 poses/model |
| Official source | commit recorded in `vggsfm/PROVENANCE.md` |
| Official model | `facebook/VGGSfM`, `vggsfm_v2_0_0.bin` |
| Camera model | `SIMPLE_RADIAL`, `shared_camera=false` (independent per-view intrinsics) |
| Query/tracker | 6 query frames, 2,048 ALIKED points, fine tracking |
| Image size / precision | 1024 / FP16 |
| Robust refine / BA iterations | 2 / 1 |
| Extra points | ten-pixel grid, 16 temporal neighbours, concatenated |
| Random seed | 0 |

### Measured 16-view pilot and publication recovery

The independent, evenly sampled pilot used 16 of the 125 input views. Official
inference succeeded in **122.063 s** and registered **16/16 views**, with 16
cameras and **33,696 points**: 5,661 tracked points and 28,035 additional
trackless grid points. Its 60 process-scoped samples observed peaks of
**26,550 MiB GPU memory** and **2,905,336 KiB summed RSS**. These sampled pilot
peaks do not describe the full 125-view run.

The wrapper initially rejected legitimate zero-based camera ID 0 after
successful inference. The validator was corrected and regression-tested; the
original failed receipt, official log and binaries remain unchanged. A
separate CPU-only review independently reloaded the saved model with
PyCOLMAP 3.10.0 and exported its RGB PLY without rerunning inference:

`DATA_ROOT/vggsfm/experiments/light-shirt-vggsfm-pilot16-v1/review-v1/review.json`

Upstream stored `-1` as an uncomputed reprojection-error sentinel for this
model. The original review's raw `-1` fields are **not measured pixel errors**.
The measured mean tracked-point length is 5.7461579226 observations. The
unfiltered three-projection preview at
`DATA_ROOT/evaluation/preview-vggsfm-pilot16-light-v1/preview.png` shows the
subject small within substantial scattered/full-scene geometry; it is neither
an aligned comparison nor evidence of dense anatomical completeness.

### Actual full-capture result

Status: **VERIFIED COMPLETE; CPU-ONLY READ-ONLY RELOAD PASSED**

Populate from `DATA_ROOT/vggsfm/install_receipt.json`, the immutable
experiment `request.json`, attempt `receipt.json`, `official.log` and
`outputs/light-shirt-vggsfm-v1/manifest.json`.

| Field | Actual value |
|---|---|
| Installed official / LightGlue Git SHAs | `e1d9d2eb2b3575525792206fb94b2c749c58dc50` / `2f23ca2ea9638cecad7f7220795210fc6b8353c3`, from `install_receipt.json` |
| Checkpoint SHA-256 | `e14f2c292686f078ae45bb323c562acc1e4e5fe5d3f940480c565b72af822778`, from `checkpoint_receipt.json`; path under method-local HF cache |
| Input images hashed | 125, from immutable `request.json` |
| Attempt / status | `0001`, complete; 2026-09-19 21:25:32 to 21:51:09 Asia/Dubai (wrapper scope) |
| Registered images | 125 / 125 |
| Cameras | 125 independent `SIMPLE_RADIAL` calibrations |
| Colored points | 414,208: 11,362 tracked, 402,846 trackless grid additions |
| Linked observations / tracked mean length | 438,655 / 38.6071994367 observations per tracked point |
| Original stored reprojection errors | `-1` means uncomputed, not negative pixel error; see the separate coordinate-consistent export review |
| Inference wall time | 1,519.284002084 s (~25.32 min), official subprocess scope; installation/input staging/export validation excluded |
| GPU identity / sampled resource peaks | allocated A100-SXM4-40GB; process-tree GPU 32,756 MiB, summed RSS 13,573,444 KiB; 738 samples at nominal two-second intervals |
| Native COLMAP model hashes | cameras `5243d879a8e4adc1309459c9561e0aceb6e5f5aa72f43da6c71f9d5162e8f8b1`; images `8a83f28a0f905a26f85459a6d617d52303aa28a9097f1e97410d208e9665da6c`; points `0062dafd3dd3bd175e771ff467406acda0d3a1ee3f9bb516a3ed98d861949515` |
| PLY bytes / SHA-256 | 6,213,355 / `a74cc2381f8177e2067fbb62bbe615c042ab45863c5380a9ab46e81aa7261a87` |
| Independence check: no external poses | PASS: immutable request and final manifest explicitly record raw inputs, no E10/external poses, alignment only after reconstruction |
| Read-only cache replay | PASS: 139 experiment/output files unchanged by bytes, SHA-256 and mtime; zero inference, runtime-probe and metadata-write calls; 0.590319 s |

Interpretation:

- registered coverage is 125/125 submitted images;
- per-view calibration physical plausibility is not independently established;
  unlike the initial
  draft, the actual reviewed profile does not force shared intrinsics;
- the full-bounds raw preview has substantial scattered/radiating geometry;
  it does not show a clean isolated person, despite the high point count; and
- extra grid points are triangulated additions, not pixel-wise depth MVS and
  not part of the earlier joint bundle adjustment.

The cache receipt is `DATA_ROOT/vggsfm/cache-check-light-shirt-vggsfm-v1.json`.
The immutable output manifest records all three weight hashes, the official
and LightGlue commits, and a content-hashed DINO Torch Hub source snapshot.
The raw preview is `DATA_ROOT/evaluation/preview-vggsfm-light-v1/preview.png`:
207,104 deterministic samples from 414,208 points, with no coordinate clipping.

Upstream's default export rescales intrinsics to original image resolution but
leaves saved 2-D observations in padded inference coordinates. A separate
coordinate-consistent COLMAP export is therefore reviewed without rerunning
inference or changing the original output. Do not interpret the raw export's
2-D residuals as original-image-pixel errors. This export issue does not
alter saved camera poses or XYZ/RGB, so the camera-center alignment below is
independent of the correction. The derived-export receipt is separately
verified; no original publication is overwritten. Its canonical path is
`DATA_ROOT/vggsfm/outputs/light-shirt-vggsfm-v1-colmap-consistent-v1/review_export_receipt.json`.
Both the binary validator and an independent PyCOLMAP 3.10 reload passed.
All 438,655 observations were transformed using the pinned float32 formula;
mean true residual changed from 1,219.277109 (mismatched frames) to **5.628313
original-image pixels**, with RMSE **6.768393**. The 2,396 finite observations
outside image bounds are preserved padding tracks, explicitly counted across
59 images. Nothing was clamped or filtered. All poses, intrinsics, XYZ/RGB,
identities and tracks are unchanged; the PLY remains byte-identical.

A separate **tracked-only diagnostic** selects precisely the 11,362 existing
points whose COLMAP track length is greater than zero. It performs no new
triangulation, refinement, XYZ/RGB changes or person filtering. The full
414,208-point official publication remains the primary result. Its separate
folder is
`DATA_ROOT/vggsfm/outputs/light-shirt-vggsfm-v1-tracked-only-review-v1/`;
the selection receipt verifies the float32 XYZ/RGB records and unique point
IDs. The 170,678-byte PLY has SHA-256
`6322e57d071e0b8ea37f2c434dab49f89c11840b2edf106ef0471d7a32bb68b9`.
All 11,362 points are shown without clipping in
`DATA_ROOT/evaluation/preview-vggsfm-light-tracked-only-v1/preview.png`.
Visual inspection finds more concentrated, partly recognizable torso/arm
structure than the full-grid preview, but still substantial background,
outliers and incomplete coverage. This diagnostic isolates the effect of
including trackless additions; it does not prove a more accurate reconstruction.

### User-requested person-only VGGSfM cleanup

Status: **VERIFIED DERIVED OUTPUT; RAW RECONSTRUCTION PRESERVED**

`vggsfm/scripts/clean_light_with_masks.py` creates a separate filtered PLY at
`DATA_ROOT/vggsfm/outputs/light-shirt-vggsfm-v1-mask-projection-clean-v1/`.
It does not rerun inference, triangulation or bundle adjustment. All 125 saved
native-resolution person masks were physically copied into the VGGSfM method's
own input folder, with exact filename coverage, dimensions, polarity and
SHA-256 verification. The cameras are **VGGSfM's independently estimated
cameras**, not E10 poses; the coordinate-consistent model preserves those
original intrinsics and poses.

For every raw point, the filter projects into each registered camera with
positive camera depth and finite coordinates, applies the camera's radial
distortion, and samples the matching native-size mask at the nearest in-bounds
pixel. A mask value at least 128 votes foreground. The selected profile requires
at least six usable projections and at least 90% foreground agreement. This is
projection-based consensus, not an occlusion test or a guarantee of true
visibility, especially for the trackless extra points. It must not be called
90% reconstruction accuracy or the historical sparse-track filter unchanged.

| Field | Actual value |
|---|---|
| Input / retained / removed | 414,208 / 138,677 / 275,531 |
| Retained tracked / trackless points | 4,026 / 134,651 |
| PLY bytes / SHA-256 | 2,080,414 / `ca1c35a7dec46aaa7aeb614c3ce79917dbb6b095222a141120cf7ade8412a04f` |
| XYZ/RGB preservation | Every selected 15-byte float32 XYZ/uint8 RGB record copied byte-identically from the original raw PLY |
| Original publication | Before/after source hashes equal; raw full PLY and COLMAP model not replaced |
| CPU-only inspection | All 138,677 points rendered, no clipping or coordinate transform |

The primary cleanup receipt is `cleanup_receipt.json` beside that PLY.
The qualitative preview and receipt are under
`DATA_ROOT/evaluation/preview-vggsfm-light-clean-v1/`. The PNG is 575,352 bytes,
SHA-256 `fa38a4ddd09973e1f8c59c28e5f622ada0b42d595d39c1b337f9f53d874b0202`.
Inspection shows a clearly isolated person, with head/hair, shirt lettering,
trousers and feet recognizable; the large radiating background is removed.
Some coverage gaps remain, and neither completeness nor physical accuracy is
established. Filtering removes inconsistent samples; it does not repair the
underlying geometry or invent missing surfaces. The threshold alternatives
are retained as support statistics, not silently substituted results.

A subsequent CPU-only audit rehashed the raw manifest's declared files,
the corrected-camera model, and all 125 masks, recomputed every projection
vote and regenerated the selected PLY. The regenerated bytes exactly match
the published 2,080,414-byte PLY, and all source files remain unchanged.
The per-image projection table is stored alongside its digest inside
`DATA_ROOT/vggsfm/evaluation/light-shirt-vggsfm-v1-mask-projection-clean-v1-validation-v1/audit_receipt.json`.
This supplemental audit does not overwrite the original cleanup receipt.

## Dark MVS cleanup: separately published derivative

`DATA_ROOT/mvs/outputs/dark-e3-mvs-body90-crutches-v1/publication.json`
records **455,181 retained / 616,827 raw points**, with 161,646 removed.
The CPU selection took 65.783392 seconds. Body-mask support selects 387,069
points; independently protected crutch candidates total 72,395 (A: 31,243;
B: 41,152). Their union retains **68,112** candidates that body consensus alone
would reject. The output PLY is 12,290,121 bytes, SHA-256
`80664efaafbd2cd2605f40100c794cf03ad7827f16f3227d3c134f460d41dd51`.

The body rule requires at least six usable projections and 90% foreground
agreement, using the original distorted E3 cameras in the verified MVS frame.
Protection requires bounded 3-D capsule membership plus at least three
annotated 2-D corridor views with separated viewing rays. All decisions are
recomputed for these dense XYZ values; old sparse point IDs are not reused.
Each output vertex preserves the exact source XYZ/RGB/normal bytes, and
publication reverified the pointwise body/protection union and source hashes.

The point-cloud preview and six camera overlays were reviewed and accepted
**with limitations**: both crutch candidate structures remain, but bands can
be thicker than metal, contain gaps, or overlap body pixels under occlusion.
This cleanup removes background; it does not complete surfaces or certify
thin-structure accuracy. The raw output is unchanged. The original
`cleanup_receipt.json` preserves `validated_pending_visual_review`; completion
is instead represented by the later immutable `publication.json` and
`visual_review.json`. Browser publication is a separate validation step.

## Dark-shirt VGGSfM extension (raw)

Status: **VERIFIED COMPLETE; CPU-ONLY READ-ONLY RELOAD PASSED**

`dark-shirt-vggsfm-all290-a10040-fit-v2` independently processed all 290 raw
dark-shirt images and registered **290/290**, with 290 per-view `SIMPLE_RADIAL`
cameras. No baseline camera poses or person masks were supplied to inference.
The explicit fit profile is 640 pixels, four query frames, 1,536 ALIKED query
points, fine FP16 tracking, two robust refinements, one BA iteration, and a
16-pixel extra grid with 16-frame neighborhoods. These settings differ from
light and must not be presented as an equal-quality recipe.

| Field | Actual value |
|---|---|
| Raw RGB points | 187,308 = 6,061 tracked + 181,247 trackless extras |
| Mean tracked-point length | 85.6409833361 observations |
| Official subprocess time | 1,995.068266801 s; exit 0 |
| Wrapper start / complete | 22:06:52 / 22:40:19 Asia/Dubai, 2026-09-19 |
| Process-scoped observed peaks | 38,486 MiB GPU; 20,727,036 KiB summed RSS; 916 samples |
| PLY bytes / SHA-256 | 2,809,855 / `43160ffaaeecff056f8e1d1f856d4b1ac6b3a094f54442ec99e0ae96e2f2359d` |
| Read-only cache replay | 304 files unchanged by bytes, SHA-256 and mtime; zero inference, runtime-probe or metadata-write calls; 0.770423 s |

The output is
`DATA_ROOT/vggsfm/outputs/dark-shirt-vggsfm-all290-a10040-fit-v2/`, with
`manifest.json`, native COLMAP binaries and `point_cloud.ply`.
The cache proof is `DATA_ROOT/vggsfm/cache-check-dark-shirt-vggsfm-all290-a10040-fit-v2.json`.
The full-bounds preview at `DATA_ROOT/evaluation/preview-vggsfm-dark-v2/preview.png`
displays all 187,308 points and visibly contains substantial scattered,
radiating background; it is not a cleaned-person result. The separate
user-requested dark cleanup is separately published below; it does not replace
this raw result or establish a corrected raw camera/track export.

The v1 failure is retained: upstream's 1024-sized ALIKED extraction received
a 640-sized invalid-padding mask. The narrow v2 compatibility adapter only
nearest-resizes that mask to the extractor image shape, preserving polarity
and upstream keypoint rescaling; the pinned source checkout was not edited.
The request binds the adapter and upstream function hashes. See
[the dark VGGSfM guide](../vggsfm/DARK_SHIRT.md).
Raw `-1` point errors are uncomputed sentinels, not pixel-error measurements;
the light-specific coordinate-consistent export is not a dark error receipt.
The raw dark export's 2-D feature observations retain the official padded
inference-coordinate convention; no corresponding corrected dark-observation
export was produced. Do not claim fully corrected dark COLMAP track/pixel
interoperability. This does not invalidate the separate cleanup projection:
it uses the model's original-resolution intrinsics and 3-D poses directly,
not those saved 2-D feature observations.

## Dark VGGSfM cleanup: separately published derivative

The separate CPU pass in job 265977 completed in **26.582505 seconds**:
`DATA_ROOT/vggsfm/outputs/dark-shirt-vggsfm-all290-a10040-fit-v2-mask-crutch-clean-v1/cleanup_receipt.json`.
It keeps **62,889 / 187,308** original points and removes 124,419. The body
consensus selects 57,061; reviewed-corridor protection selects 6,140, of which
312 already pass body consensus. The union therefore saves **5,828** additional
crutch candidates. Retained points comprise 1,995 tracked and 60,894 trackless
grid extras. The raw reconstruction and retained XYZ/RGB are unchanged.

All 290 native masks are matched to the independent model's original-image
names/dimensions. Projections use **VGGSfM's own** poses, intrinsics and radial
distortion, not baseline poses or MVS world-space capsules. Body membership
requires six usable views and 90% foreground agreement. Crutch protection
requires a triple of reviewed image-corridor hits with all three pairwise
camera-to-point ray separations at least 15 degrees. There is no occlusion
test; annotations cannot recover geometry absent from the raw cloud.

The resulting PLY is 943,607 bytes, SHA-256
`0ec0e6983502cd567b1616a1cc315e468b889feabcc77c961c645920448b67c8`.
`DATA_ROOT/evaluation/preview-vggsfm-dark-clean-v1/preview_receipt.json`
records all 62,889 points displayed without clipping or coordinate transforms.
The reviewed preview shows an isolated person and bilateral bright crutch
candidate bands, with substantially less background scatter; nonuniform
density, gaps and body-contact ambiguity remain. The preview alone does not
establish surface completeness or metric accuracy.

The subsequent `publication.json` in that output directory is **complete**
with `visual_review.status=accepted_with_limitations`. Both the full 3-D
preview and six original-image overlays were reviewed: confirmed candidate
bands broadly follow both crutch shaft/fork corridors, with hand, armpit,
leg and floor-contact ambiguity explicitly retained. The independent audit at
`DATA_ROOT/vggsfm/evaluation/dark-shirt-vggsfm-all290-a10040-fit-v2-mask-crutch-clean-v1-validation-v1/audit_receipt.json`
rehashes 290 masks and source/image-identity evidence, reprojects every point,
regenerates the selection/PLY byte-identically, and verifies unchanged raw
and code hashes. It is a separate audit execution using the same projection
helpers, not an independently implemented mathematical oracle. Browser
publication has its own real-render receipt.

## Post-hoc VGGSfM-to-E10 alignment

Status: **VERIFIED COMPLETE; ROBUST ALIGNMENT GATE PASSED**

Populate only from
`DATA_ROOT/evaluation/light-shirt-vggsfm-v1-vs-e10/report.json`.

| Field | Actual value |
|---|---|
| E10 registered cameras | 125 |
| VGGSfM registered cameras | 125 |
| Exact matched cameras | 125, mapped through recorded normalized filename lineage |
| Coverage vs E10 / VGGSfM / request | 100% / 100% / 100% |
| RANSAC inliers / fraction | 95 / 125 = 76%; 30 outliers |
| Similarity scale | 2.00070021643 E10 units per VGGSfM unit |
| Rotation determinant | 1.0000000000000007; no reflection |
| All-camera RMSE / median / p90 / max | 0.211323 / 0.084615 / 0.330546 / 0.871786 E10 arbitrary units |
| Normalized RMSE / median / p90 / max | 0.053810 / 0.021546 / 0.084168 / 0.221987, divided by E10 matched-camera RMS radius 3.927195 |
| Inlier RMSE / median / p90 / max | 0.085768 / 0.049453 / 0.147600 / 0.195896 E10 units |
| Aligned PLY status and gate reason | Written: 414,208 RGB points; 95 inliers exceed both six-camera and 60% gates at 5%-radius threshold |

The aligned PLY has 6,213,434 bytes and SHA-256
`5d672b820a11f03dca6288dcc9e15b5fb9c5335aad127bb4be3d29aa9c93aec5`.
Passing this gate supports a stable post-hoc frame alignment, not the quality
of every point or every estimated camera. All 125 errors remain in the report;
the 30 non-consensus cameras are not silently excluded from overall statistics.

The errors are camera-center disagreement after proper similarity alignment.
Raw values use E10's arbitrary units; normalized values use the E10 matched-
camera RMS radius. They are not centimetres and E10 is not physical ground
truth.

## Cross-method result table

Complete this table only from the validated sections above.

| Method | Estimates cameras? | Registered/reference views | Colored points | Primary wall time | Peak VRAM | Coordinate frame | Status |
|---|---|---:|---:|---:|---:|---|---|
| E10 historical sparse | historical fixed-E1-pose refinement | 125 | 21,744 | 4,299.237 s reconstruction span | not recorded here | E1 arbitrary frame | VERIFIED HISTORICAL |
| Fresh Linux SfM E11 | yes | 125 | 10,204 | 84.476 s engine receipt scope | not sampled | independent arbitrary frame | VERIFIED COMPLETE |
| Fixed-pose portability E12 | no; fixed/scaled E1 cameras | 125 | 41,675 | 16.090 s post-input-validation scope | not sampled | E1 arbitrary frame | VERIFIED COMPLETE; portability validation, not another method |
| COLMAP MVS | no; fixed E10 cameras | 125 | 487,451 | 3,834.510 s successful-controller scope | 990 MiB observed PatchMatch tail only; full-run peak unavailable | E10 frame | VERIFIED COMPLETE |
| VGGSfM v2 | yes | 125 | 414,208 (11,362 tracked + 402,846 extras) | 1,519.284 s official inference | 32,756 MiB sampled process-tree peak | independent arbitrary frame; separate E10-aligned export | VERIFIED COMPLETE |

Point totals are not a direct quality ranking: MVS predicts pixel-wise surface
samples, while the SfM methods triangulate selected tracks. Runtime is also not
an equal-work comparison because the methods solve different problems and may
exclude/include environment setup differently. State the measured time scope.

## Tests and validation record

The table records executed commands and generated receipts. A unit-test pass
validates controlled behavior; it does not substitute for a completed real GPU
run.

| Suite | Command/evidence | Actual result |
|---|---|---|
| Canonical integration harness | `DATA_ROOT/runtime/validation/integration-v2/receipt.json`; Slurm job 264198 with empty `CUDA_VISIBLE_DEVICES` | PASS: 114 / 114 unit/fixture tests across six suites; every return code zero; per-suite log hashes recorded |
| Server SfM source suites | From `CODE_ROOT/sfm`: `headless-cuda/bin/python -B -m unittest discover -s tests -v`; log `DATA_ROOT/sfm/logs/server-sfm-tests-final.log` | PASS: 39 / 39 in 1.824 s: 19 headless-engine, 11 fixed-pose and 9 historical-browser tests |
| Preserved starter/Mac backend | From `CODE_ROOT/sfm/assignment1`: `python -B -m unittest discover -s tests -v`; log `DATA_ROOT/sfm/logs/original-assignment1-tests.log`; environment supplement `DATA_ROOT/sfm/environment-receipt-after-legacy-tests.json` | PASS: 23 / 23 in 9.917 s; Open3D mocked, CPU-only; preserved source hashes recorded |
| MVS | Integration-v2 `mvs` suite: 16 / 16; later `DATA_ROOT/mvs/runtime/completed-cache-unit-tests.log` includes eight completed-cache tests | PASS: 24 / 24 after cache correction; no GPU needed for tests |
| VGGSfM wrapper | Integration-v2 `vggsfm` suite with explicit top-level directory `-t .` | PASS: 20 / 20 |
| Evaluation | Integration-v2 `evaluation` suite with explicit top-level directory `-t .` | PASS: 6 / 6 |
| Storage relocation | Integration-v2 `storage` suite | PASS: 10 / 10 |
| Historical E1-E10 PLYs/viewer | catalog `validate` plus final Chromium receipt `DATA_ROOT/sfm/logs/browser-checks/20260919T165700Z/receipt.json` | PASS: 19 / 19 available clouds matched documented PLY counts and rendered; E10 dark correctly not run; no WebGL/page errors |
| Real E11 cache/reload | `DATA_ROOT/sfm/e11-cache-check-v1.json` | PASS: 16 artifacts unchanged; cache hit; zero reconstruction-stage calls; 125 images / 10,204 points revalidated |
| Real E12 cache/reload | `DATA_ROOT/sfm/fixed-pose-E12-cache-check-v2.json` | PASS: 14 files unchanged; zero triangulation calls; fixed-pose result revalidated |
| Real MVS cache/reload | `DATA_ROOT/mvs/runtime/cache-validation/light1600-readonly-v1.json` | PASS: four stage receipts and 487,451-point PLY; 1,038 file metadata records and 14 JSON hashes unchanged; zero executor/runtime-probe/write calls |
| Real VGGSfM cache/reload | `DATA_ROOT/vggsfm/cache-check-light-shirt-vggsfm-v1.json` | PASS: 139 files unchanged; zero inference/runtime-probe/write calls; 125 views / 414,208 RGB points revalidated |
| Updated integration harness | `DATA_ROOT/runtime/validation/integration-v4/receipt.json` | PASS: 165 / 165 tests: 40 portable SfM + 23 original + 29 MVS + 40 VGGSfM + 10 storage + 6 evaluation + 17 orchestration/recovery |
| Light VGGSfM UI integration checkpoint | `DATA_ROOT/runtime/validation/integration-v5/receipt.json` | PASS: 180 / 180 tests: 41 portable SfM + 24 original + 29 MVS + 45 VGGSfM + 10 storage + 6 evaluation + 25 orchestration/recovery; predates the requested person-cleanup addition |
| Cleanup and dark-MVS integration checkpoint | `DATA_ROOT/runtime/validation/integration-v6/receipt.json` | PASS: 186 / 186 tests: 41 portable SfM + 24 original + 29 MVS + 49 VGGSfM + 10 storage + 6 evaluation + 27 orchestration/recovery; CPU-only, all seven suite return codes zero |
| Both dark cleanups and scoped continuation integration | `DATA_ROOT/runtime/validation/integration-v7/receipt.json` | PASS: 235 / 235 tests: 41 portable SfM + 24 original + 32 MVS + 52 VGGSfM + 10 storage + 6 evaluation + 70 orchestration/allocation/recovery; CPU-only in job 265977; predates the subsequently requested background-colour control |
| Background-control integration checkpoint | `DATA_ROOT/runtime/validation/integration-v8/receipt.json` | PASS: 236 / 236 tests: 41 portable SfM + 24 original + 32 MVS + 52 VGGSfM + 10 storage + 6 evaluation + 71 orchestration/allocation/recovery; CPU-only in job 265977; subsequent receipt-seal hardening is tested separately |
| Final frozen-source integration | `DATA_ROOT/runtime/validation/integration-v10/receipt.json` | PASS: 238 / 238 tests: 41 portable SfM + 24 original + 32 MVS + 52 VGGSfM + 10 storage + 6 evaluation + 73 orchestration/allocation/recovery; all seven return codes zero, CPU-only job 265977; includes final terminal-recovery and exact saved-view identity regressions |

That earlier SfM test total was **62 / 62 across two invocations**: 39 server
tests plus 23 preserved starter/backend tests. This aggregate does not include
the real-output/cache/browser checks in later rows and must not be described as
one monolithic test process.

The earlier integration-v2 cross-project total was **114 / 114**:
`39 + 23 + 16 + 20 + 10 + 6`. These remain unit and fixture tests; the
integration receipt explicitly points to separate real-run and browser
receipts. The earlier `integration-v1` harness used an incorrect unittest
discovery/top-level invocation for the SfM suite. `integration-v2` corrected the
harness flag and passed; this was a harness error, not an engine failure.

Eight MVS completed-cache tests were added after integration-v2; the later
24-test MVS invocation passed separately. Do not rewrite integration-v2's
historical 114-test count as though it had executed those new tests.

The newer integration-v4 receipt records **165/165 passing tests** across seven
suites after the added viewer, cache, export, environment and recovery checks.
The intervening integration-v3 run preserved one stale recovery-test fixture
error (missing MVS method tag); its corrected fixture is covered by v4.

Mac compatibility is preserved in source and covered by Linux-run backend
regression tests with the Open3D dependency mocked. The actual Mac launcher,
Mac Open3D window and macOS-specific preparation tools were **not executed on
a Mac during this server validation**. The separately tested native desktop
window runs Linux Open3D through the allocation's display service; it does not
constitute a Mac runtime test.

## Qualitative inspection record

Do not write “looks good” alone. For interactive MeshLab inspections, record
the same orientation and point-size convention where possible. The E11/E12
entries below instead use deterministic three-projection previews and say
explicitly what those previews cannot establish:

| Output | Foreground completeness | Floaters/background | Thin structures | Camera trajectory | Screenshot/evidence path |
|---|---|---|---|---|---|
| E10 | Browser rendering verified upright; completeness not scientifically assessed | Not assessed by the browser execution check | Not assessed | Cameras not rendered | `DATA_ROOT/sfm/logs/browser-checks/20260919T165700Z/E10-light.png` |
| Fresh SfM E11 | Not established by the preview; all 10,204 points displayed | Full-scene preview visibly retains substantial room/background and scattered points | Not established | Cameras not rendered; trajectory plausibility remains unassessed | `DATA_ROOT/evaluation/preview-e11-light-v1/preview.png` and receipt |
| Fixed-pose E12 | Not established by the preview; all 41,675 points displayed | Full-scene preview visibly retains background and scattered points; no person cleanup was applied | Not established | Cameras not rendered; receipt proves invariance, not visual plausibility | `DATA_ROOT/evaluation/preview-e12-light-v1/preview.png` and receipt |
| MVS light fused | Recognisable head, shirt/lettering, arms and trousers; shirt region substantially denser than the legs; completeness not established | Some scattered peripheral points remain; input masking suppresses much room background | Leg/foot coverage is visibly sparse and contains gaps; no surface/mesh or anatomical-accuracy claim | inherited E10, not re-estimated | `DATA_ROOT/evaluation/preview-mvs-light-v1/preview.png` (243,726 deterministic sampled points of 487,451, original coordinates) and upright native `DATA_ROOT/sfm/logs/desktop/validation/MVS-light-whole-window.png` |
| VGGSfM raw full set | No clean isolated-person completeness established by this full-bounds view | Substantial radiating/scattered geometry and background remain; 402,846 grid additions are not jointly bundle-adjusted | Not established by this preview | Numerical post-hoc alignment: 95/125 inliers, 30 outliers; cameras not rendered here | `DATA_ROOT/evaluation/preview-vggsfm-light-v1/preview.png` and receipt |
| VGGSfM tracked-only diagnostic | Concentrated, partly recognizable torso/arm structure; no completeness claim | Substantial background and outliers remain; trackless additions omitted by the explicit diagnostic predicate only | Not established | Same source cameras, no new estimation | `DATA_ROOT/evaluation/preview-vggsfm-light-tracked-only-v1/preview.png`; all 11,362 selected points, no clipping |
| VGGSfM light mask-cleaned derivative | Clearly isolated person with recognizable head/hair, shirt lettering, trousers and feet; some gaps remain | Large room/radiating outliers removed by the explicit 90%/six-view projection rule | No certified completeness or anatomy claim | Original independent VGGSfM cameras unchanged | `DATA_ROOT/evaluation/preview-vggsfm-light-clean-v1/preview.png`; all 138,677 retained points, no clipping |

Mention human motion, plain clothing, blur, occlusion, mask-boundary errors and
arbitrary scale when they explain observed limitations. Do not invent a
ground-truth accuracy percentage.

## Failure and recovery record

Keep every material failure because it explains the final recipe and proves
that recovery did not silently reuse corrupt state.

| Time / method | Attempt | Failure evidence | Root cause | Change made | Reused state | Final status |
|---|---|---|---|---|---|---|
| 2026-09-19 20:02-20:05 Asia/Dubai / MVS | initial container launch | `DATA_ROOT/logs/slurm-264198.out`: generated SIF then `bad superblock for squashfs image partition` | The 3.0-GB hashed SIF could not be mounted on `gpu-04`; PatchMatch never started in this attempt | Launched the explicit official PyCOLMAP 4.2 CUDA API fallback in the same allocation | Verified method-local E10 model/images/masks and MVS Python environment; no failed dense output was accepted | **VERIFIED COMPLETE** at 21:18:18 Asia/Dubai; 125 views / 487,451 colored points, followed by a read-only cache check |
| 2026-09-19 20:05 Asia/Dubai / VGGSfM | initial Conda bootstrap | `DATA_ROOT/logs/slurm-264198.out` and `DATA_ROOT/vggsfm/logs/install-slurm-264198.log` | `install_linux.sh` set `CONDARC=/dev/null`; cluster Conda 4.11's `conda shell.bash hook` returned rc 1 with `KeyError: 8192` under `set -e`, before clone/install | Replaced only the project-local setting with `vggsfm/configs/condarc`, keeping environments and caches below `DATA_ROOT`; no system Conda or unrelated environment modification | Method-local input copy only; the failed launch created no accepted model/result | **VERIFIED COMPLETE** installation and full-light inference; see installation/environment audit and final output manifest |
| 2026-09-19 / VGGSfM pilot | `light-shirt-vggsfm-pilot16-v1/attempts/0001` | Original failed receipt and successful official log preserved | Wrapper rejected legitimate zero-based camera ID 0 after successful inference | Corrected ID validation, retained reciprocal-track checks, added regression tests; separate CPU review exported saved pilot | Original official binaries, not a new inference | **REVIEWED SUCCESSFUL INFERENCE**: 16/16, 33,696 points; original failed publication receipt remains unchanged |

If no failure occurred for a method, write `none observed` after its successful
receipt exists. Do not delete this section.

## Limitations and defensible conclusion

All three primary server methods processed the full 125-view light capture.
Fresh-camera SfM E11 registered 125 views and published 10,204 RGB points;
E10-initialized PatchMatch MVS used all 125 calibrated references and fused
487,451 RGB points; independent official VGGSfM registered 125 views and
published 414,208 RGB points, including 402,846 trackless grid additions.
Their complete primary PLYs passed finite-XYZ/RGB validation, and saved-model
or completed-cache reload checks proved that inspection does not reconstruct.
Historical E1–E10 results and their provenance remain preserved separately.

SfM E11 and VGGSfM estimate cameras independently; this MVS pipeline does not.
MVS contributes pixel-wise dense depth/fusion, while the SfM outputs arise
from triangulated correspondences. The output counts therefore measure
different sampling processes and are not an accuracy ranking. Full-grid
VGGSfM visibly retains substantial outliers; light MVS is more recognizable
around the shirt but has sparse, incomplete legs/feet. The CPU smoke test's
2/16 registration is an explicitly weak result, not evidence of full quality.
The separately requested VGGSfM person-mask derivative removes the major
background scatter and yields a recognizable 138,677-point person. Its
unchanged retained coordinates and remaining gaps distinguish visual cleanup
from geometric refinement or a claim of complete anatomy.

Post-hoc similarity alignment matched all 125 camera names, with 95 robust
inliers and normalized all-camera RMSE 0.053810 relative to E10's camera-radius
scale. This enables a coordinate-consistent comparison but supplies neither
metric scale nor ground truth. Original-image-pixel VGGSfM residuals are
reported only from the separately corrected observation export; an export
coordinate correction is not a new geometric optimization.

These results establish reproducible execution, internal validation and
viewable artifacts, not metric anatomical accuracy. Possible limitations
include human motion, plain clothing, blur, occlusion and imperfect masks;
the previews do not establish which cause dominates each error. Camera
trajectory/calibration physical plausibility is not independently established.
Dark-shirt extensions, including crutch coverage and their distinct lower-
resolution profiles, must be reported from their own receipts rather than
inferred from light-shirt success. The final orchestration status is also
separate from individual recovered method success.

The 19 historical E1–E10 outputs remain viewable/reloadable, but their retained
Mac-layout experiment scripts are forensic recipes, not promised exact
one-click Linux reruns. The documented fresh-camera and fixed-pose Linux
engines are the portable server paths. Mac GUI compatibility is preserved in
source and regression checks; this server session did not execute the GUI on
a Mac. No additional expensive rerun is warranted merely to replace an
explicitly unavailable historical measurement.

The handoff requested pilots before full runs as a risk-control sequence.
The actual recovery chronology did not consistently follow that order: E13's
CPU smoke and the later measured validation runs followed the original E11,
and the completed light MVS run was not preceded by a separately measured
subset pilot. Preserve this procedural deviation rather than rewriting the
chronology. Original E11 peaks and full-lifecycle attributable light-MVS peaks
are unavailable; E14 and the scoped MVS tail are explicitly different evidence.
The later dark MVS resource measurement describes its own 1024-pixel profile,
not a replacement measurement for the light 1600-pixel run.

## Final evidence bundle

The authoritative recovery bundle is
`DATA_ROOT/runtime/jobs/264198/manual-recovery-final-v1.json`, sealed from
`manual-recovery-terminal-preclosure-v1.json` in the same directory. Accept it
only with status `manual_recovery_complete_initial_orchestration_interrupted`.
It binds the validated raw/clean method artifacts, final 26-cloud/background
browser evidence, the 238-test integration-v10 receipt, permanent-storage
audit and scoped one-hour continuation allocation. The original orchestration
remains FAILED with exit `75:0` and derived exit `2:0`; neither its missing
engine-status receipt nor a successful original wrapper is invented.

Deliverable completion is separate from the viewer's remaining allocation
life. Job 265977 can remain active for inspection until 23:49:47 Dubai without
rerunning reconstruction. No additional allocation, Git push, or deletion of
the retained large Singularity cache is authorized by this completion record.

## Finalization checklist

- [x] Every `PENDING` value is either replaced from a named receipt or left
      explicitly pending/failed.
- [x] Intended configuration is not presented as completed execution.
- [x] The MVS result is described as E10-camera-dependent.
- [x] VGGSfM independence is proved by its request/manifest, not assumed.
- [x] Alignment is described as post-hoc and E10 is not called ground truth.
- [x] Runtime scopes state whether setup/downloads are included.
- [x] Point counts are never used alone as a quality ranking.
- [x] Every published path remains below the permanent data root.
- [x] No image, model, log, environment, cache or generated report was added to
      the code root.
- [x] The README links to this measured report and the viva guide.
