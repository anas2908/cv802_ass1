# CV802 Project 1: SfM, MVS, and VGGSfM

This workspace contains three reproducible 3D-reconstruction engines for the
same capture. The first validation target is the 125-view light-shirt set.
For use outside the MBZUAI server, all three engines accept one explicit
`CV802_DATA_ROOT`; source stays in this checkout, while images, environments,
caches, checkpoints, and outputs live below that data directory.

| Method | Source | Runtime data |
|---|---|---|
| Sparse SfM | `sfm/` | `/l/users/anas.khan/cv_802_ass1/sfm` |
| Dense MVS | `mvs/` | `/l/users/anas.khan/cv_802_ass1/mvs` |
| Learned SfM (official VGGSfM) | `vggsfm/` | `/l/users/anas.khan/cv_802_ass1/vggsfm` |

The permanent server policy requires a small, GitHub-ready home/code tree.
Every new dataset, environment, download, cache, log and generated result goes only to
the Lustre data root. The transferred heavy home duplicates were removed only
after the independent checksum, mixed-source extraction, input-reference and
active-writer gates passed. Their full recovery copy remains in the data root;
the final code-root audit reports zero violations. See [the results
report](docs/EXPERIMENT_RESULTS.md) for receipts and `AGENTS.md` for policy.

The transferred SfM history includes E1-E10.  The authoritative E10
light-shirt result registers all 125 images and contains 21,744 cleaned colored
points.  It is preserved as a reference rather than recomputed. The fresh
Linux/CUDA E11 run is also verified complete: it independently registered all
125 images with five cameras and 10,204 colored sparse points (1.32737 px mean
reprojection error) in 84.476 seconds of engine-receipt time. Dense MVS starts
from E10's calibrated cameras; VGGSfM instead estimates its own cameras from an
independent copy of the images and uses E10 only for post-hoc aligned
evaluation. Light-shirt MVS is verified complete with 487,451 colored dense
points. Official light-shirt VGGSfM completed independently with 125/125
registered views and 414,208 colored points in 1,519.284 seconds of inference.
Of these, 11,362 have feature tracks and 402,846 are additional grid points;
the large point count does not imply a cleaner or more accurate reconstruction.
The separate dark VGGSfM fit-profile run is also complete: 290/290 registered
views and 187,308 colored points (6,061 tracked + 181,247 extras), with a
CPU-only read-only reload. A separately audited crutch-aware cleanup retains
**62,889 points** (including 5,828 protected candidates that body masking alone
rejects), without changing raw coordinates/colors or rerunning inference.
Both dark cleanup previews and camera overlays are accepted with documented
limitations, not as complete or metrically accurate surfaces.
A separate CPU-only person-mask cleanup now retains 138,677 points using
VGGSfM's own cameras and a documented 90%-agreement/six-view rule. It preserves
the original raw result and does not rerun inference or alter retained XYZ/RGB.
Post-hoc camera alignment to E10 passed its robustness gate with 95/125 inliers.
The separately requested dark-shirt MVS run uses all 290 views with a
time-bounded 1024-pixel raw/unmasked profile, preserving the opportunity to
reconstruct crutches; see [its recipe and limitations](mvs/DARK_SHIRT.md).
That dark MVS run completed with 616,827 finite colored points and passed its
read-only reload. Its raw output retains background; crutch overlays are a
qualitative review, not a cleaned segmentation or thin-structure accuracy score.
A separately published dark MVS cleanup retains **455,181 points**, including
both reviewed crutch candidate structures. Its body-mask/crutch union saves
68,112 crutch candidates that the body-mask rule alone would reject. The raw
616,827-point cloud remains unchanged; shafts still have thickness/gaps, so
this is not a certified surface or geometric-accuracy result.

Method-specific commands and implementation explanations are documented in
each method directory. Start with the cross-method
[viva guide](docs/VIVA_GUIDE.md) for every non-UI engine stage and use the
[experiment-results report](docs/EXPERIMENT_RESULTS.md) to distinguish verified
measurements from pending runs. The permanent layout is defined in
[the storage policy](docs/STORAGE_LAYOUT.md).

## Photo bundle, saved-result UI, and inference on another machine

The original 125 light-shirt and 290 dark-shirt-with-crutches photos are
distributed separately as `CV802_Project1_Images_v1.zip` (a private GitHub
Release asset). It includes checksums for every photo. A second release asset,
`CV802_Project1_SavedResults_v1.zip`, contains the 26 already-computed E1-E10,
MVS, and VGGSfM display clouds plus their validation sidecars. These saved maps
can be viewed without the photos.

    export CODE_ROOT=/path/to/cloned/CV802-project1
    export CV802_DATA_ROOT=/absolute/path/to/cv802-data
    python3 "$CODE_ROOT/scripts/prepare_project_images.py" install \
      --archive /path/to/CV802_Project1_Images_v1.zip

The installer creates verified physical copies under the SfM and VGGSfM input
trees and refuses to overwrite existing image folders. MVS receives a
per-experiment copy when `stage-inputs` is run with the selected photos and
calibrated COLMAP model. Segmentation masks, the historical E1 baseline, and
quality-refinement sidecars are not included in the photo asset; they are
needed to reproduce the specialized fixed-pose/cleaned experiments. Fresh SfM
or VGGSfM inference can start from the photos.

Install the old display maps, without photos, into their catalog paths with:

    python3 "$CODE_ROOT/scripts/package_saved_results.py" install \
      --archive /path/to/CV802_Project1_SavedResults_v1.zip

To view existing results without running reconstruction, start the included
CPU-only browser:

    cd "$CODE_ROOT/sfm"
    CUDA_VISIBLE_DEVICES='' python3 -B browse_historical.py \
      --data-root "$CV802_DATA_ROOT" serve --port 8767

Open `http://127.0.0.1:8767/`. On a remote server, forward the port from the
laptop with `ssh -N -L 8767:127.0.0.1:8767 user@server`, then open the same URL
locally. The dropdown displays saved E1-E10, MVS, and VGGSfM clouds when their
result files are present under the data root; it does not compute missing maps.
The current UI is view-only. A future combined dashboard can keep this saved-
result tab and add a separate drag-and-drop upload/run tab; new inference needs
a GPU-backed job runner, and MVS must follow camera estimation.

Example fresh SfM inference:

    python3 "$CODE_ROOT/sfm/run_headless.py" run \
      --experiment reproduction_light \
      --images "$CV802_DATA_ROOT/sfm/inputs/light_shirt/images" \
      --device cuda --camera-model SIMPLE_RADIAL --camera-mode per_folder \
      --matcher exhaustive --max-image-size 1600 --max-num-features 8192

Use each method guide to set up a compatible CUDA environment and run MVS or
VGGSfM. MVS requires a calibrated sparse model; VGGSfM downloads its pinned
pretrained checkpoint on first inference. GPU, CUDA, and dependency versions
can cause small differences even with the same photos and settings.

## Documentation map

| Topic | Document |
|---|---|
| Complete non-UI viva preparation | [Cross-method viva guide](docs/VIVA_GUIDE.md) |
| Verified values and pending server fields | [Experiment results](docs/EXPERIMENT_RESULTS.md) |
| Fresh-camera classic SfM | [Headless SfM engine](sfm/HEADLESS_ENGINE.md) |
| Fixed-pose quality-feature reuse | [Fixed-pose SfM engine](sfm/FIXED_POSE_ENGINE.md) |
| SfM environments and both test suites | [Dependency audit](sfm/DEPENDENCIES.md) |
| E1–E10 + MVS + VGGSfM saved-result dropdowns | [Saved-result viewers](sfm/HISTORICAL_BROWSER.md) |
| Historical E1-E10 lineage | [SfM experiments](sfm/EXPERIMENTS.md) and [comparison](sfm/E1_E10_COMPARISON.md) |
| Pixel-wise dense MVS | [MVS README](mvs/README.md) and [viva guide](mvs/docs/VIVA_GUIDE.md) |
| Independent learned SfM | [VGGSfM README](vggsfm/README.md) and [viva notes](vggsfm/VIVA_NOTES.md) |
| VGGSfM person-mask cleanup and its viva explanation | [Cleanup implementation](vggsfm/CLEANUP.md) |
| Dark VGGSfM body/crutch-union cleanup and validation | [Dark cleanup implementation](vggsfm/DARK_CLEANUP.md) |
| Exact output paths and external-viewer semantics | [Viewing the three results](docs/VIEW_RESULTS.md) |
| Post-hoc trajectory alignment | [Evaluation README](scripts/evaluation/README.md) and [viva guide](scripts/evaluation/VIVA_GUIDE.md) |
| Verified relocation | [Storage tool](scripts/storage/README.md) |

`docs/EXPERIMENT_RESULTS.md` replaces a server measurement only when its
generated data-root receipt validates. E11 is populated from its completed
receipt. Light and dark raw reconstructions, their requested cleanups and the
light post-hoc alignment are recorded from completed receipts. Unmeasured
historical resource or physical-accuracy fields remain explicitly unavailable;
intended settings are never presented as completed results.

Current SfM Python is
`/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python`;
post-hoc evaluation uses
`/l/users/anas.khan/cv_802_ass1/mvs/envs/mvs-engine/bin/python`.
Follow the method guides' `-B` commands and data-root cache/temporary settings.
Older Mac paths in the handoff are provenance. On the server, relocated
historical datasets and reports are under
`/l/users/anas.khan/cv_802_ass1/sfm/historical/transferred/sfm/`, while extracted
historical helper source is in `sfm/legacy_tools/`.

Live terminal monitoring is available with `tmux attach -t cv802-ass1` on the
server. Window `0` shows allocated-GPU utilisation/VRAM and method progress;
in-stage percentages are estimates, not measured fractions of total work.
The original job 264198 ended at its advance-warning signal with exit 75;
its completed per-method recoveries are separate from that wrapper failure.
The user authorized exactly one additional hour: job **265977**, short
`cscc-gpu-p/gpu-debug-qos`, is verified on `gpu-04` from 22:49:47 to 23:49:47
Asia/Dubai on 2026-09-19. Its storage/GPU verification is recorded at
`DATA_ROOT/provenance/gpu-allocation-265977-scoped-v1.json`, preserving the
initial receipt separately. Continuation work restored the loopback viewer
and completed CPU-only dark cleanup, without rerunning raw inference or
relocation. Viewing uses CPU software rendering; a GPU is
not needed to load saved results. A viewer tied to this job stops at expiry.

The lightweight browser at `http://127.0.0.1:8765` now includes **VGGSfM →
Light shirt → Load preserved cloud**. Refresh after keeping the existing SSH
tunnel open. This loads the complete 414,208-point raw result and retains the
original layout with expanded zoom. **VGGCLEAN** offers the cleaned light and
dark results; **MVSCLEAN → Black shirt + crutches** offers the cleaned dark MVS
result. Raw results remain separately selectable. See
[exact viewing steps](docs/VIEW_RESULTS.md).
The top-right **Background** menu offers Dark, Black, White and Grey; choose
Custom and click the colour square for another colour. The saved browser
preference changes only the canvas background, never reconstructed point RGB.

Final validation: **238/238 unit/fixture tests** in
`DATA_ROOT/runtime/validation/integration-v10/receipt.json`, plus a separate
real-browser sweep of all 26 clouds and the background controls. The final
recovery evidence is
`DATA_ROOT/runtime/jobs/264198/manual-recovery-final-v1.json`; its explicit
recovery status preserves the original wrapper failure rather than rewriting
it as a successful run. See the results report for measurement limitations.
For the isolated person, select **VGGCLEAN → Light shirt → Load preserved
cloud** instead; the raw **VGGSFM** option remains available.
