# E10 light-shirt results

E10 is complete and validated: **21,744 saved sparse points from the same 125 images**. Only the final 90%-cleaned light-shirt result is published. The dark E10 run was removed before it started.

## Open the saved map

1. Open [run_sfm.command](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/run_sfm.command>).
2. In the starter GUI, choose **File → Open existing result**.
3. Select [the completed light-shirt folder](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/reconstructions/experiments/E10_quality_exhaustive_guided_consensus90/light_shirt>). The saved point cloud loads from its cache.

## Compare with E4

E4 and E10 use identical 90% mask-consensus cleanup. The comparison also fixes the E1 camera frame, bounds and pixel units; comparable points need at least three distinct views, mean error ≤ 3 E1 pixels and maximum acute triangulation angle ≥ 1.5°.

| Model | Saved cleaned points | Comparable points | Mean error (E1 px) | Occupied voxels (height/100) |
|---|---:|---:|---:|---:|
| E4 | 19,519 | 19,517 | 1.2009 | 4,560 |
| E10 | 21,744 | 21,744 | 1.2224 | 5,041 |

E10 has +2,227 comparable points (+11.41%) relative to E4. This describes sparse coverage, not anatomical accuracy; more points can include noise. These automated checks do not establish a visual improvement or a best reconstruction.

[Front comparison](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/comparison/light_shirt_front.png>) · [Side comparison](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/comparison/light_shirt_side.png>) · [Detailed measurements](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/comparison/comparison.json>)

## Method and measured time

The same E3 native images and saved affine/DSP SIFT features were reused. All **7,750 unique image pairs** were submitted to COLMAP with guided matching enabled on CPU. The recorded matching-worker sequence was 1 → 2 → 4; the final setting used 4 workers. Checkpoint batch sizes followed 128 pairs. Each handover preserved the previously completed guided pairs. The total runtime is not a benchmark for any one worker setting. Guidance can operate only when initial geometric verification supplies suitable geometry. E1 camera poses stayed fixed, intrinsics were scaled to native resolution, and the same E3 triangulation settings were used. The existing segmentation masks then applied the pooled-view 90% consensus rule; no 97% variant was added.

Full reconstruction wall time: **1 h 11 min 39 s** (4299.237 seconds), measured from the first reconstruction step's start through the latest successful reconstruction step's finish. This includes the earlier work, all worker handovers and intervening pauses or cache reopening, while excluding prior feature extraction, subsequent cleanup and final validation. The preserved successful resumed call alone reports 42 min 33 s; that per-call time is recorded separately and is not presented as the full E10 runtime.

[Preserved compute report](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/measured_reconstruction_report.json>) · [Timing provenance](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/reconstruction_timing.json>)

## Validation

Both final matching tables contain all 7,750 requested pair IDs, including unsuccessful attempts. The matching records agree with the durable checkpoint, feature bytes agree with E3, geometry and camera checks passed, and the cleaned result opens without extraction, matching or triangulation. The 90% value describes foreground-mask agreement, not reconstruction accuracy.

[Matching audit](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/final_matching_audit.json>) · [Final cache check](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/final_cache_verification.json>) · [Finalization record](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/work/E10-exhaustive-guided/finalization.json>)
