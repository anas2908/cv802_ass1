# E6–E9: completed results and how to open them

All four experiments are finished for both people. They use the same 125
light-shirt images and 290 dark-shirt images as the higher-detail E3 result.
Every experiment has its own folder. E1–E5 and the original photos are preserved.

## Open a saved map

1. Double-click [run_sfm.command](run_sfm.command) in the SfM folder.
2. Choose **File → Open existing result** in the Mac's top menu bar.
3. Select the light or dark **dataset folder** linked in the table below.

Select the listed folder itself, not its `images` or `colmap` subfolder. Opening
an existing result loads the saved map. **Fit Colmap** starts a new computation.
Drag to rotate and scroll to zoom; reduce Cam Size to 0.1 if camera markers clutter
the person. These results use the existing starter GUI.

The SfM folder is:

`/Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm`

## Saved person views

The numbers below count points in the view you open. E6 and E8 already use the
same rectangle-based person preview as E3. E7 and E9 add segmentation at 90%.

| Experiment | Method | Light shirt | Dark shirt and crutches |
| --- | --- | ---: | ---: |
| E6 | Quality, guided OFF | [22,431 points](reconstructions/experiments/E6_guided_off/light_shirt_quality/subject_preview/) | [40,110 points](reconstructions/experiments/E6_guided_off/black_shirt_crutches_quality/subject_preview/) |
| E7 | E6 + segmentation at 90% | [17,614 points](reconstructions/experiments/E7_guided_off_consensus90/light_shirt/) | [28,429 points](reconstructions/experiments/E7_guided_off_consensus90/black_shirt_crutches/) |
| E8 | Vocabulary tree, guided ON | [24,123 points](reconstructions/experiments/E8_vocab_guided/light_shirt_quality/subject_preview/) | [43,205 points](reconstructions/experiments/E8_vocab_guided/black_shirt_crutches_quality/subject_preview/) |
| E9 | E8 + segmentation at 90% | [18,781 points](reconstructions/experiments/E9_vocab_guided_consensus90/light_shirt/) | [31,097 points](reconstructions/experiments/E9_vocab_guided_consensus90/black_shirt_crutches/) |

For example, the dark E9 dataset folder is:

`/Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/reconstructions/experiments/E9_vocab_guided_consensus90/black_shirt_crutches`

E6 and E8 also save full sparse models in the parent `*_quality` folders.
Their full counts are 33,696 / 78,888 for E6 and 40,581 / 95,896 for E8
(light / dark). Those full models include background points.

## Compare with the original quality results

- Original E3 person previews: 25,063 light / 46,111 dark points.
- Original E4 consensus-90 cleanup: 19,519 light / 32,997 dark points.
- [Light: front comparison](work/matching-ablation/comparison/light_shirt/light_shirt_front.png) · [side comparison](work/matching-ablation/comparison/light_shirt/light_shirt_side.png).
- [Dark: front comparison](work/matching-ablation/comparison/black_shirt_crutches/black_shirt_crutches_front.png) · [side comparison](work/matching-ablation/comparison/black_shirt_crutches/black_shirt_crutches_side.png).
- [Detailed comparison measurements](work/matching-ablation/comparison/README.md).

The figures use the same orientation, scale, point size and quality checks for
all versions. Their comparable-point counts can be slightly smaller than the
saved totals above because they apply shared error/angle/bounds checks.
More points do not by themselves prove a more accurate reconstruction.

In these runs, the original E3/E4 retains the most comparable sparse detail for
both people. Guided OFF reduces comparable points by about 10% for light and
13–14% for dark, with more visible gaps. Vocabulary retrieval is closer to the
original, but still has about 4% fewer comparable points for light and 6% fewer
for dark. Its spatial coverage is also slightly lower under the common checks.
This top-20 vocabulary setting did not improve coverage on these two captures;
it does not establish that vocabulary matching is worse on other datasets.

## What changed

E6 keeps E3's ordinary image pairs and disables its guided pass. E8 changes how
candidate image pairs are found: vocabulary retrieval chooses the top 20 similar
images for each image. Its guided pass stays enabled on a selected subset with
the same budgets as E3: 238 light and 552 dark pairs. E3 also used guidance on a
subset, not every pair. E8 has more ordinary pairs than E3, so this is not an
equal-workload timing comparison.

Both use byte-identical E3 extracted features and the same image resolutions,
person crops, camera poses, calibration and triangulation settings. Application
code and the paste-ready `_estimate_cameras` function are unchanged for E6–E9.

E7 and E9 reuse E4's masks and exactly its 90% rule. They filter their respective
new source models; they do not rerun feature extraction or modify the photos.
Crutches remain protected using the same reviewed regions, with membership
recomputed and checked against photos for each new model. E7 protects 2,302
existing crutch-region points and E9 protects 2,695.

The 90% value means a point agrees with the foreground mask in at least 90% of
its usable projected views. It is not an accuracy score. The earlier 97% trial
remains separate; it removed more unwanted dots but also more valid-looking leg
and footwear detail. That trade-off remains recorded for the final report.

Model/database audits, mask and subset checks, and cache-only loading checks
passed. All 614 previously fingerprinted result/application files were unchanged.
See the [full method and validation record](work/matching-ablation/README.md) and
[experiment index E1–E9](EXPERIMENTS.md).
