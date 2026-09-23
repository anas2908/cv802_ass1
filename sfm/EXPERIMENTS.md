# SfM experiment record

The requested single-table comparison is in [E1_E10_COMPARISON.md](E1_E10_COMPARISON.md), including exact pairing counts, guided matching, feature settings, camera handling and cleanup.

## E10: exhaustive high-quality matching with guided refinement and 90% cleanup

Started on 19 September 2026. E10 uses the exact E3 native images and saved
features and all 7,750 light-shirt unique image pairs, with
guided matching enabled for every attempted pair. Initial geometric verification
must still succeed before COLMAP can apply guided refinement. Fixed E1 camera
poses, scaled intrinsics and the established triangulation settings are retained.

The user narrowed E10 to **light shirt only**, with **only the 90%-cleaned result**.
The dark-shirt run was removed from the queue before it started. The final folder is
`reconstructions/experiments/E10_quality_exhaustive_guided_consensus90/light_shirt`;
the raw reconstruction and rectangle filtering are internal intermediates under
`work/E10-exhaustive-guided/`. The earlier dark-shirt experiments remain unchanged.
E10 is complete and validated: **21,744 cleaned light-shirt points**.
See [E10 results and opening guide](</Users/Anas Anwarul Haq Khan/Documents/mbzuai/AAR_ya_PAAR/ass1/sfm/E10_RESULTS.md>) for the E4 comparison and measured runtime. Point counts alone
do not establish visual improvement or anatomical accuracy.

The completed run used opt-in checkpoints and batches of 128 pairs. At the user's
request it changed from one guided worker to two after 1,152 completed pairs,
then tried four after 1,920 completed pairs. Verified saved pairs were transferred
unchanged, preserving original checkpoints and migration provenance. The Mac
has 15 CPU cores and 24 GB RAM. A monitor can return the run to two workers if
four create sustained memory pressure; a distinct checkpoint cadence of 129
preserves each execution recipe on that fallback. Timing spans all worker
settings and their handovers, with no changes to features or matching thresholds.
E10 changes both the ordinary pair list and the number of pairs offered guided
refinement relative to E3/E4. Its result therefore cannot isolate the benefit of
exhaustive pairing from the benefit of wider guided coverage. The saved E3
features are reused, so its measured run time excludes feature extraction and
is not directly comparable to the full E1 end-to-end run time. The light-shirt
databases contain 941,176 features in E1 (7,529 per image on average) versus
1,741,593 in E10 (13,933 per image on average). Exhaustive pairing alone does not
specify how expensive those individual pair comparisons will be.
This extension changes only `_estimate_cameras` in the application; its paste
copy is synchronized. Existing quality recipes remain compatible. The checkpoint
feature passed 17 existing tests and six interruption/recovery tests. Earlier
experiments and source photographs remain separate. E10 matching, cleanup and
final verification are complete; the result is available through the starter GUI.

## Required discussion in the final report

### Segmentation: 90% versus 97%

User explicitly requested that the report explain the 90% versus 97% cleanup
trade-off. Both variants use the same images, source reconstruction and masks.
The percentage is the minimum foreground-mask agreement across valid, in-frame
projections of each existing 3D point; it is not an accuracy score or a fraction
of input images retained. For example, a point with foreground agreement in
95 of 100 usable projections passes 90% but fails 97%.

The 97% setting is stricter and removes more unwanted points, but segmentation
errors and subject movement can also cause genuine body points to fail. Visual
inspection of our results showed additional loss of leg and footwear detail at
97%. We recommend the 90% versions as the better balance for these captures,
without claiming that 90% is universally optimal or that point counts establish
accuracy. Both dark-shirt variants retain the same separately verified crutch
protection. Include this observation and the before/after comparisons when
writing the report.

### Selective image-pair matching and its relationship to guided matching

The user also explicitly requested that the report retain the explanations below.
Include both an accessible explanation and the reproducible selection settings.

**All input images were retained.** Selective matching reduces which pairs of
images are compared; it does not reduce the 125 light-shirt or 290 dark-shirt
input images. E1 first performed exhaustive matching and estimated a baseline
reconstruction. Its camera positions and rectangle-filtered person preview
provided the information used to shortlist pairs for the higher-detail E3 pass.

The script estimated the person's center as the coordinate-wise median of the
preview's 3D points. It measured the angle between camera-position directions
from that center, and counted how many existing person-region 3D point tracks
were observed in both images. These are approximate overlap cues from the
earlier reconstruction, not semantic labels or ground-truth body points.

For each image, the custom rules combined:

| Rule | Selection |
| --- | --- |
| Nearby viewpoints | 12 nearest camera-position directions around the person |
| Known overlap | Up to 8 images with the most shared person-region 3D points, positive shared support and angles from 2° to 65° |
| Links across captures | Up to 4 nearby views from each other phone/lens/video group, within 65° |
| Sequence continuity | The previous 2 and next 2 images in capture order within the same group |

The union removed duplicate and self pairs. Ordinary candidates more than 85°
apart were excluded. Strict guided candidates were also included in the ordinary
list. The script checked that all images belonged to one connected matching
graph; neither actual subject required extra connectivity bridges. These are
hand-set heuristic rules, not a trained image-pair selection model or a built-in
COLMAP exhaustive/vocabulary matcher. Exact code and recorded pair reasons are
in [prepare_quality_datasets.py](work/quality-sfm/prepare_quality_datasets.py)
and each E3 dataset's `pair_metadata.json` and `graph_statistics.json`.

For the accessible explanation: take a front-facing photo A. A nearby front-left
photo B is a promising partner, particularly if the earlier reconstruction found
many 3D points observed in both A and B. A rear photo may share little visible
detail. The script creates a shortlist of such partners, then asks COLMAP to
match their features. Any illustrative shared-point counts, such as 200 versus
5, are hypothetical examples, not measurements from these captures.

**A shortlisted pair is only a candidate.** COLMAP's `match_image_pairs` performs
SIFT descriptor matching and geometric verification on the supplied pairs.
Guided matching subsequently runs on a subset to search for additional feature
correspondences constrained by estimated two-view geometry. For the epipolar
case, a pixel in one image specifies a ray with unknown depth; its possible
locations in the other image lie on an epipolar line. Guidance searches a narrow
tolerance band and still checks appearance. A feature lying near the line is
not automatically a correct correspondence. COLMAP can also guide using a
homography for suitable two-view configurations.

State the distinction clearly: **pair selection chooses which photos to compare;
guided matching helps choose corresponding features inside those photo pairs.**
The guided subset favored shared support, different view angles and connections
across capture groups, using the recorded budgets below.

### Why exhaustive matching has N(N−1)/2 pairs

For N images, N × N includes N self comparisons and counts every distinct pair
twice: A–B and B–A. Removing self comparisons and duplicate directions gives
`(N × N − N) / 2 = N(N−1)/2` unique unordered image pairs.

| Subject | Input images | All possible distinct pairs / E1 exhaustive | E3 selected ordinary pairs | E3 guided subset |
| --- | ---: | ---: | ---: | ---: |
| Light shirt | 125 | 7,750 | 1,439 | 238 |
| Dark shirt with crutches | 290 | 41,905 | 3,974 | 552 |

### Correct interpretation of E3/E4 and the comparison results

Describe the lineage explicitly: **E1 exhaustive baseline → E3 higher-detail
features with custom pair selection and guided refinement → E4 segmentation
cleanup at 90%.** E3 used the existing cameras, scaled calibration and new
triangulated points; it was a refinement of the earlier reconstruction, not a
fresh camera-estimation experiment. E4 inherits E3's matching and only filters
its points. E3/E4 did not repeat exhaustive matching at higher detail. Record the
actual pair-list recipe rather than the GUI's misleading `exhaustive_matcher`
dropdown label for these saved quality runs.

E3/E4 retained the most comparable sparse detail in these runs; E4 was the
strongest cleaned result by the measured coverage and visual comparisons. Do
not describe that as demonstrated ground-truth anatomical accuracy or fastest
runtime. E6/E7 isolate guided OFF with the same ordinary pairs and features.
E8/E9 use genuine top-20 vocabulary retrieval with guided ON for a subset; the
ordinary pair counts and chosen guided edges differ, so their comparison is not
an equal-workload timing benchmark. Use the
[saved comparison measurements](work/matching-ablation/comparison/README.md)
and [matching provenance](work/matching-ablation/README.md) in the report.

## Experiment overview

The main work starts with baseline and higher-detail SfM for two subjects,
followed by filtered copies and two matching comparisons. The cleanup steps do
not run SfM again. Completed results are preserved in their existing folders.
These labels document the work; no result folders have been renamed.

The full input sets are unchanged between baseline and quality runs:

- Light shirt: 91 photos + 34 video frames = 125 images.
- Dark shirt with crutches: 183 photos + 107 video frames = 290 images.

## Main stages

| Label | Stage | What changed | Subjects |
| --- | --- | --- | --- |
| E1 | Baseline SfM | Photos at 2400 × 3200; video frames at 1080 × 1920; SIFT and exhaustive image-pair matching; estimated cameras and sparse points | Both |
| E2 | Initial rectangle cleanup | Checked existing points against person bounding rectangles across views; removed much of the surrounding room | Both |
| E3 | Higher-detail SfM refinement | Photos reconverted from original HEICs at 3024 × 4032; finer affine/DSP SIFT on person crops; selected overlapping pairs and guided matching; new sparse points using fixed baseline cameras | Both |
| E4 | Apple Vision masks, 90% agreement | Person-shaped masks filtered the E3 subject preview; dark result additionally protects photo-checked crutch points | Both |
| E5 | Apple Vision masks, 97% agreement | Same inputs and masks as E4, with stricter body agreement; the same crutch protection remains | Both |
| E6 | Guided matching OFF | Identical E3 native images, extracted features, ordinary pairs, fixed cameras and triangulation; omit the guided pass | Both, complete |
| E7 | E6 with masks at 90% | Same frozen masks and filter settings as E4; crutch protection recalculated and photo-checked on E6 points | Both, complete |
| E8 | Vocabulary-tree matching, guided ON | Genuine top-20 vocabulary-retrieved neighbors; same E3 features/cameras/triangulation and guided-pair budgets | Both, complete |
| E9 | E8 with masks at 90% | Same E4 cleanup and crutch-protection policy, applied to E8 | Both, complete |
| E10 | Exhaustive + guided + 90% | E3 native features, every unique pair, guided ON throughout; publish only the consensus90-cleaned light-shirt map | Light shirt only; complete, 21,744 cleaned points |

E3 combined several changes. There was no full run changing only image resolution,
so the improvement cannot be attributed to pixels alone. Video resolution and
image counts stayed unchanged. E3 also has a rectangle-filtered subject preview
using the E2 approach before mask cleanup.

E4 and E5 are two cleanup settings, not separate feature extraction or matching
runs. The percentages describe mask agreement for each projected 3D point.
The 90% results retain more useful body and footwear detail; 97% removes some
valid-looking detail along with unwanted points.

E6/E7 compare guided OFF against E3/E4. E8/E9 compare vocabulary-based image-pair
selection against E3/E4, keeping guided matching ON for a selected subset.
E3 used 238 guided pairs for light and 552 for dark, not guidance on every pair.
E8 uses the same budgets within its retrieved graph. See the
[matching experiment method](work/matching-ablation/README.md) for settings,
pair counts, validation and the workload differences that limit timing claims.

## Main saved results

| Result | Light shirt | Dark shirt with crutches |
| --- | --- | --- |
| E1 full SfM | [light_shirt](reconstructions/light_shirt/) — 22,230 points | [black_shirt_crutches](reconstructions/black_shirt_crutches/) — 57,440 points |
| E2 baseline subject preview | [subject_preview](reconstructions/light_shirt/subject_preview/) — 7,247 points | [subject_preview](reconstructions/black_shirt_crutches/subject_preview/) — 15,546 points |
| E3 full SfM | [light_shirt_quality](reconstructions/light_shirt_quality/) — 41,675 points | [black_shirt_crutches_quality](reconstructions/black_shirt_crutches_quality/) — 101,150 points |
| E3 subject preview | [subject_preview](reconstructions/light_shirt_quality/subject_preview/) — 25,063 points | [subject_preview](reconstructions/black_shirt_crutches_quality/subject_preview/) — 46,111 points |
| E4 masks at 90% | [mask_consensus_90](reconstructions/experiments/light_person_mask_cleanup/mask_consensus_90/) — 19,519 points | [body_mask_consensus_90_crutches](reconstructions/experiments/black_person_crutches_cleanup/body_mask_consensus_90_crutches/) — 32,997 points |
| E5 masks at 97% | [mask_consensus_97](reconstructions/experiments/light_person_mask_cleanup/mask_consensus_97/) — 14,812 points | [body_mask_consensus_97_crutches](reconstructions/experiments/black_person_crutches_cleanup/body_mask_consensus_97_crutches/) — 26,790 points |
| E6 guided OFF, full SfM | [light](reconstructions/experiments/E6_guided_off/light_shirt_quality/) — 33,696 points | [dark](reconstructions/experiments/E6_guided_off/black_shirt_crutches_quality/) — 78,888 points |
| E6 person preview | [light](reconstructions/experiments/E6_guided_off/light_shirt_quality/subject_preview/) — 22,431 points | [dark](reconstructions/experiments/E6_guided_off/black_shirt_crutches_quality/subject_preview/) — 40,110 points |
| E7 masks at 90% | [light](reconstructions/experiments/E7_guided_off_consensus90/light_shirt/) — 17,614 points | [dark](reconstructions/experiments/E7_guided_off_consensus90/black_shirt_crutches/) — 28,429 points |
| E8 vocabulary, full SfM | [light](reconstructions/experiments/E8_vocab_guided/light_shirt_quality/) — 40,581 points | [dark](reconstructions/experiments/E8_vocab_guided/black_shirt_crutches_quality/) — 95,896 points |
| E8 person preview | [light](reconstructions/experiments/E8_vocab_guided/light_shirt_quality/subject_preview/) — 24,123 points | [dark](reconstructions/experiments/E8_vocab_guided/black_shirt_crutches_quality/subject_preview/) — 43,205 points |
| E9 masks at 90% | [light](reconstructions/experiments/E9_vocab_guided_consensus90/light_shirt/) — 18,781 points | [dark](reconstructions/experiments/E9_vocab_guided_consensus90/black_shirt_crutches/) — 31,097 points |

Point counts describe saved results, not accuracy scores. Full scenes include
background. Subject previews and mask results deliberately contain fewer points.

All E6–E9 datasets are complete and load from cache in the starter application.
See the [E6–E9 opening guide](E6_E9_RESULTS.md) and
[fixed-view numerical comparisons](work/matching-ablation/comparison/README.md).
E7 protects 2,302 existing dark crutch-region points; E9 protects 2,695. These
counts differ from E4's 3,028 because matching produces different source points;
the same protection recipe is evaluated anew and photo-checked for each model.

## Smaller completed trials

- Initial starter-code smoke test: 20 images supplied; 11 registered and 1,128
  sparse points in [sample_capture](sample_capture/).
- Rectangle-filter trials: an observation-only light preview retained 8,750
  points before all-view volume filtering; a dark 80% volume preview retained
  15,712 before tightening to 90%. Both are in
  [preview_archive](work/full-capture/preview_archive/). The preview reports also
  record threshold sensitivity; every threshold is not a separate SfM run.
- Feature/matching pilot: fine SIFT versus affine/DSP SIFT on four sample images
  across the two subjects, matching with guided mode off and on, and checks
  against baseline camera geometry. These informed E3. Reports are in
  [pilot](work/quality-sfm/pilot/).
- Segmentation pilot: Apple Vision person segmentation versus foreground-instance
  segmentation on six dark-shirt images. Neither reliably included exposed
  crutches, so foreground-instance masks were not used for a full cleanup.
  See [pilot findings](reconstructions/experiments/black_person_crutches_cleanup/PILOT_QC.md).
- Crutch-protection development: visible guides in six photos were checked first
  against baseline points and then against the quality model. The final E4/E5
  protection retains 3,028 existing quality points around the two crutches.
  This is part of the dark cleanup, not a third SfM reconstruction method.

Software tests, cache checks, and the correction to count distinct image views
are validation/correctness work rather than additional reconstruction methods.
Vocabulary-tree reconstruction is recorded as E8 above. The historical E1-E10
series itself contains no MVS, VGGT or VGGSfM experiment; the separate server
engines are documented at the assignment root and must not be retroactively
described as part of this Mac experiment series.

To view a saved result, start `run_sfm.command`, choose **File → Open existing
result**, and select its dataset folder. The [cleanup guide](reconstructions/experiments/README.md)
explains the mask settings and limitations.
