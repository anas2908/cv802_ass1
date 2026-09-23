# Light-shirt VGGSfM projection cleanup

The dark-shirt body-and-crutch policy is documented separately in
[`DARK_CLEANUP.md`](DARK_CLEANUP.md); it must not be described as the same
body-only rule because its published set unions separately verified crutch
corridor candidates.

This is a **derived display result**, not another reconstruction and not a
replacement for the official raw VGGSfM output.  The browser intentionally
exposes both:

- `VGGSFM / Light`: all 414,208 official raw points;
- `VGGCLEAN / Light`: 138,677 points retained by the rule below.

The cleaned file is
`DATA_ROOT/vggsfm/outputs/light-shirt-vggsfm-v1-mask-projection-clean-v1/point_cloud.ply`.
The raw file and its manifest remain byte-for-byte unchanged.

## Inputs and coordinate system

`scripts/prepare_light_cleanup_masks.py` copies the 125 existing light-person
masks into the method-local
`DATA_ROOT/vggsfm/inputs/light_shirt_cleanup_masks/masks` tree.  Its receipt
records each relative name, native dimensions, byte size, mode and SHA-256.
Preparation requires an exact one-to-one name match; it does not borrow another
method's camera poses.

`scripts/clean_light_with_masks.py` uses the corrected, original-resolution
COLMAP export of the **independently estimated VGGSfM** cameras:

1. transform world point `X` with `image.cam_from_world`;
2. reject non-finite coordinates or camera-space `Z <= 0`;
3. form normalized coordinates `(Xc/Zc, Yc/Zc)`;
4. call the camera model's `img_from_cam`, including its SIMPLE_RADIAL
   distortion, to obtain original-image pixels;
5. round with NumPy `rint` and count the view as usable only when the pixel is
   finite and inside both the camera and native mask dimensions;
6. count foreground when the uint8 person mask value is at least 128.

For point `p`, let `u(p)` be its usable-view count and `f(p)` its foreground
vote count.  The published profile keeps exactly

```text
u(p) >= 6  and  f(p) >= ceil(0.90 * u(p))
```

No pixel is clamped.  There is no depth-buffer or occlusion test.  In
particular, an in-bounds projection is not proof that the point is visible.
This distinction matters for the 402,846 official grid extras, which have no
COLMAP tracks.  Therefore the result is correctly called **projection-based
mask consensus**, not historical track-visible consensus and not ground-truth
segmentation.

## What is and is not changed

The script parses `points3D.bin` only to preserve point IDs/order and classify
tracked versus trackless points.  Every selected 15-byte
`float32 XYZ + uint8 RGB` record is copied directly from the raw PLY.  It does
not optimize poses, retriangulate, recolor, interpolate, denoise or run a
neural network.  Only membership changes.

Published counts are:

| Quantity | Count |
|---|---:|
| Raw points | 414,208 |
| Retained | 138,677 |
| Removed | 275,531 |
| Retained tracked | 4,026 |
| Retained trackless extras | 134,651 |

The threshold sweep from the same projection pass retained 165,114 at 70%,
157,477 at 80%, and 138,677 at 90% (minimum 3 and minimum 6 happened to differ
by no selected points for these thresholds; two raw points had only 3--5 usable
views).  Ninety percent was retained as a separately labelled review profile
after visual inspection, not presented as a universal optimum or accuracy
score.

## Validation and evidence

The independent audit is
`DATA_ROOT/vggsfm/evaluation/light-shirt-vggsfm-v1-mask-projection-clean-v1-validation-v1/audit_receipt.json`.
It rehashes every raw manifest file, corrected-camera file and mask before and
after use; independently recomputes the votes; regenerates the PLY; and
requires byte identity with the published derived PLY.  Its per-image audit
table is `projection_rows.json`, so camera/name/dimension and usable-projection
coverage can be rechecked rather than trusted as an opaque digest.

Key evidence:

- cleaned PLY: 2,080,414 bytes, SHA-256
  `ca1c35a7dec46aaa7aeb614c3ce79917dbb6b095222a141120cf7ade8412a04f`;
- count conservation: `138,677 + 275,531 = 414,208`;
- all selected XYZ values are finite and RGB is present;
- all 125 mask dimensions equal their VGGSfM camera dimensions;
- raw output, masks and camera model are unchanged;
- inference, pose recalculation, XYZ/RGB recalculation and raw replacement are
  all explicitly false.

The inspected preview is under
`DATA_ROOT/evaluation/preview-vggsfm-light-clean-v1`.  It shows an isolated
person with head, shirt, trousers and feet, while retaining some coverage gaps.
That visual check is useful for failure discovery; it is not an anatomical or
metric-accuracy claim.

## Reproduction commands

Run these only in the dedicated data-root environment and redirect temporary
and bytecode files to the data root:

```bash
export CV802_DATA_ROOT=/l/users/anas.khan/cv_802_ass1
export TMPDIR=$CV802_DATA_ROOT/vggsfm/tmp
export PYTHONPYCACHEPREFIX=$CV802_DATA_ROOT/vggsfm/cache/pycache
PY=$CV802_DATA_ROOT/vggsfm/envs/vggsfm/bin/python

$PY -B vggsfm/scripts/prepare_light_cleanup_masks.py
$PY -B vggsfm/scripts/clean_light_with_masks.py \
  --minimum-usable 6 --agreement 0.9
$PY -B vggsfm/scripts/audit_light_cleanup.py
```

The output directories are immutable: an existing published result is
validated/reused or rejected; it is never silently overwritten.  Use a new
run ID in the source before testing a different policy.

## Viva answers

**Why not use E10 poses?**  That would leak a baseline into the independent
VGGSfM method.  Every projection uses VGGSfM's own calibrated camera and pose.

**Does 90% mean 90% accuracy?**  No.  It is the fraction of usable camera
projections landing in a foreground mask, not a precision/recall measurement.

**Why are trackless points allowed?**  They dominate the official raw output
and can still be tested geometrically against all cameras.  They are reported
separately because those projections are not observed feature tracks.

**Can background points survive?**  Yes.  Pose/mask error, missing occlusion
reasoning, or repeatedly projected foreground can retain an outlier.  The
cleanup reduces raw radiating clutter; it does not certify geometry.

**Why preserve the raw result?**  Derived filtering is a policy choice.  Keeping
the immutable raw reconstruction makes the comparison auditable and prevents a
cleaned display from being misrepresented as VGGSfM's native output.

## Audit scope and remaining helper limitations

The supplemental audit is an independently executed **reproducibility check**:
it reuses the cleaner's projection/selection implementation, rather than
providing an independently implemented mathematical oracle. The projection
unit fixtures and external code review provide separate checks of those rules.
The recorded original and repeated per-image projection digests agree, and
the audit stores the complete 125-row table; the helper currently records that
agreement evidence rather than explicitly asserting equality to the original
row digest. Its published PLY byte-equality check is mandatory.

Mask preparation's existing-folder fast path only checks receipt status/count.
Do not use it alone as a cache-integrity proof: the cleaner and supplemental
audit rehash the actual masks against the receipt before using them. Both
validated the currently published result's 125 files. No raw data was changed
to pass these checks.
