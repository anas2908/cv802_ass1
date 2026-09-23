# Dark-shirt VGGSfM body and crutch cleanup

This is a **derived, CPU-only point-selection result**. It is not another
VGGSfM inference run, does not change the reconstructed cameras or points, and
does not replace the 187,308-point raw dark-shirt result. The browser exposes
both:

- `VGGSFM / Black`: 187,308 raw points;
- `VGGCLEAN / Black`: 62,889 selected points.

The published result is below
`DATA_ROOT/vggsfm/outputs/dark-shirt-vggsfm-all290-a10040-fit-v2-mask-crutch-clean-v1`.
Its `publication.json` is the authority used by the UI.

## Why body masks alone are insufficient

The 290 person masks describe the body but omit much of the two crutch shafts.
A body-only mask vote would therefore delete useful assistive-device geometry.
The published policy takes the union of a strict body selection and a separate
crutch-corridor selection:

```text
keep(p) = body_consensus(p) OR separated_crutch_support(p)
```

The two terms are reported separately so crutch protection cannot be confused
with passing the body-mask threshold.

## Body projection rule

Every raw world point is projected with the **dark VGGSfM run's own** 290
original-resolution `SIMPLE_RADIAL` cameras and poses:

1. `X_cam = cam_from_world(X_world)`;
2. require finite coordinates and positive camera-space depth;
3. divide by depth and call PyCOLMAP's `camera.img_from_cam`, which applies the
   camera's radial model;
4. require the continuous pixel to be inside the camera and mask bounds, then
   use NumPy nearest-pixel rounding;
5. count foreground when the uint8 mask value is at least 128.

For usable-view count `u(p)` and foreground count `f(p)`, body consensus is:

```text
u(p) >= 6  and  f(p) >= ceil(0.90 * u(p))
```

Pixels are not clamped and there is no depth-buffer/occlusion test.

## Crutch-corridor rule

Six original dark images contain independently reviewed 2D polylines for both
crutches. Coordinates are normalized by the native image width and height, and
each polyline has a declared half-width fraction. A projected point produces a
corridor hit when it falls inside either left/right polyline corridor.

A point receives crutch protection only when at least three annotated cameras
hit and **every pair** of their camera-centre-to-point rays differs by at least
15 degrees. The angular rule rejects a set of nearly duplicate views that
could agree on an accidental projection. The protection uses no E1/E3 world
capsule, point ID, pose or geometry. It only uses the independent dark VGGSfM
cameras and existing raw VGGSfM XYZ.

This is evidence of multi-view corridor agreement, not proof of visibility or
semantic/metric crutch accuracy. An occluded background point can project into
a corridor, and corridor width can keep bands around a shaft.

## Published counts

| Quantity | Count |
|---|---:|
| Raw points | 187,308 |
| Body-consensus points | 57,061 |
| Crutch-protected points | 6,140 |
| In both sets | 312 |
| Protected only by crutch rule | 5,828 |
| Union retained | 62,889 |
| Removed | 124,419 |

The identities conserve exactly: `62,889 + 124,419 = 187,308` and
`57,061 + 6,140 - 312 = 62,889`.

## Byte-preserving implementation

`vggsfm_engine/projection_cleanup.py` contains the reusable projection,
nearest-mask and separated-ray helpers. The entry point
`scripts/clean_dark_with_masks_and_crutches.py` computes membership, then copies
the selected 15-byte `float32 XYZ + uint8 RGB` raw PLY records byte-for-byte.
It does not optimize poses, retriangulate, move XYZ, recolor, interpolate or run
a neural model. The published PLY is 943,607 bytes with SHA-256:

```text
0ec0e6983502cd567b1616a1cc315e468b889feabcc77c961c645920448b67c8
```

## Provenance and fail-closed validation

`scripts/prepare_dark_cleanup_inputs.py` prepares method-local mask and
corridor provenance. `scripts/audit_dark_cleanup_pixel_identity.py` proves that
all 290 source images used for those masks are byte-identical to the VGGSfM
inputs, not merely name- or dimension-matched.

The independent audit is:

```text
DATA_ROOT/vggsfm/evaluation/
  dark-shirt-vggsfm-all290-a10040-fit-v2-mask-crutch-clean-v1-validation-v1/
  audit_receipt.json
```

It rehashes every mask against its preparation receipt before and after use,
binds the input images/cameras/corridors and cleaner source, independently
recomputes membership, regenerates the output PLY byte-for-byte, and verifies
the raw source manifest remains unchanged. `projection_rows.json` retains the
complete per-image audit table rather than only a digest.

The six annotated overlays are under the output's `overlays/` directory. The
contact sheet and the deterministic preview were visually inspected. Candidate
bands broadly follow both crutch shafts/forks across multiple views and the
raw radial background is greatly reduced. Remaining limitations include gaps,
nonuniform density, body-adjacent bands, ambiguous top/floor contacts and the
lack of an occlusion test. The `visual_review.json` status is therefore
`accepted_with_limitations`, not an accuracy certification.

## Reproduction

Use the data-root environment and caches. These commands perform only input
validation, projection, selection and review; they do not rerun inference:

```bash
export CV802_DATA_ROOT=/l/users/anas.khan/cv_802_ass1
export TMPDIR=$CV802_DATA_ROOT/vggsfm/tmp
export PYTHONPYCACHEPREFIX=$CV802_DATA_ROOT/vggsfm/cache/pycache
PY=$CV802_DATA_ROOT/vggsfm/envs/vggsfm/bin/python

$PY -B vggsfm/scripts/prepare_dark_cleanup_inputs.py
$PY -B vggsfm/scripts/audit_dark_cleanup_pixel_identity.py
$PY -B vggsfm/scripts/clean_dark_with_masks_and_crutches.py
$PY -B vggsfm/scripts/audit_dark_cleanup.py
$PY -B vggsfm/scripts/render_dark_crutch_overlays.py
$PY -B vggsfm/scripts/publish_dark_cleanup_review.py
```

Published output directories are immutable. A different threshold, corridor or
camera policy needs a new run ID and a separate publication.

## Viva answers

**Why union two tests?** The body masks intentionally omit much of the crutches.
The union keeps strong body consensus while retaining independently supported
crutch candidates.

**Why require separated cameras?** Three nearly collinear projections provide
weak independent evidence. Pairwise ray separation makes the corridor vote a
genuine multi-view check, although it still does not solve occlusion.

**Did this borrow COLMAP/E10 geometry?** No. Pixel identity and annotations are
reused as observations, but all projections use the dark VGGSfM run's own
cameras and raw XYZ. No historical world capsule is applied.

**Is the cleaned result better reconstruction geometry?** It is a more focused
display subset, not newly estimated geometry. Filtering cannot repair missing,
noisy or inaccurate surfaces.

**Why keep the raw cloud?** Cleanup is a policy choice with known failure modes.
Keeping raw and clean as separate UI choices preserves the original evidence
and makes every retained/removed decision auditable.
