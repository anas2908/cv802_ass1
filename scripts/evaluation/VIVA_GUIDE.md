# Viva guide: VGGSfM-to-E10 alignment

## Why alignment is necessary

Monocular SfM determines a scene only up to a global similarity transform:
orientation, origin and scale are gauge freedoms. VGGSfM and COLMAP E10 can
therefore reconstruct the same capture with numerically different coordinate
systems. Comparing their raw coordinates would be meaningless. E10 is used as
the assignment reference after VGGSfM completes independently; it is not
physical ground truth and its units are not metres.

The methods also triangulate different 3-D points, so this tool does not invent
point-to-point correspondences. Registered images provide the legitimate
correspondence: the same photograph defines one camera center in each model.

## Exact image-name correspondence

The official VGGSfM demo needs flat filenames. Its wrapper transforms an input
such as `images/photos_iphone_13_26mm/0002_IMG_8289.jpg` into a digest-prefixed
flat name and records both strings in the immutable experiment `request.json`.
The evaluator reverses only this recorded map, removes the leading `images/`,
and exactly matches the resulting nested name to E10's `images.bin` name.

It never guesses by basename: two camera folders may both contain `same.jpg`.
Unsafe paths, duplicate official/original names, and a registered VGGSfM image
missing from the receipt are fatal provenance errors. An image present in only
one reconstruction is reported and reduces coverage.

## From COLMAP pose to camera center

COLMAP stores a Hamilton quaternion `q=(qw,qx,qy,qz)` and translation `t` for
the **world-to-camera** equation

```text
x_camera = R(q) X_world + t.
```

The optical center is the world point mapped to zero, hence

```text
C_world = -R(q)^T t.
```

`colmap_source.py` reuses the repository's tested, read-only binary COLMAP
parser, requires binary `cameras.bin/images.bin/points3D.bin`, normalizes the
nearly-unit quaternion, computes `C`, and rejects non-finite poses. Imports are
anchored to the checked-out `mvs/src` and `vggsfm` source roots and verified by
resolved module path; mutable/downloaded code is not selected from a cache.

## Proper Umeyama similarity

For matched VGGSfM centers `x_i` and E10 centers `y_i`, the model is

```text
y_i ≈ s R x_i + t,       s > 0, det(R) = +1.
```

After subtracting both centroids, the implementation forms

```text
Sigma = (1/N) sum (y_i-y_bar)(x_i-x_bar)^T
      = U diag(d) V^T.
```

It sets `S=diag(1,1,sign(det(UV^T)))`, then computes

```text
R = U S V^T
s = trace(diag(d) S) / mean(||x_i-x_bar||^2)
t = y_bar - s R x_bar.
```

The sign correction guarantees `det(R)=+1`. A mirror reflection is therefore
flagged but never mislabeled as a rotation. Unit tests show that a known
proper transform is recovered to numerical precision and reflected synthetic
data retains nonzero error.

Coincident or collinear center sets are rejected because they do not constrain
a 3-D similarity. Rank two is accepted: a non-collinear, near-planar camera
trajectory can still determine the proper mapping.

## Robust fitting and the PLY gate

One bad predicted camera can distort ordinary least squares. A deterministic
RANSAC loop samples three non-collinear correspondences, applies Umeyama,
scores all cameras, and refits the consensus until membership stabilizes. Its
distance threshold defaults to 5% of the RMS radius of matched E10 camera
centers. This ratio is scale-aware without claiming metric units.

The report always includes errors over **all** matched cameras, plus separate
inlier errors, so outliers cannot disappear from the result. The colored
VGGSfM sparse PLY is transformed only when:

- at least six cameras match;
- at least six and at least 60% are inliers;
- source and target consensus geometry is non-collinear;
- inlier p90 and all-match median pass the configured relative gates; and
- every transform, residual and transformed point is finite.

With `auto`, a failed gate produces an honest `withheld` record rather than a
misleading aligned cloud. With `required`, the JSON remains available for
diagnosis but the command exits nonzero.

## Reading the numbers

- **Coverage vs E10:** matched cameras divided by E10 registered cameras.
- **Coverage vs VGGSfM:** matched cameras divided by VGGSfM registered cameras.
- **Coverage vs request:** matched cameras divided by all submitted images.
- **RMSE:** emphasizes large camera-center disagreements.
- **Median:** typical disagreement and less outlier-sensitive.
- **p90:** 90% of residuals are at or below this value.
- **Maximum:** worst matched camera.
- **Scale:** E10 arbitrary units per one VGGSfM arbitrary unit.
- **Normalized error:** raw error divided by matched E10 camera RMS radius.

Low error supports geometric agreement of camera trajectories after gauge
alignment. It does not prove metric accuracy, point-cloud completeness, or
surface quality. Motion, textureless clothing, rolling shutter, differing
intrinsics assumptions and E10's own estimation error remain limitations.

## File-level implementation map

- `alignment.py`: finite checks, proper Umeyama, deterministic robust fit and
  residual statistics.
- `colmap_source.py`: anchored reuse of tested COLMAP binary helpers and
  `C=-R^Tt`.
- `matching.py`: receipt parsing, safe exact name mapping and coverage.
- `evaluator.py`: provenance hashes, optional atomic colored PLY and atomic
  JSON report.
- `io_utils.py`: permanent-root enforcement, SHA-256 and atomic writes.
- `cli.py` / `run_evaluation.py`: user arguments, fixed defaults and concise
  machine-readable terminal summary.

