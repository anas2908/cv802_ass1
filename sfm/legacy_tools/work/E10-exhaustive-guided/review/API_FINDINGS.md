# PyCOLMAP 4.2 E10 matching review

This is a source/API investigation, not an additional reconstruction run.
Downloaded upstream files alongside this note are from the official `4.2.0`
tag and retain their original license headers. Source photos, feature
checkpoints and reconstructions were not modified by this review.

## Exhaustive coverage through explicit lists

`match_exhaustive` and `match_image_pairs` use the same feature-matcher
controller. An explicit list containing every unordered image pair therefore
provides exhaustive coverage: 7,750 pairs for 125 light images and 41,905 pairs
for 290 dark images. `ImportedPairingOptions.block_size` is the number of pairs
per internal batch **and** the image-feature/index cache capacity. Its default
is 1,225. Set a bounded explicit value for E10; a smaller cache does not remove
features or pairs.

Sources: [feature_matching.cc](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/controllers/feature_matching.cc),
[pairing.h](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/controllers/pairing.h),
[matcher_cache.cc](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/controllers/matcher_cache.cc).

## Resume semantics

`FeatureMatcherController::Match` skips a pair when **both** a `matches` row
and a `two_view_geometries` row exist. Row existence is independent of the
number of correspondences or geometry configuration: zero-row/invalid results
are completed attempts and must not loop forever on resume.

If only the raw row exists, COLMAP reads its matches, deletes the row,
re-verifies those matches, applies guided matching and writes both results.
If only geometry exists, it deletes that geometry and matches the pair again.
Consequently, an interrupted working database may contain incomplete pairs.
Stable checkpoint completion should require exactly equal raw/geometry pair-ID
sets, containing only authorized exhaustive pairs, and the expected image IDs.

An existing ordinary-only geometry is not proof that guided matching ran. E10
starts from a verified feature-only checkpoint, rather than treating old
ordinary-only pair records as completed guided results.

Sources: [feature_matching_utils.cc](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/controllers/feature_matching_utils.cc),
[database_sqlite.cc](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/scene/database_sqlite.cc).

## Guided eligibility and preserved settings

With guided matching enabled, initial matching is unguided, followed by
geometric verification, followed by the guided worker. `SiftCPUFeatureMatcher`
requires a supported estimated essential matrix, fundamental matrix or
homography to perform its constrained search; it returns without that search
when initial geometry is unavailable. Thus **guided enabled for every pair**
does not promise that every pair can produce valid geometry or guided matches.

Installed defaults: ratio 0.8, distance 0.7, cross-check enabled,
`max_num_matches=32768`, geometric minimum 15 inliers, RANSAC maximum error
4 pixels, confidence 0.999, minimum 100/maximum 10,000 trials,
`random_seed=-1`. The matcher requires the inner RANSAC thread count to equal
one. Outer matching workers may be changed for resource scheduling. Seed -1
means nondeterministic sampling, so fresh runs or different scheduling need
not produce bit-identical estimates.

Sources: [sift.cc](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/feature/sift.cc),
[ransac.h](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/optim/ransac.h),
plus installed `FeatureMatchingOptions().todict()` and
`TwoViewGeometryOptions().todict()`.

## Memory and durability

The CPU guided path allocates two full float distance matrices and two full
integer index matrices. Their combined storage is approximately `16*N*M`
bytes, before caches, temporary allocations and other pipeline stages. At
18,000 features in both images this is 5.184 GB per worker, or 10.368 GB for
two simultaneous workers. The previously measured 9.3 GB single-worker process
peak means two workers cannot be assumed safe on a 24 GB machine. A monitored
small batch with two genuinely geometry-valid maximum-feature pairs is a more
informative stress check than easy/invalid pairs. Keep one worker if memory
pressure/compression/swap rises. No descriptor count, matching threshold or
pair-coverage reduction is required to limit workers/cache size.

The CPU dense matrices use actual feature counts. Lowering `max_num_matches`
is not a solution for these allocations; that option sizes buffers in other
matcher paths.

COLMAP's SQLite opener sets WAL mode and `synchronous=OFF`. Its two match-table
writes are individual statements, not an atomic whole batch. A robust runner
therefore keeps a disposable working database separate from a stable
checkpoint. After a successful bounded batch and closed matcher connection:

1. Verify SQLite integrity and exact completion coverage.
2. Use SQLite backup into a temporary checkpoint (not a bare live `.db` copy).
3. Flush/fsync the temporary database, atomically replace the stable checkpoint,
   and fsync its directory.
4. Write progress metadata afterward; the committed database is authoritative
   if a crash leaves progress stale.

Restore the stable checkpoint after a crash, discard working WAL/SHM files,
and retry only the uncommitted batch. A signature must bind the checkpoint to
the feature database, image/pair identities, PyCOLMAP version and full matching
and verification options. If worker count or batch size is included in that
signature, changing it creates a new checkpoint identity unless a deliberate
validated migration is implemented.
