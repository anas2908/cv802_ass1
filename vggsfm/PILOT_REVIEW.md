# Read-only pilot review

The first 16-image light-shirt inference exited successfully in **122.063 s**,
but its original wrapper publication failed because the binary validator
incorrectly rejected valid COLMAP camera ID zero. The corrected validator and
actual PyCOLMAP 3.10.0 independently reloaded the saved model: 16/16 requested
images, 16 cameras, 33,696 RGB points (5,661 tracked + 28,035 trackless extras).
This is a pilot subset of the 125-view dataset, not a full-set result.

`scripts/review_pilot.py` never runs inference. It hashes original evidence,
checks source/staged images against the frozen request, verifies all registered
names, parses the official completion marker and process-specific samples,
loads the binary model twice, then exports into a **new** review directory.
It refuses existing reviews. Original files must retain their bytes, sizes,
mtimes, ctimes and inodes throughout; the failed attempt receipt stays failed.

The actual review lives under
`DATA_ROOT/vggsfm/experiments/light-shirt-vggsfm-pilot16-v1/review-v1`:
`review.json` contains the evidence and `point_cloud.ply` the colored export.
The saved 60 samples cover 121.906 s and show peaks of 26,550 MiB GPU memory
and 2,905,336 KiB summed process-tree RSS. They are sampled inference-process
peaks, not node-wide readings or estimates for 125 images.

Important: the official model stored reprojection-error fields as **-1
(uncomputed)**. The preserved review-v1 receipt exposes that raw sentinel;
it is not a measured negative pixel error. Updated converters use null for
unavailable measured means. Trackless extras were not bundle-adjusted; the
tracked mean track length is 5.7461579226.

The exact completed command was the following CPU-only step in the existing
job. Do not repeat with `review-v1`; a subsequent audit needs a new review ID.

```bash
srun --jobid=264198 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --gres=none --export=ALL \
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR=/l/users/anas.khan/cv_802_ass1/vggsfm/tmp \
  PYTHONPYCACHEPREFIX=/l/users/anas.khan/cv_802_ass1/vggsfm/cache/pycache \
  /l/users/anas.khan/cv_802_ass1/vggsfm/envs/vggsfm/bin/python -B \
  /home/anas.khan/cv802_project/project1/ass1/vggsfm/scripts/review_pilot.py \
  --run-id light-shirt-vggsfm-pilot16-v1 --attempt 0001 \
  --review-id review-v1 --resources resources-677766.jsonl
```

Five focused tests cover exact/final completion markers, missing completion,
requested-name mapping, resource attribution and frozen-file identity.
