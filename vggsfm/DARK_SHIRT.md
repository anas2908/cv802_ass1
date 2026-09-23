# Dark-shirt/crutches VGGSfM run

This run reconstructs the second person independently with official VGGSfM.
It uses all **290 original views**: 183 native stills at 3024x4032 and 107
video frames at 1080x1920. It never consumes E1/E3/E10 cameras, poses, sparse
points or an MVS workspace. Dark E10 does not exist.

## Input and crutch policy

The method-local dataset is fixed at:

```text
/l/users/anas.khan/cv_802_ass1/vggsfm/inputs/dark_shirt/images
```

The source copy is the already verified, self-contained raw image set under
the dark MVS experiment. Copying is a provenance source only: VGGSfM receives
new regular files in its own method directory and has no runtime dependency on
MVS. `input_receipt.json` records every destination size/SHA-256 and the exact
source path. `source_provenance_audit.json`, produced by
`scripts/audit_dark_input.py`, independently rehashes both sides against the
frozen source manifest and records the 183/107 dimension split. The launcher
refuses anything except `selection=all_images`,
`image_count=original_image_count=290`, no symlinks and no masks.

This first result is raw/unmasked. Historical person masks can omit thin
crutch shafts; applying them would risk deleting the mobility aids. Any later
foreground cleanup must be a separately named derived output with new crutch
projection review, not a silent change to this run.

## Why the profile differs from light

The completed 125-view light profile (`1024 px`, six query frames, 2,048 query
points, 10-pixel additional grid) reached a sampled **32,756 MiB** process-tree
GPU peak on a 40-GB A100. Keeping it unchanged while increasing the view count
to 290 is not a defensible memory request.

`configs/dark_shirt_all290_a10040_fit_v2.json` keeps every input and retains
SIMPLE_RADIAL per-image cameras, ALIKED, fine tracking, FP16, robust refinement,
one BA iteration, 16-frame extra-point neighborhoods and the fixed seed. Its
explicit resource-fit changes are:

| Knob | Light | Dark all-290 fit |
|---|---:|---:|
| resized square | 1024 | 640 |
| query frames | 6 | 4 |
| maximum query points | 2048 | 1536 |
| extra-point grid interval | 10 | 16 |

All-image tensor pixels become `290*640^2 / (125*1024^2) = 0.90625` of the
measured light run. The main query output budget is about 1.16x light, while
the additional-grid sample budget is about 0.354x light. These ratios are
planning bounds, not a promised memory/runtime model; the saved resource
samples are authoritative. This is an **A100-40GB fit profile**, not the same
quality setting as light and not evidence that a larger point count is better.

The light raw output itself contains 11,362 tracked points plus 402,846
non-BA, trackless additional points and visible radiating outliers. The dark
profile retains a sparser additional grid for method continuity, but reports
tracked/trackless counts separately and makes no quality claim from density.

## Preparation and run

Bulk copying must happen only inside the existing CV802 Slurm allocation. The
generic importer atomically copies and checksum-verifies all files:

```bash
VGG_CODE=/home/anas.khan/cv802_project/project1/ass1/vggsfm
MVS_DARK=/l/users/anas.khan/cv_802_ass1/mvs/experiments/dark_e3_colmap_mvs_1024_raw_v1/inputs/images
/l/users/anas.khan/cv_802_ass1/vggsfm/envs/vggsfm/bin/python -B \
  "$VGG_CODE/scripts/prepare_input.py" \
  --dataset dark_shirt --source-images "$MVS_DARK"
"$VGG_CODE/scripts/audit_dark_input.py"
```

Exact immutable identifiers:

```text
dataset: dark_shirt
run_id: dark-shirt-vggsfm-all290-a10040-fit-v2
profile: vggsfm/configs/dark_shirt_all290_a10040_fit_v2.json
```

The preserved `...fit-v1` attempt ran for 223.221 seconds and failed before
reconstruction because pinned LightGlue resizes ALIKED query images to 1024
but passed VGGSfM's 640x640 invalid-padding mask to that 1024-resolution score
map. It was not an OOM. V2 runs the pinned official demo through
`scripts/official_demo_compat.py`. The adapter reproduces upstream
`Extractor.extract`, nearest-resizes only the boolean invalid mask when its
spatial shape differs from the extractor image, preserves mask polarity, and
leaves upstream keypoint-to-input coordinate mapping unchanged. The immutable
request and result receipt bind the wrapper file hash, pinned LightGlue
`utils.py` file hash and exact `Extractor.extract` function hash. Neither
upstream checkout is edited.

After other GPU work is finished, launch inside that same allocation:

```bash
/home/anas.khan/cv802_project/project1/ass1/vggsfm/scripts/run_dark_shirt.sh
```

The measured V2 attempt shares only this project's allocated physical A100
with dark COLMAP MVS (about 0.5--1 GiB); it must not overlap another VGGSfM
process or any unrelated project. Its timing is therefore not an uncontended
VGGSfM benchmark. A deadline guard terminates the official child shortly
before the allocation ends if necessary, allowing the engine to write a
failed receipt instead of leaving a misleading `running` receipt. If it fails,
preserve the attempt and use an explicit `--resume` in a later authorized
allocation; never silently drop views or relabel a subset as the full result.
