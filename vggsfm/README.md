# VGGSfM

`run_vggsfm.py` runs the pinned official VGGSfM model on an image folder and
exports a colored point cloud plus a COLMAP model. It is independent of the
classic SfM and MVS pipelines.

Use `scripts/prepare_input.py`, `scripts/install_linux.sh`, and the commands in
the root README.

For the simplest CIAI path, run this from the repository on a login node:

```bash
bash scripts/run_ciai_gpu_ui.sh
```

The launcher requests one A100 for up to three hours and opens the shared MVS /
VGGSfM browser interface. VGGSfM performs checksum-verified input staging,
installs the pinned official repository and CUDA environment, downloads the
official pretrained weights on first use, runs independent inference, and
exports `point_cloud.ply`. It does not reuse classic SfM cameras or MVS output.

After inference, select **VGGSfM cleanup** in the CIAI UI to create a separate
CPU-filtered cloud. This does not rerun inference or alter the raw cloud.
It is statistical 3D outlier removal, not the earlier mask-based person/crutch
cleanup. The filter settings and raw/output checksums are saved beside the
cleaned PLY in `cleanup_receipt.json`.

Nothing is downloaded to the source checkout or home directory. The official
source, environment, Torch/Hugging Face model caches, attempts, logs and output
remain below `/l/users/$USER/cv802_ass1/vggsfm`. A later allocation reuses them.
The reviewed A100 profiles are `configs/light_shirt.json` and
`configs/dark_shirt.json`; the dark-shirt profile uses a smaller image size and
query budget so all 290 photographs fit an A100 40 GB.
