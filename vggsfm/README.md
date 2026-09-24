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

After inference, select **VGGSfM Mac-mask cleanup** to reproduce the saved
person-mask projection rule using the current VGGSfM cameras and historical
Mac masks from `CV802_REFERENCE_DATA_ROOT` (default: sibling `cv_802_ass1`).
The dark-shirt recipe also uses the historical reviewed image-space crutch
corridors. No historical 3D geometry or camera poses are reused. Select
**VGGSfM geometric cleanup** only if masks are unavailable; it is a different,
less selective outlier filter. Both are CPU post-processes and keep the raw
output untouched. Each writes a separate PLY and `cleanup_receipt.json`.

Nothing is downloaded to the source checkout or home directory. The official
source, environment, Torch/Hugging Face model caches, attempts, logs and output
remain below `/l/users/$USER/cv802_ass1/vggsfm`. A later allocation reuses them.
The reviewed A100 profiles are `configs/light_shirt.json` and
`configs/dark_shirt.json`; the dark-shirt profile uses a smaller image size and
query budget so all 290 photographs fit an A100 40 GB.
