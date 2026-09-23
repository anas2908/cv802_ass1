# VGGSfM provenance

The integration runs the official Meta/VGG implementation; it is not a local
reimplementation of the neural model.

- Repository: <https://github.com/facebookresearch/vggsfm>
- Reviewed/pinned commit: `e1d9d2eb2b3575525792206fb94b2c749c58dc50`
- Official model identity: `facebook/VGGSfM`, file `vggsfm_v2_0_0.bin`
- LightGlue fork used by the official installer:
  <https://github.com/jytime/LightGlue>, pinned commit
  `2f23ca2ea9638cecad7f7220795210fc6b8353c3`
- Review date: 2026-09-19 UTC

The upstream README documents Python 3.10, PyTorch 2.1, CUDA 12.1, the
`python demo.py SCENE_DIR=...` entry point, input images under
`SCENE_DIR/images`, and binary COLMAP output under `SCENE_DIR/sparse`.
The upstream [`cfgs/demo.yaml`](https://github.com/facebookresearch/vggsfm/blob/e1d9d2eb2b3575525792206fb94b2c749c58dc50/cfgs/demo.yaml)
is the source of the Hydra option names used by our allow-listed profile.

`provenance.json` is the machine-readable counterpart. The install receipt
under `/l/users/anas.khan/cv_802_ass1/vggsfm/install_receipt.json` records the
actual Git SHAs, complete `pip freeze`, PyTorch/CUDA versions, GPU name and
cache locations. The first official inference downloads its checkpoint through
the official `hf_hub_download` call. `HF_HOME` and `HF_HUB_CACHE` point below
the method data root, so no model file enters the Git checkout or home cache.

The intentional server OpenCV provider is
`opencv-python-headless==4.10.0.84`. LightGlue 0.0's metadata names
`opencv-python`, so `pip check` cannot recognize the headless alternative and
emits one known warning. The read-only
`/l/users/anas.khan/cv_802_ass1/vggsfm/environment-audit-v1.json` records the
exact command/output, successful `cv2` import, provider versions and SHA-256 of
the fresh 104-line `pip freeze --all` inventory. It preserves the original
install receipt and does not install, uninstall or modify a package.

## Updating upstream

Do not silently pull `main`. To update, review the new official CLI/output
contract, change both constants in `vggsfm_engine/constants.py`, update
`provenance.json` and this file, rerun all tests, reinstall into a new data-root
environment, and use a new experiment run ID. Existing runs remain tied to the
old SHA by their request manifests.
