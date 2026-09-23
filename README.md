# CV802 Project 1 — 3D Reconstruction

This repository is ready to use on an Apple-silicon Mac. It contains:

- the light-shirt and dark-shirt input photographs;
- the SfM E1–E10 reconstruction pipeline;
- the saved E1–E10, MVS, and VGGSfM point clouds; and
- simple browser interfaces for reconstruction and viewing results.

No separate image bundle or saved-model download is required.

## Option 1: Open the saved results

This is the fastest option. It does not reconstruct anything and does not need
a GPU. Copy and paste the complete block into Terminal:

```bash
git clone https://github.com/anas2908/cv802_ass1.git
cd cv802_ass1
bash scripts/run_saved_models_mac.sh
```

The script verifies and installs the bundled models, selects a free local port,
starts the read-only viewer, and opens it in your browser. Use its dropdown to
view the available SfM E1–E10, MVS, and VGGSfM results for both subjects.

Keep the Terminal window open while viewing. Press `Control-C` to stop the
viewer. A later launch only requires:

```bash
cd cv802_ass1
bash scripts/run_saved_models_mac.sh
```

## Option 2: Recompute SfM E1–E10

This option reconstructs the included photographs locally on the Mac. It
requires an Apple-silicon Mac running macOS 14 or newer and an internet
connection for the first environment installation.

Copy and paste:

```bash
git clone https://github.com/anas2908/cv802_ass1.git
cd cv802_ass1
bash sfm/scripts/run_mac_e1_e10.sh
```

The launcher automatically:

1. installs its own Python 3.11 environment without Homebrew;
2. installs and verifies Open3D, PyCOLMAP, and the other pinned packages;
3. keeps the environment, cache, databases, masks, and generated results in a
   separate `cv802-data` folder beside the clone; and
4. opens the reconstruction UI at <http://127.0.0.1:8770/>.

In the UI, select `light_shirt` or `dark_shirt`, select an experiment, and click
**Recompute**. Required earlier experiments run automatically and completed
prerequisites are reused. Only the selected experiment's final output is
created: for example, E7 may prepare/reuse E6, but it does not create E8 or E9;
E4 and E5 also create only their selected cleanup threshold. E10 is
intentionally available only for `light_shirt`.

The UI shows a stage label and progress bar while it runs. After completion,
click **View reconstructed result** to open the new coloured point cloud in the
interactive 3D viewer. Detailed engine output remains available in a collapsed
log section when it is needed. The newest valid result is restored automatically
when the reconstruction UI is launched again.

Keep Terminal open during reconstruction. SfM runs on the CPU on macOS, so the
larger experiments can take a long time.

If the launcher reports that Apple command-line tools are missing, run this
once, finish the installation, and repeat the same launcher command:

```bash
xcode-select --install
```

## Already cloned the repository?

Update it and launch either interface:

```bash
cd "$(git rev-parse --show-toplevel)"
git pull
```

For saved results:

```bash
bash scripts/run_saved_models_mac.sh
```

For SfM reconstruction:

```bash
bash sfm/scripts/run_mac_e1_e10.sh
```

## Adding another image folder

Create the following folder inside the clone and place at least two overlapping
images in it:

```text
datasets/my_dataset/images/
```

Restart the reconstruction launcher. `my_dataset` will appear in the dataset
dropdown. New generic datasets support E1 and E2. E3–E10 use assignment-specific
subject recipes prepared for the included light-shirt and dark-shirt captures.

## If a local port is occupied

Both launchers normally select or use an available local port. You may also
choose one explicitly:

```bash
CV802_VIEWER_PORT=8768 bash scripts/run_saved_models_mac.sh
CV802_UI_PORT=8771 bash sfm/scripts/run_mac_e1_e10.sh
```

## Repository layout

```text
datasets/       included light-shirt and dark-shirt photographs
saved_models/   bundled display point clouds
sfm/            SfM engine, E1–E10 recipes, and interfaces
mvs/            COLMAP multi-view stereo implementation
vggsfm/         VGGSfM implementation
scripts/        portable saved-result and dataset utilities
```

Detailed method documentation is available in
[`sfm/README.md`](sfm/README.md), [`mvs/README.md`](mvs/README.md), and
[`vggsfm/README.md`](vggsfm/README.md).
