# CV802 Project 1 — 3D Reconstruction

This repository is ready to use on an Apple-silicon Mac. It contains:

- the light-shirt and dark-shirt input photographs;
- the SfM E1–E10 reconstruction pipeline;
- the saved E1–E10, MVS, and VGGSfM point clouds; and
- simple browser interfaces for reconstruction and viewing results.

No separate image bundle or saved-model download is required.

It also includes a one-command CIAI GPU interface for rebuilding sparse SfM,
dense MVS or VGGSfM results on an A100. That workflow is described in Option 3.

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
log section when it is needed. E10 uses four CPU matching workers, saves a
checkpoint after every 128 pairs, and displays its completed count out of 7,750
pairs. The newest valid result is restored automatically when the reconstruction
UI is launched again.

Keep Terminal open during reconstruction. SfM runs on the CPU on macOS, so the
larger experiments can take a long time.

If the launcher reports that Apple command-line tools are missing, run this
once, finish the installation, and repeat the same launcher command:

```bash
xcode-select --install
```

## Option 3: Recompute SfM, MVS or VGGSfM on CIAI

Run this option on a **CIAI login node**, not on a Mac. The clone is placed on
Lustre because the repository includes the assignment photographs. Copy and
paste:

```bash
cd "/l/users/$USER"
git clone https://github.com/anas2908/cv802_ass1.git cv802_ass1-source
cd cv802_ass1-source
bash scripts/run_ciai_gpu_ui.sh
```

The launcher requests one A100 40 GB GPU, 16 CPU cores and 96 GB RAM from the
CIAI debug queue for at most three hours. It may first wait in the Slurm queue.
Once the allocation starts, the terminal prints exactly two things to use on
your laptop:

1. an `ssh -N -L ...` tunnel command; and
2. a private `http://127.0.0.1:8780/?token=...` browser URL.

Run the printed SSH command in a second terminal on your laptop, then open the
printed URL. Keep both terminals open. In the UI, choose the dataset and either:

- **SfM** — estimates cameras and a raw sparse colored cloud, then creates a
  separate quality-cleaned display cloud without changing the source model;
- **MVS** — reproduces the reviewed saved-run recipe: light shirt uses the E10
  quality model and person masks at 1600 px, while dark shirt uses its E3 model
  at 1024 px; it then runs COLMAP CUDA image undistortion, PatchMatch Stereo
  and depth-map fusion; or
- **VGGSfM** — independently reconstructs from the raw photographs using the
  pinned official model; or
- **VGGSfM cleanup** — after that raw run finishes, removes isolated 3D
  outliers on the CPU without rerunning inference. The raw cloud stays intact,
  and both versions appear as separate viewer links.

The generic SfM cleanup rejects weak two-view points, points above a 3-pixel
reprojection error and extreme coordinate outliers. It is intentionally labeled
as geometric cleanup, not semantic person segmentation. The raw SfM result is
also retained and viewable for comparison.

VGGSfM cleanup is likewise geometric: it filters points with unusually large
nearest-neighbour distances. It does **not** reproduce the historical
person-mask/crutch review, and may remove thin structures or leave coherent
background. Inspect raw and cleaned clouds before choosing one to present.

The UI shows overall progress, detailed logs, and live GPU utilization and
memory. When a run completes, click its result link to rotate, pan, zoom and
change the background of the colored point cloud.

All generated state belongs to the person running the command. Environments,
downloaded VGGSfM weights, caches, databases, temporary files, logs and outputs
are written below:

```text
/l/users/$USER/cv802_ass1/
```

MVS has no learned weights. VGGSfM downloads its pretrained weights on the
first run and reuses them from that Lustre folder afterward. Both methods are
restartable: if the three-hour allocation ends, rerun the same launcher and
select the same dataset and method. Press `Control-C` in the CIAI job terminal
when finished to release the GPU immediately.

For the two included datasets, the MVS UI intentionally uses the calibrated
inputs from the completed assignment rather than silently replacing them with
a new generic SfM run. The launcher automatically recognizes the original
sibling data root `/l/users/$USER/cv_802_ass1`. If those reviewed inputs are
stored elsewhere, set `CV802_REFERENCE_DATA_ROOT` to that absolute assignment
data root before starting the launcher. VGGSfM is independent: it uses only the
raw photographs and its pretrained weights, never E3 or E10 cameras.

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

For the CIAI SfM/MVS/VGGSfM interface, run `git pull` in the Lustre clone and then:

```bash
bash scripts/run_ciai_gpu_ui.sh
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
