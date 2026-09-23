# CV802 Project 1 — 3D Reconstruction

## Quick start

Copy and paste these commands first:

```bash
git clone https://github.com/anas2908/cv802_ass1.git
cd cv802_ass1
export CV802_DATA_ROOT="$(pwd)/cv802-data"
mkdir -p "$CV802_DATA_ROOT"
```

This repository has two jobs:

1. Reconstruct a new image folder with **SfM**, **COLMAP MVS**, or **VGGSfM**.
2. Open the saved E1–E10, MVS, and VGGSfM point clouds in the browser viewer.

The saved display models are included in this repository. New reconstruction
inputs and runtime files stay outside the Git checkout. Set one absolute data
directory before running anything:

```bash
export CV802_DATA_ROOT=/absolute/path/to/cv802-data
mkdir -p "$CV802_DATA_ROOT"
```

Environments, downloaded model weights, databases, caches, and newly generated
results go under `CV802_DATA_ROOT`. The light-shirt and dark-shirt photos used
for the included experiments are already in this repository; no separate photo
download is needed.

## 1. Reconstruct a new image folder

Choose one of the included photo sets from the cloned repository. Run this
from the repository root (use `dark_shirt` instead if desired):

```bash
export CV802_REPO_ROOT="$(pwd)"
export DATASET=light_shirt
export IMAGE_ROOT="$CV802_REPO_ROOT/datasets/$DATASET/images"
```

Use overlapping photographs of one stationary scene. A GPU is recommended.
On the cluster, run setup and reconstruction inside a GPU allocation. On a
personal CUDA machine, set `CV802_ALLOW_NON_SLURM=1` before setup.

### macOS: E1–E10 reconstruction UI

The Linux/CUDA setup script is for cluster or Linux machines and should not be
run on macOS. The Mac UI discovers every `datasets/*/images` folder, lets you
choose the dataset and experiment, and writes databases, caches, masks and new
results under `CV802_DATA_ROOT`. E10 is intentionally available only for
`light_shirt`; the original reviewed crutch-corridor policy is regenerated for
the dark-shirt cleanup experiments.

```bash
cd cv802_ass1
export CV802_DATA_ROOT="$(pwd)/cv802-data"
mkdir -p "$CV802_DATA_ROOT"

python3 -m venv "$CV802_DATA_ROOT/sfm/mac-env"
source "$CV802_DATA_ROOT/sfm/mac-env/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r sfm/requirements-mac-e1-e10.txt

export CV802_ALLOW_NON_SLURM=1
cd sfm
python mac_reconstruction_ui.py ui
```

The UI automatically runs missing prerequisites. For example, choosing E7
first creates E1, E2, E3 and E6 before applying the E7 cleanup. Completed
prerequisites are reused. You can also run without the window:

```bash
python mac_reconstruction_ui.py list
python mac_reconstruction_ui.py plan --dataset light_shirt --experiment E10
python mac_reconstruction_ui.py run --dataset light_shirt --experiment E10
```

To add another dataset, create `datasets/NAME/images/` and place at least two
overlapping images below it; it will appear in the dataset dropdown. Generic
new datasets can run E1 and E2 immediately. E3–E10 use the two assignment-
specific subject profiles because their guided-pair budgets and cleanup rules
were defined for these captures. CPU reconstruction, especially E8/E10, may
take many hours. MVS also requires a compatible macOS COLMAP installation;
VGGSfM is not recommended on a CPU-only Mac.

### A. SfM: cameras and sparse colored points

```bash
cd sfm
bash scripts/setup_linux_cuda.sh

"$CV802_DATA_ROOT/sfm/envs/headless-cuda/bin/python" run_headless.py run \
  --experiment "$DATASET" \
  --images "$IMAGE_ROOT" \
  --camera-model SIMPLE_RADIAL \
  --matcher exhaustive \
  --device cuda
```

The result is written under:

```text
$CV802_DATA_ROOT/sfm/experiments/$DATASET/
```

### B. MVS: dense colored points

MVS uses the original photos plus the calibrated sparse model produced by SfM.

```bash
cd mvs
bash scripts/bootstrap_python_environment.sh
MVS_PYTHON="$CV802_DATA_ROOT/mvs/envs/mvs-engine/bin/python"

"$MVS_PYTHON" run_mvs.py stage-inputs \
  --experiment "$DATASET" \
  --images-source "$IMAGE_ROOT" \
  --model-source "$CV802_DATA_ROOT/sfm/experiments/$DATASET/outputs/colmap/sparse/0"

"$MVS_PYTHON" run_mvs.py plan --config configs/example.json
"$MVS_PYTHON" run_mvs.py run --config configs/example.json --resume
```

The dense point cloud is:

```text
$CV802_DATA_ROOT/mvs/experiments/$DATASET/outputs/fused.ply
```

### C. VGGSfM: learned reconstruction

VGGSfM is independent of the COLMAP SfM result. It starts directly from the
photos and downloads the official pretrained model during setup.

```bash
cd vggsfm
python3 scripts/prepare_input.py \
  --dataset "$DATASET" \
  --source-images "$IMAGE_ROOT"

bash scripts/install_linux.sh
VGG_PYTHON="$CV802_DATA_ROOT/vggsfm/envs/vggsfm/bin/python"

"$VGG_PYTHON" run_vggsfm.py doctor --require-ready
"$VGG_PYTHON" run_vggsfm.py run \
  --dataset "$DATASET" \
  --run-id "${DATASET}_vggsfm" \
  --profile configs/example.json
```

The point cloud is:

```text
$CV802_DATA_ROOT/vggsfm/outputs/${DATASET}_vggsfm/point_cloud.ply
```

## 2. Open the saved models

If you only want to view the saved models, start by cloning the repository:

```bash
export CV802_DATA_ROOT="$(cd .. && pwd)/cv802-data"
mkdir -p "$CV802_DATA_ROOT"
```

The verified model archive is already included in the clone. Install it into
the data directory:

```bash
python3 scripts/package_saved_results.py install \
  --archive saved_models/CV802_Project1_SavedResults_v1.zip
```

Start the CPU-only browser:
if for some reason you are already using 8767 port, change it with 8768 or any other port available, also preferably do it in Mac!
```bash
cd sfm
CUDA_VISIBLE_DEVICES='' python3 -B browse_historical.py \
  --data-root "$CV802_DATA_ROOT" serve --port 8767
```

Open <http://127.0.0.1:8767/>. or the port which you decided. If the code is running on a remote server, first
create this tunnel from your laptop:


again cross check the port
```bash
ssh -N -L 8767:127.0.0.1:8767 USER@SERVER
```

The dropdown contains the available SfM E1–E10, MVS, and VGGSfM results. The
viewer does not run reconstruction and does not need a GPU.

## Repository layout

```text
sfm/       classic COLMAP/PyCOLMAP sparse reconstruction and result viewer
mvs/       COLMAP PatchMatch dense reconstruction
vggsfm/    official VGGSfM inference wrapper
scripts/   photo and saved-result installers
saved_models/  bundled E1–E10, MVS, and VGGSfM display clouds
```

For method-specific command options, run:

```bash
python3 sfm/run_headless.py --help
python3 mvs/run_mvs.py --help
python3 vggsfm/run_vggsfm.py --help
```
