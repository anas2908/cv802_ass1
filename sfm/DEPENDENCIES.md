# SfM dependencies and original backend tests

There are two requirements sets. Do not install `pycolmap` and
`pycolmap-cuda12` together: both provide the same Python module.

`requirements.txt` records the complete preserved Mac GUI and historical helper
requirements. `assignment1/requirements.txt` includes it. These versions were
read from the transferred Mac environment's package metadata, without executing
its Mac binaries:

| Dependency | Historical version | Use |
| --- | --- | --- |
| NumPy | 1.26.4 | Features, camera/point calculations, tests |
| Open3D | 0.19.0 | Original desktop GUI and scene conversion |
| PyCOLMAP | 4.2.0 | SfM API, database and model helpers |
| Pillow | 12.3.0 | Crops, masks, photo preparation, image fixtures |
| SciPy | 1.17.1 | Sparse matching graphs, frame scoring, mask helpers |
| Matplotlib | 3.11.2 | Historical comparisons and previews |
| imageio-ffmpeg | 0.6.0 | Historical video-frame extraction |

The historical set requires Python 3.11–3.12 because the SciPy/Matplotlib pins
require Python >=3.11. Some preserved HEIC preparation scripts also call macOS
tools such as `sips`; listing their Python dependencies does not make those
old media-preparation commands portable. They retain historical paths and are
not run automatically on the server.

`requirements-headless-linux-cuda.txt` contains the smaller server set:
NumPy 1.26.4, `pycolmap-cuda12` 4.2.0 and Pillow 11.3.0. Its actual environment
is `/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda`. The fresh and
fixed-pose engines do not require Open3D, SciPy, Matplotlib or FFmpeg. Pillow
is needed by the original backend's quality-path tests. Version 11.3.0 was
explicitly tested on Linux; it does not claim to reproduce the Mac's 12.3.0.
The original Mac environment remains unchanged.

The optional `tests/browser_smoke.py` uses Playwright and its browser executable
for a visual UI check; it is separate from SfM engines and the unit suites.
See `HISTORICAL_BROWSER.md` for that workflow.

## Original assignment suite

`assignment1/tests` is separate from the new `sfm/tests`. It covers the starter
API's orchestration/cache behavior, quality feature reuse and fixed-camera mode,
E10 SQLite checkpoint/resume behavior, and COLMAP text-model conversion. It
mocks Open3D and expensive reconstruction while exercising real SQLite
transactions/files and installed PyCOLMAP option types. It does not launch the
GUI or prove portability of every old helper.

On 2026-09-19, a CPU-only step in existing allocation 264198 installed only
Pillow 11.3.0 in the SfM environment, using `--only-binary=:all: --no-deps`.
All package/Python/model cache variables and temporary paths were explicitly
redirected under `DATA_ROOT/sfm`. No other environment was changed. The original
suite passed **23 tests in 9.917 seconds**. `pip check` reported no broken
requirements. No compatibility fix to `api.py` or `estimate_cameras.py` was
necessary.

Reproduce on an approved compute node:

```sh
cd /home/anas.khan/cv802_project/project1/ass1/sfm/assignment1
export TMPDIR=/l/users/anas.khan/cv_802_ass1/sfm/tmp
export PYTHONPYCACHEPREFIX=/l/users/anas.khan/cv_802_ass1/sfm/cache/python
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
/l/users/anas.khan/cv_802_ass1/sfm/envs/headless-cuda/bin/python -B \
  -m unittest discover -s tests -v
```

The shown temporary directory must already exist and be writable; setup creates
it. `-B` prevents new code-root bytecode and `TMPDIR` keeps image/database test
fixtures in data storage. All CPU-only Slurm steps in this verification used
`--ntasks=1`, `--gres=none`, and empty `CUDA_VISIBLE_DEVICES`.

The portable suite is run separately from `sfm/`; it discovers the fresh
headless, fixed-pose, historical-browser and desktop-selector checks that exist
at execution time. Do not preserve an aggregate count in this dependency file:
use the measured `test_count` fields in the final `run_validation_suite.py`
receipt linked from [`docs/EXPERIMENT_RESULTS.md`](../docs/EXPERIMENT_RESULTS.md).
That receipt also keeps the 23-test preserved backend suite separate from the
portable suite. Actual E11/E12 executions and the instrumented E11/E12 reloads
remain separate real-output evidence rather than unit tests.

Evidence under `/l/users/anas.khan/cv_802_ass1/sfm`:

- `logs/pillow-original-tests-install.log`
- `logs/original-assignment1-tests.log`
- `environment-freeze-after-legacy-tests.txt`
- `environment-receipt-after-legacy-tests.json`

The earlier `environment-receipt.json` and `environment-freeze.txt` remain as
the pre-Pillow record. The supplemental receipt records the added package and
resulting freeze. Future fresh setup uses the corrected requirements and emits
a richer v2 environment receipt.
