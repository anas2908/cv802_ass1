# SfM

`run_headless.py` reconstructs cameras and a sparse colored point cloud from a
folder of overlapping photographs. Runtime data is stored under
`$CV802_DATA_ROOT/sfm`.

The `historical_browser` package and `browse_historical.py` provide the
read-only saved-result viewer described in the root README. The original
Open3D starter application is preserved in `assignment1/`.

The CIAI GPU UI in the root README also exposes a simple demonstration choice:

```bash
bash scripts/run_ciai_gpu_ui.sh
```

Selecting **SfM — raw + generic quality cleanup** creates or resumes the normal
CUDA sparse reconstruction and displays both its raw colored PLY and a separate
derived cleanup. `clean_sparse.py` keeps camera poses and the COLMAP model
untouched; it filters only the display points using track support, reprojection
error and a broad robust coordinate gate. This is generic geometric cleanup,
not semantic person segmentation.
