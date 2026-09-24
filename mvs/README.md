# MVS

The CIAI reconstruction interface follows the completed assignment recipes:
light shirt uses the reviewed E10 sparse model, foreground masks, and a
1600-pixel PatchMatch run; dark shirt uses the reviewed E3 sparse model and a
1024-pixel PatchMatch run. VGGSfM is a separate reconstruction and does not
provide cameras to this pipeline.

`run_mvs.py` turns a calibrated COLMAP sparse model into a dense colored point
cloud using image undistortion, PatchMatch Stereo, consistency filtering, and
depth-map fusion.

MVS needs registered images and calibrated cameras. For the two included
datasets, the reviewed E10/E3 sparse calibrations and light-shirt masks are
bundled in `datasets/reviewed`, so the user does not have to run SfM first or
have the original assignment data folder. A newly added dataset uses the
generic SfM prerequisite instead.

From a CIAI login node, use the single launcher documented in the root README:

```bash
bash scripts/run_ciai_gpu_ui.sh
```

The launcher allocates one A100 for up to three hours. In the UI, selecting
MVS performs these resumable stages:

1. checksum-verified image staging into the user's Lustre data root;
2. installation of the bundled E10/E3 cameras (or generic SfM for a new dataset);
3. independent MVS input staging;
4. image undistortion;
5. CUDA PatchMatch Stereo with geometric consistency; and
6. depth-map fusion into `fused.ply`.

There are no learned MVS weights. The generated environment, database, dense
workspace, logs and point cloud remain below
`/l/users/$USER/cv802_ass1-runs/mvs` by default. `configs/ciai_template.json` is the reviewed
A100 recipe; the launcher writes its dataset-specific copy into the data root.
The same UI can run and display a separate generic sparse SfM experiment.
