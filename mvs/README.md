# MVS

`run_mvs.py` turns a calibrated COLMAP sparse model into a dense colored point
cloud using image undistortion, PatchMatch Stereo, consistency filtering, and
depth-map fusion.

MVS must be run after SfM because it needs the registered images and camera
model. The CIAI browser workflow creates or resumes that SfM prerequisite
automatically; the user does not have to supply a model manually.

From a CIAI login node, use the single launcher documented in the root README:

```bash
bash scripts/run_ciai_gpu_ui.sh
```

The launcher allocates one A100 for up to three hours. In the UI, selecting
MVS performs these resumable stages:

1. checksum-verified image staging into the user's Lustre data root;
2. CUDA sparse SfM camera calibration;
3. independent MVS input staging;
4. image undistortion;
5. CUDA PatchMatch Stereo with geometric consistency; and
6. depth-map fusion into `fused.ply`.

There are no learned MVS weights. The generated environment, database, dense
workspace, logs and point cloud remain below
`/l/users/$USER/cv802_ass1/mvs`. `configs/ciai_template.json` is the reviewed
A100 recipe; the launcher writes its dataset-specific copy into the data root.
