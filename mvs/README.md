# MVS

`run_mvs.py` turns a calibrated COLMAP sparse model into a dense colored point
cloud using image undistortion, PatchMatch Stereo, consistency filtering, and
depth-map fusion.

MVS must be run after SfM because it needs the registered images and camera
model. See the root README for the minimal stage-and-run commands.

