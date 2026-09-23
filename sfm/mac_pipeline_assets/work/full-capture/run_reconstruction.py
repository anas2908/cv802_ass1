import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/"assignment1"))
from modules.colmap.api import ColmapAPI
import pycolmap
import open3d as o3d
name=sys.argv[1]
dataset=root/"reconstructions"/name
api=ColmapAPI(0,"SIMPLE_RADIAL","exhaustive_matcher")
api.data_path=str(dataset)
started=time.monotonic()
print("START",name,flush=True)
api.estimate_cameras(recompute=False)
api._thread.join()
if api.estimate_error:
    raise api.estimate_error
models=[]
for folder in sorted(Path(api.sparse_dir).iterdir()):
    if not folder.is_dir() or not (folder/"cameras.bin").exists(): continue
    r=pycolmap.Reconstruction(folder)
    models.append((r,folder))
best,folder=max(models,key=lambda x:(x[0].num_reg_images(),x[0].num_points3D()))
report={"subject":name,"elapsed_seconds":time.monotonic()-started,"input_images":len(api._list_images_in_folder(api.image_dir)),"registered_images":best.num_reg_images(),"points3D":best.num_points3D(),"mean_reprojection_error_px":best.compute_mean_reprojection_error(),"mean_track_length":best.compute_mean_track_length(),"model_folder":str(folder),"registered_groups":dict(Counter(Path(best.images[i].name).parts[0] for i in best.reg_image_ids())),"components":[{"folder":f.name,"images":r.num_reg_images(),"points3D":r.num_points3D()} for r,f in models],"cameras":[str(c) for c in best.cameras.values()]}
best.export_PLY(dataset/"sparse_full_scene.ply")
(dataset/"reconstruction_report.json").write_text(json.dumps(report,indent=2))
print("COMPLETE",json.dumps(report,indent=2),flush=True)
