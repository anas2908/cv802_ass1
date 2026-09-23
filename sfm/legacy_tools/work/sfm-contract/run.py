from pathlib import Path
import numpy as np
import open3d as o3d
import pycolmap
from api import ColmapAPI
print("Versions", np.__version__, o3d.__version__, pycolmap.__version__, flush=True)
api=ColmapAPI(gpu_index=0,camera_model="OPENCV",matcher="exhaustive_matcher")
api.data_path=str(Path("work/sfm-real-test").resolve())
api.estimate_cameras()
api._thread.join()
assert api.num_cameras >= 2
xyz=np.asarray(api.pcd.points)
rgb=np.asarray(api.pcd.colors)
assert len(xyz)>0 and xyz.shape==rgb.shape
assert np.isfinite(xyz).all() and ((rgb>=0)&(rgb<=1)).all()
for name in api.camera_names:
    K,E=api.extract_camera_parameters(name)
    assert E.shape==(4,4)
    assert np.allclose(E[:3,:3]@E[:3,:3].T,np.eye(3))
    assert K.intrinsic_matrix[0,0]>0
print("REAL INTEGRATION PASS",api.num_cameras,len(xyz),flush=True)
o3d.io.write_point_cloud(str(Path(api.data_path)/"sparse_preview.ply"),api.pcd)
# Cached loading must work without executing any SfM stages.
def should_not_run(*args,**kwargs):
    raise AssertionError("Cache failed: attempted recomputation")
pycolmap.extract_features=should_not_run
api.estimate_cameras()
api._thread.join()
assert api.num_cameras>=2
print("CACHE RELOAD PASS",flush=True)
