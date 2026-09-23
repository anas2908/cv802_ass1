from pathlib import Path
import json,shutil,time,sqlite3
import numpy as np
import pycolmap

root=Path(__file__).resolve().parents[2]
work=Path(__file__).resolve().parent
records=json.loads((work/'pilot/feature_report.json').read_text())
results=[]
for subject in ('light_shirt','black_shirt_crutches'):
    base=root/'reconstructions'/subject
    for variant in ('fine_sift','affine_dsp'):
        subset=[r for r in records if r['subject']==subject and r['variant']==variant]
        database=work/'pilot'/f'{subject}_{variant}.db'
        if database.exists():raise FileExistsError(database)
        shutil.copy2(base/'colmap/database.db',database)
        model=pycolmap.Reconstruction(base/'colmap/sparse/0')
        db=pycolmap.Database.open(database)
        db.clear_keypoints();db.clear_descriptors();db.clear_matches();db.clear_two_view_geometries()
        for cam in model.cameras.values():
            if cam.width==2400:cam.rescale(3024,4032)
            cam.has_prior_focal_length=True;db.update_camera(cam)
        ids=[]
        for record in subset:
            im=db.read_image_with_name(record['name']);ids.append(im.image_id)
            f=np.load(record['npz'])
            db.write_keypoints(im.image_id,f['keypoints'])
            db.write_descriptors(im.image_id,pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT,f['descriptors']))
        db.close()
        pairs=work/'pilot'/f'{subject}_{variant}_pairs.txt'
        pairs.write_text(subset[0]['name']+' '+subset[1]['name']+'\n')
        for guided in (False,True):
            if guided:
                db=pycolmap.Database.open(database);db.clear_matches();db.clear_two_view_geometries();db.close()
            start=time.monotonic()
            pycolmap.match_image_pairs(database_path=database,matching_options={'num_threads':1,'guided_matching':guided},
                pairing_options={'match_list_path':str(pairs)},device=pycolmap.Device.cpu)
            db=pycolmap.Database.open(database)
            geometry=db.read_two_view_geometry(*ids)
            row={'subject':subject,'variant':variant,'guided':guided,'seconds':time.monotonic()-start,'inliers':len(geometry.inlier_matches),'configuration':str(geometry.config)}
            db.close();results.append(row);print(json.dumps(row),flush=True)
            (work/'pilot/matching_report.json').write_text(json.dumps(results,indent=2))
