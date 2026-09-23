from pathlib import Path
import json, time, sqlite3
import numpy as np
from PIL import Image
import pycolmap

root=Path(__file__).resolve().parent
cases=[('light_shirt','photos_iphone_11_26mm/0000_IMG_6763.jpg'),
       ('light_shirt','photos_iphone_13_26mm/0002_IMG_8289.jpg'),
       ('black_shirt_crutches','photos_iphone_11_26mm/0000_IMG_6653.jpg'),
       ('black_shirt_crutches','photos_iphone_13_26mm/0005_IMG_8212.jpg')]
out=root/'pilot';out.mkdir(exist_ok=True)
report=[]
for variant,extra in [('fine_sift',{}),('affine_dsp',{'estimate_affine_shape':True,'domain_size_pooling':True})]:
    options=pycolmap.FeatureExtractionOptions({'max_image_size':-1,'num_threads':1,
        'sift':{'first_octave':-1,'max_num_features':16384,'peak_threshold':0.004,**extra}})
    extractor=pycolmap.FeatureExtractor.create(options,pycolmap.Device.cpu)
    for subject,name in cases:
        dataset=root/'photos_native'/subject
        boxes=json.loads((dataset/'boxes.json').read_text())
        if isinstance(boxes,list):boxes={b['image_name']:b for b in boxes}
        record=boxes[name]
        with Image.open(dataset/'images'/name) as image:
            width,height=image.size
            box=record['padded_bbox_xyxy']
            x0,y0=max(0,int(box[0])-32),max(0,int(box[1])-32)
            x1,y1=min(width,int(np.ceil(box[2]))+32),min(height,int(np.ceil(box[3]))+32)
            pixels=np.ascontiguousarray(image.crop((x0,y0,x1,y1)).convert('L'),dtype=np.uint8)
        start=time.monotonic()
        kp,desc=extractor.extract_from_uint8_array(pixels)
        matrix=np.asarray([[k.x+x0,k.y+y0,k.a11,k.a12,k.a21,k.a22] for k in kp],dtype=np.float32).reshape(-1,6)
        path=out/f'{variant}_{subject}_{Path(name).stem}.npz'
        np.savez(path,keypoints=matrix,descriptors=desc.data)
        row={'variant':variant,'subject':subject,'name':name,'seconds':time.monotonic()-start,'features':len(kp),'roi':[x0,y0,x1,y1],'image_size':[width,height],'npz':str(path)}
        report.append(row);print(json.dumps(row),flush=True)
        (out/'feature_report.json').write_text(json.dumps(report,indent=2))
