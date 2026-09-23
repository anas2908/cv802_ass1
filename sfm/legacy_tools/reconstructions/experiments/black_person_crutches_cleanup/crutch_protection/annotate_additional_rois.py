#!/usr/bin/env python3
"""Image-inspected visible-crutch corridors for independent geometric QC only."""
from pathlib import Path
import json,math
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
SFM=HERE.parents[3]
SOURCE=SFM/'reconstructions/black_shirt_crutches/images'
VIEWS={
 'photos_iphone_11_26mm/0038_IMG_6673.jpg':{
  'near':{'half_width_fraction_of_image_width':0.009,'polylines':[
   [[.620,.368],[.596,.472],[.559,.609]],
   [[.554,.626],[.521,.746],[.491,.794],[.462,.904]],
   [[.500,.619],[.475,.731],[.477,.791],[.450,.901]],
   [[.481,.796],[.454,.904],[.448,.937],[.445,.953]],
   [[.625,.354],[.637,.348],[.643,.353]],
   [[.537,.611],[.545,.613]],
  ],'notes':'Near crutch visible fork branches are separate polylines. Omit hand/arm occlusion; orange pad and grip only where visible.'},
  'far_visible_lower':{'half_width_fraction_of_image_width':0.008,'polylines':[
   [[.690,.601],[.663,.646],[.651,.713],[.648,.721]],
  ],'notes':'Only short exposed lower section behind near leg; rest is occluded and unannotated.'},
 },
 'photos_iphone_11_26mm/0112_IMG_8430.jpg':{
  'near':{'half_width_fraction_of_image_width':0.009,'polylines':[
   [[.444,.347],[.459,.436],[.480,.548]],
   [[.495,.592],[.511,.695],[.502,.762],[.523,.869]],
   [[.403,.541],[.417,.610],[.440,.708],[.478,.768],[.502,.871]],
   [[.491,.761],[.513,.873],[.520,.914],[.524,.930]],
   [[.419,.577],[.425,.579]],
   [[.455,.322],[.458,.322]],
  ],'notes':'One clearly visible near crutch. Upper tubes behind shirt and hand are not joined through occlusion; far crutch is not visibly separable.'},
 },
}

VIEWS.update({
 'photos_iphone_11_26mm/0097_IMG_8437.jpg':{
  'near_image_left':{'half_width_fraction_of_image_width':0.010,'polylines':[
   [[.354,.356],[.373,.518],[.379,.577]],
   [[.383,.613],[.394,.727],[.374,.792],[.388,.917]],
   [[.305,.614],[.316,.737],[.347,.791],[.360,.917]],
   [[.364,.797],[.375,.921],[.379,.960],[.383,.979]],
   [[.354,.346],[.367,.337],[.371,.341]],
  ],'notes':'Near crutch with both visible fork branches; omit hand occlusion. Physical identity is not assigned by this annotation.'},
  'far_image_right_visible_lower':{'half_width_fraction_of_image_width':0.008,'polylines':[
   [[.455,.519],[.467,.601],[.456,.631],[.472,.708]],
   [[.451,.635],[.458,.706]],
   [[.452,.635],[.467,.711],[.469,.732],[.471,.740]],
  ],'notes':'Far upper support and hand region are hidden; only visible lower fork/shaft/foot are marked.'},
 },
 'photos_iphone_11_26mm/0088_IMG_8444.jpg':{
  'near_image_left':{'half_width_fraction_of_image_width':0.009,'polylines':[
   [[.294,.348],[.289,.466],[.285,.581]],
   [[.285,.614],[.277,.724],[.253,.794],[.249,.920]],
   [[.224,.610],[.219,.724],[.232,.772],[.235,.794],[.227,.918]],
   [[.244,.795],[.239,.923],[.237,.961],[.237,.980]],
   [[.288,.338],[.305,.329],[.313,.333]],
  ],'notes':'Two visible near fork branches and lower stem/foot; omit hand occlusion.'},
  'far_image_right':{'half_width_fraction_of_image_width':0.008,'polylines':[
   [[.466,.346],[.483,.451],[.490,.490]],
   [[.498,.540],[.517,.624],[.515,.658],[.535,.741]],
   [[.478,.539],[.491,.600],[.510,.656],[.525,.740]],
   [[.511,.660],[.528,.743],[.534,.771],[.538,.778]],
   [[.472,.315],[.473,.318]],
  ],'notes':'Far support/tube sections visible along arm; wrist/hand occlusion omitted. Separate fork branches and foot.'},
 },
})

def main():
 output={'schema_version':1,'coordinate_system':'Normalized source image x/width, y/height; physically upright baseline JPEGs, top-left origin. Corridor distance is in pixels with half-width times source width.','source_dataset':str(SOURCE.parent),'purpose':'Provisional visible-crutch annotations for independent existing-point protection QC; no filtering or geometry creation.','ready_for_filter':False,'physical_identity':'Labels near/far/image-left/right describe each photograph only; associate physical crutches using calibrated projections.','views':VIEWS}
 (HERE/'additional_rois_normalized.json').write_text(json.dumps(output,indent=2)+'\n')
 tiles=[]
 for name,records in VIEWS.items():
  with Image.open(SOURCE/name) as im: original=im.convert('RGB');original.thumbnail((720,960),Image.Resampling.LANCZOS)
  overlay=Image.new('RGBA',original.size,(0,0,0,0));draw=ImageDraw.Draw(overlay)
  for i,(label,record) in enumerate(records.items()):
   color=[(255,65,30),(30,220,255),(255,210,0)][i%3];radius=record['half_width_fraction_of_image_width']*original.width
   for line in record['polylines']:
    points=[(x*original.width,y*original.height) for x,y in line]
    draw.line(points,fill=color+(75,),width=round(radius*2),joint='curve')
    for x,y in points:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color+(75,))
  combined=Image.alpha_composite(original.convert('RGBA'),overlay).convert('RGB');d=ImageDraw.Draw(combined)
  for i,(label,record) in enumerate(records.items()):
   color=[(255,65,30),(30,220,255),(255,210,0)][i%3]
   for line in record['polylines']:
    points=[(x*original.width,y*original.height) for x,y in line]
    d.line(points,fill=color,width=2,joint='curve')
    for x,y in points:d.ellipse((x-2,y-2,x+2,y+2),fill='white')
   d.text((8,18+i*18),label,fill=color,stroke_width=1,stroke_fill='black')
  tile=Image.new('RGB',(1440,1000),'white');tile.paste(original,(0,32));tile.paste(combined,(720,32));d=ImageDraw.Draw(tile);d.text((8,8),name+' | original / provisional visible-only crutch corridors',fill='black')
  tile.save(HERE/('additional_roi_'+Path(name).stem+'.jpg'),quality=96);tiles.append(tile)
 sheet=Image.new('RGB',(1440,1000*len(tiles)),'white')
 for i,tile in enumerate(tiles):sheet.paste(tile,(0,1000*i))
 sheet.save(HERE/'additional_roi_overlay_grid.jpg',quality=94)
 print('Wrote',len(VIEWS),'provisional image-grounded ROI views.')
if __name__=='__main__':main()
