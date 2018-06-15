from PIL import Image
import sys
import numpy as np
import os
from pathlib import Path
import shutil
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import vis

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_raw_dir = dataset_dir + 'SegmentationClassRaw/'
save_dir = dataset_dir + 'Blended/'

if os.path.exists(save_dir):
    shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir)

p = Path(png_images_dir)
#pdb.set_trace()
for glob in p.glob("*.png"):
    img_name = glob.parts[-1]
    #pdb.set_trace()
    sat = Image.open(png_images_dir + img_name)
    label = np.array(Image.open(segmentation_class_raw_dir + img_name))
    #pdb.set_trace()
    vis_img = Image.fromarray(vis.vis_seg(sat, label, vis.make_palette(4)))
    vis_img.save(os.path.join(save_dir, img_name))

