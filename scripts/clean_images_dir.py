import shutil
from pathlib import Path
import shutil
from os.path import splitext
import os.remove
import pdb

def get_set(p):
    return set([splitext(glob.parts[-1])[0] for glob in p.glob("*.png")])

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
blended_dir = dataset_dir + 'Blended/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_dir = dataset_dir + 'SegmentationClass/'
segmentation_class_raw_dir = dataset_dir + 'SegmentationClassRaw/'

p_blended = Path(blended_dir)
p_img = Path(png_images_dir)

img_names = get_set(p_img)
blended_names = get_set(p_blended)

to_delete_imgs = list(img_names - blended_names)

for img in to_delete_imgs:
    shutil.rmtree(segmentation_class_dir + img, ignore_errors=True)
    os.remove(segmentation_class_raw_dir + '{}.png'.format(img), ignore_errors=True)
    os.remove(png_images_dir + '{}.png'.format(img), ignore_errors=True)