import shutil
from pathlib import Path
import shutil
from os.path import splitext
import pdb

def get_subdirs_names(p):
    return [x.parts[-1] for x in p.iterdir() if x.is_dir()]

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_dir = dataset_dir + 'SegmentationClass/'
    
p_img = Path(png_images_dir)
p_mask = Path(segmentation_class_dir)

img_names = set([splitext(glob.parts[-1])[0] for glob in p_img.glob("*.png")])
masks_folders = set(get_subdirs_names(p_mask))
to_delete_masks = list(masks_folders - img_names)

for mask_folder in to_delete_masks:
    shutil.rmtree(segmentation_class_dir + mask_folder, ignore_errors=True)