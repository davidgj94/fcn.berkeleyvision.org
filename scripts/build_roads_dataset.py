import shutil
from pathlib import Path
import os
import numpy as np
from PIL import Image
import shutil
import sys

def get_subdirs(p):
    return [x for x in p.iterdir() if x.is_dir()]

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_dir = dataset_dir + 'SegmentationClass/'
segmentation_class_raw_dir = dataset_dir + 'SegmentationClassRaw/'
image_sets_dir = dataset_dir + 'ImageSets/Segmentation/'

labeled_roads_dir = sys.argv[1]

if os.path.exists(_dataset_dir):
    shutil.rmtree(dataset_dir, ignore_errors=True)
    
os.makedirs(dataset_dir)
os.makedirs(png_images_dir)
os.makedirs(segmentation_class_dir)
os.makedirs(segmentation_class_raw_dir)
os.makedirs(image_sets_dir)
os.mknod(image_sets_dir + 'train.txt')
os.mknod(image_sets_dir + 'val.txt')
os.mknod(image_sets_dir + 'trainval.txt')
    
p = Path(labeled_roads_dir)
roads = get_subdirs(p)
train_split = 0.7

train_size = 0
val_size = 0
idx = 0

for road in roads:
    
    for glob in road.glob('*/*/sat.png'):
        
        idx += 1
        
        mask_path = '/'.join(glob.parts[:-1]) + '/mask_road.png'
        sat_path = '/'.join(glob.parts)
        
        if not os.path.exists(mask_path):
            continue
        
        mask_length, mask_width = np.array(Image.open(mask_path)).shape[:-1]
        sat_length, sat_width = np.array(Image.open(sat_path)).shape[:-1]
        
        if mask_length != sat_length or mask_width != sat_width:
            continue
        
        new_name = ':'.join(glob.parts[-4:-1])
        
        if new_name == '39.273155_-3.446121:sec_3:sec_3_1':
            continue

        if idx % 3 != 0:
            train_size += 1
            desc_txt = 'train.txt'
        else:
            val_size += 1
            desc_txt = 'val.txt'
            
        with open(image_sets_dir + desc_txt, 'a') as txt:
            txt.write(new_name + '\n')

        with open(image_sets_dir + 'trainval.txt', 'a') as txt:
            txt.write(new_name + '\n')
        
        shutil.copy(sat_path, png_images_dir + new_name + '.png')
        shutil.copy(mask_path, segmentation_class_dir + new_name + '.png')
    
print('train_size: {}\n'.format(str(train_size)))
print('val_size: {}\n'.format(str(val_size)))
print('trainval_size: {}\n'.format(str(train_size + val_size)))
