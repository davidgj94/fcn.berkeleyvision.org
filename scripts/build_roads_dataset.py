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
    
p = Path(labeled_roads_dir)
roads = get_subdirs(p)

for road in roads:
    
    for glob in road.glob('*/*/sat.png'):
        
        disconn_mask_path = '/'.join(glob.parts[:-1]) + '/mask_disconn.png'
        other_mask_path = '/'.join(glob.parts[:-1]) + '/mask_other.png'
        sat_path = '/'.join(glob.parts)
        
        if not (os.path.exists(disconn_mask_path) and os.path.exists(other_mask_path)):
            print disconn_mask_path
            print 'Masks incompletas en {}'.format(sat_path)
            continue
        
        disconn_mask_length, disconn_mask_width = np.array(Image.open(disconn_mask_path)).shape[:-1]
        other_mask_length, other_mask_width = np.array(Image.open(other_mask_path)).shape[:-1]
        sat_length, sat_width = np.array(Image.open(sat_path)).shape[:-1]
        
        if disconn_mask_length == other_mask_length and disconn_mask_width == other_mask_width:
            if sat_length != disconn_mask_length and sat_width != disconn_mask_width:
                print 'Dimensiones sat.png no coinciden en {}'.format(sat_path)
                continue
        else:
            print 'Dimensiones mask_disconn.png y mask_other.png no coinciden en {}'.format(sat_path)
            continue
        
        new_name = ':'.join(glob.parts[-4:-1])
        
        if new_name == '39.273155_-3.446121:sec_3:sec_3_1':
            continue
        
        shutil.copy(sat_path, png_images_dir + new_name + '.png')
        new_dir = '{}/{}/'.format(segmentation_class_dir, new_name)
        os.makedirs(new_dir)
        shutil.copy(disconn_mask_path, new_dir + 'disconn.png')
        shutil.copy(other_mask_path, new_dir + 'other.png')
    
