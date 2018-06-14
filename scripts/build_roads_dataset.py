import shutil
from pathlib import Path
import os
import numpy as np
from PIL import Image
import shutil
import sys

def get_subdirs(p):
    return [x for x in p.iterdir() if x.is_dir()]

dataset_dir = 'data/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_dir = dataset_dir + 'SegmentationClass/'
segmentation_class_raw_dir = dataset_dir + 'SegmentationClassRaw/'
labeled_roads_dir = 'new_roads/'

if os.path.exists(dataset_dir):
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
        short_mask_path = '/'.join(glob.parts[:-1]) + '/mask_short.png'
        sat_path = '/'.join(glob.parts)
        
        if not os.path.exists(other_mask_path):
            print 'Masks incompletas en {}'.format(sat_path)
            continue
        
        sat = np.array(Image.open(sat_path))
        other_mask = np.array(Image.open(other_mask_path))
        
        if sat.shape == other_mask.shape:
            
            new_name = ':'.join(glob.parts[-4:-1])
            shutil.copy(sat_path, png_images_dir + new_name + '.png')
            new_dir = '{}/{}/'.format(segmentation_class_dir, new_name)
            os.makedirs(new_dir)
            shutil.copy(other_mask_path, new_dir + 'other.png')
            
            if os.path.exists(disconn_mask_path):
                disconn_mask = np.array(Image.open(disconn_mask_path))
                if sat.shape == disconn_mask.shape:
                    shutil.copy(disconn_mask_path, new_dir + 'disconn.png')
                else:
                    print 'Dimensiones no coinciden en {}'.format(sat_path)
            
            if os.path.exists(short_mask_path):
                short_mask = np.array(Image.open(short_mask_path))
                if sat.shape == short_mask.shape:
                    shutil.copy(short_mask_path, new_dir + 'short.png')
                else:
                    print 'Dimensiones no coinciden en {}'.format(sat_path)
                    
        else:
            print 'Dimensiones no coinciden en {}'.format(sat_path)
            
        
    
