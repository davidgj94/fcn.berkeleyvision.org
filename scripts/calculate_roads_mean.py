import shutil
from pathlib import Path
import os
import numpy as np
from PIL import Image
import shutil
import sys

def get_subdirs(p):
    return [x for x in p.iterdir() if x.is_dir()]

labeled_roads_dir = sys.argv[1]
p = Path(labeled_roads_dir)
roads = get_subdirs(p)

num_roads = 0
roads_mean = np.array([0.0, 0.0, 0.0])

for road in roads:
    
    for glob in road.glob('*/*/sat.png'):
        
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
        
        num_roads += 1
        img = np.array(Image.open(sat_path))
        roads_mean += np.mean(img, axis=(0, 1))
        
        
roads_mean /= num_roads
print('roads_mean: {}\n'.format(str(roads_mean))) #[109.31270171 112.73650684 107.62839719]
