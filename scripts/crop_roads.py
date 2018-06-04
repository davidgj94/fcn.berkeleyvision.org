from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
image_sets_dir = dataset_dir + 'ImageSets/Segmentation/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_raw_dir = dataset_dir + 'SegmentationClassRaw/'

images_cropped_dir = dataset_dir + 'CroppedImages/'
labels_cropped_dir = dataset_dir + 'CroppedLabels/'

if os.path.exists(images_cropped_dir):
    shutil.rmtree(images_cropped_dir, ignore_errors=True)
os.makedirs(images_cropped_dir)

if os.path.exists(labels_cropped_dir):
    shutil.rmtree(labels_cropped_dir, ignore_errors=True)
os.makedirs(labels_cropped_dir)

if os.path.exists(limage_sets_dir):
    shutil.rmtree(image_sets_dir, ignore_errors=True)
os.makedirs(image_sets_dir)
os.mknod(image_sets_dir + 'train.txt')
os.mknod(image_sets_dir + 'val.txt')
os.mknod(image_sets_dir + 'trainval.txt')

crop_height = int(sys.argv[1])
crop_step = int(sys.argv[2])

p = Path(png_images_dir)
indices = [glob.parts[-1] for glob in p.glob('*.png')]

train_split = 0.7
train_size = 0
val_size = 0
idx = 0
for img_name in indices:
    
    image_dir = png_images_dir + img_name + '.png'
    label_dir = segmentation_class_raw_dir + img_name + '.png'
    img = Image.open(image_dir)
    label = Image.open(label_dir)
    
    width, height = img.size   # Get dimensions
    bottom = np.arange(crop_height, height, crop_step)
    top = bottom - crop_height
    
    for index, (b, t) in enumerate(zip(bottom, top)):
        
        cropped_img = img.crop((0, int(t), width, int(b)))
        cropped_label = label.crop((0, int(t), width, int(b)))
        new_name = '{}:{}.png'.format(os.path.splitext(img_name)[0], index)
        
        idx += 1
        
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
            
        cropped_img.save(images_cropped_dir + new_name)
        cropped_label.save(labels_cropped_dir + new_name)
        
shutil.rmtree(png_images_dir, ignore_errors=True)
shutil.rmtree(segmentation_class_raw_dir, ignore_errors=True)

print('train_size: {}\n'.format(str(train_size)))
print('val_size: {}\n'.format(str(val_size)))
print('trainval_size: {}\n'.format(str(train_size + val_size)))

