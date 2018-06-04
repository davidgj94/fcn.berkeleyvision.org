from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
png_images_dir = dataset_dir + 'PNGImages/'
segmentation_class_raw_dir = dataset_dir + 'SegmentationClassRaw/'
images_cropped_dir = dataset_dir + 'TrainImages/'
labels_cropped_dir = dataset_dir + 'TrainLabels/'
images_test_dir = dataset_dir + 'TestImages/'
labels_test_dir = dataset_dir + 'TestLabels/'

if os.path.exists(images_cropped_dir):
    shutil.rmtree(images_cropped_dir, ignore_errors=True)
os.makedirs(images_cropped_dir)

if os.path.exists(labels_cropped_dir):
    shutil.rmtree(labels_cropped_dir, ignore_errors=True)
os.makedirs(labels_cropped_dir)

if os.path.exists(images_test_dir):
    shutil.rmtree(images_test_dir, ignore_errors=True)
os.makedirs(images_test_dir)

if os.path.exists(labels_test_dir):
    shutil.rmtree(labels_test_dir, ignore_errors=True)
os.makedirs(labels_test_dir)

crop_height = int(sys.argv[1])
crop_step = int(sys.argv[2])

#p = Path(png_images_dir)
split_train  = '{}/ImageSets/Segmentation/{}.txt'.format(dataset_dir, 'train')
split_test  = '{}/ImageSets/Segmentation/{}.txt'.format(dataset_dir, 'val')
indices_train = open(split_train, 'r').read().splitlines()
indices_test = open(split_test, 'r').read().splitlines()
#images_globs = p.glob('*.png')

for img_name in indices_train:
    
    #img_name = glob.parts[-1]
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
        cropped_img.save(images_cropped_dir + new_name)
        cropped_label.save(labels_cropped_dir + new_name)
 

for img_name in indices_test:
    
    #img_name = glob.parts[-1]
    image_dir = png_images_dir + img_name + '.png'
    label_dir = segmentation_class_raw_dir + img_name + '.png'
    img = Image.open(image_dir)
    img.save(images_test_dir + img_name + '.png')
    label = Image.open(label_dir)
    label.save(labels_test_dir + img_name + '.png')
    
    #width, height = img.size   # Get dimensions
    #bottom = np.arange(crop_height, height, crop_step)
    #top = bottom - crop_height
    
    #for index, (b, t) in enumerate(zip(bottom, top)):
        
        #cropped_img = img.crop((0, int(t), width, int(b)))
        #cropped_label = label.crop((0, int(t), width, int(b)))
        #new_name = '{}:{}.png'.format(os.path.splitext(img_name)[0], index)
        #cropped_img.save(images_test_dir + new_name)
        #cropped_label.save(labels_test_dir + new_name)
