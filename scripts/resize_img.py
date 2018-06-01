from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np

_dataset_dir = '../data/roads/'
dataset_dir = _dataset_dir + 'ROADS/'
png_images_dir = dataset_dir + 'PNGImages/'
img = Image.open(png_images_dir + sys.argv[1])

rate = int(sys.argv[2])
basewidth = img.size[0] / rate
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
plt.imshow(img)
plt.show()

width, height = img.size   # Get dimensions
bottom = np.arange(200, height, 5)
top = bottom - 200

for b, t in zip(bottom, top):
    
    plt.imshow(img.crop((0, int(t), width, int(b))))
    plt.show()
