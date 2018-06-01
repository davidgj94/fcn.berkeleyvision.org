import caffe
import numpy as np
from PIL import Image
import random
import skimage.io
import matplotlib.pyplot as plt
from itertools import islice
from pathlib import Path

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

class RoadsDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        #self.random = params.get('randomize', True)
        #self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        if 'train' not in self.split:
            self.img_dir = self.voc_dir + 'TrainImages/'
            self.label_dir = self.voc_dir + 'TrainLabels/'
        else:
            self.img_dir = self.voc_dir + 'TestImages/'
            self.label_dir = self.voc_dir + 'TestLabels/'
            
        p = Path(self.img_dir)
        indices = [glob.parts[-1] for glob in p.glob('*.png')]
        self.batch_size = params["batch_size"]
        self.batches = list(chunk(indices, self.batch_size))
        self.idx = 0
        #self.random = False

        # make eval deterministic
        #if 'train' not in self.split:
        #   self.random = False

        # randomization: seed and pick
        #if self.random:
        #    random.seed(self.seed)
        #    self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        #self.data = self.load_image(self.indices[self.idx])
        #self.label = self.load_label(self.indices[self.idx])
        self.data = self.load_image(self.batches[self.idx])
        self.label = self.load_label(self.batches[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        self.idx += 1

        # pick next input
        #if self.random:
        #    self.idx = random.randint(0, len(self.indices)-1)
        #else:
        #    self.idx += 1
        #    if self.idx == len(self.indices):
        #        self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, batch):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        batch_result = np.zeros((self.batch_size, 3, 200, 257))
        for img_idx in range(self.batch_size):
            im = Image.open('{}/{}'.format(self.img_dir, batch[img_idx]))
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean
            in_ = in_.transpose((2,0,1))
            batch_result[img_idx,:,:,:] = in_
        return batch_result


    def load_label(self, batch):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        batch_result = np.zeros((self.batch_size, 1, 200, 257))
        for img_idx in range(self.batch_size):
            im = Image.open('{}/{}'.format(self.label_dir, batch[img_idx]))
            label = np.array(im, dtype=np.uint8)
            label = label[np.newaxis, ...]
            batch_result[img_idx,:,:,:] = label
        return batch_result
