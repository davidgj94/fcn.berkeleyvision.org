from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import vis
import matplotlib.pyplot as plt

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def vis_val(net, dataset):
    print '>>>', datetime.now(), 'Begin seg tests'
    n_iter = 0
    for val_img in dataset:
        
        n_iter += 1
        net.forward()
        
        if n_iter < 5:
        
            im = Image.open('/home/davidgj/projects_v2/fcn.berkeleyvision.org/data/roads/ROADS/PNGImages/{}.png'.format(val_img))
            plt.imshow(im)
            plt.show()
        
            score = net.blobs["score"].data[...][0,:,:,:]
            score = score.transpose((1,2,0))
            print score.shape
            print "-----------"
            label = np.argmax(score, axis=2)
            print label.shape
            print "-----------"
        
            plt.imshow(vis.vis_seg(im, label, vis.make_palette(2)))
            plt.show()
        

