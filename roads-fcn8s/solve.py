import caffe
import surgery, score, score_vis

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass


# init
caffe.set_mode_gpu()

voc_net = caffe.Net('../voc-fcn8s/deploy.prototxt', '../voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TRAIN)

solver = caffe.SGDSolver('solver.prototxt')

# surgeries

surgery.transplant(solver.net, voc_net);

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/roads/ROADS/ImageSets/Segmentation/val.txt', dtype=str)
train = np.loadtxt('../data/roads/ROADS/ImageSets/Segmentation/train.txt', dtype=str)

for _ in range(5):
    solver.step(190)
    solver.test_nets[0].share_with(solver.net)
    val_net = solver.test_nets[0]
    train_net = solver.net
    #score.seg_tests(solver, False, val, layer='score')
    score_vis.vis_val(train_net, train)
