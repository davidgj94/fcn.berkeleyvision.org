import caffe
import surgery, score, score_vis

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

results_path = 'results_{}/'.format(sys.argv[1])
if os.path.exists(results_path):
    shutil.rmtree(results_path, ignore_errors=True)
else:
    os.mkdir(results_path)
# init
caffe.set_mode_gpu()

voc_net = caffe.Net('../voc-fcn8s/deploy.prototxt', '../voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TRAIN)

solver = caffe.SGDSolver('solver.prototxt')

# surgeries

surgery.transplant(solver.net, voc_net, '_roads');

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print interp_layers
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/roads/ROADS/ImageSets/Segmentation/val.txt', dtype=str)
train = np.loadtxt('../data/roads/ROADS/ImageSets/Segmentation/train.txt', dtype=str)

niter = train.shape[0]
#niter = 190
nepoch = 2
#train_loss = np.zeros(nepoch)

for epoch in range(nepoch):
    solver.step(niter)
    #train_loss[epoch] = solver.net.blobs['loss'].data
    #solver.test_nets[0].share_with(solver.net)
    #val_net = solver.test_nets[0]
    #train_net = solver.net
    score.seg_tests(solver, results_path + 'iter_{}', val, layer='score')
    #score_vis.vis_val(train_net, train)
    
#plt.plot(range(nepoch), train_loss)  
#plt.show()

