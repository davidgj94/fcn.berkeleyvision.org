import caffe
import surgery, score, score_vis

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil
import pickle
import time

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

results_path = 'results/{}/'.format(sys.argv[1])
if not os.path.exists(results_path):
    #shutil.rmtree(results_path, ignore_errors=True)
    os.mkdir(results_path)
#else:
#    os.mkdir(results_path)
    
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
nepoch = 1

print train.shape[0]
sys.exit()

train_loss = []
val_loss = []
acc = []
iu = []

if os.path.exists(results_path + 'results.p'):
    with open(results_path + 'results.p', 'rb') as f:
        train_loss , val_loss, acc, iu = pickle.load(f)
    
for epoch in range(nepoch):
    solver.step(niter)
    train_loss_ = solver.net.blobs['loss'].data
    val_loss_, acc_, iu_ = score.seg_tests(solver, results_path + 'iter_{}', val, layer='score')
    train_loss.append(train_loss_)
    val_loss.append(val_loss_)
    acc.append(acc_)
    iu.append(iu_)

plt.plot(np.arange(len(train_loss)), train_loss)
plt.show()

with open(results_path + 'results.p', 'wb') as f:
    pickle.dump((train_loss , val_loss, acc, iu), f) 

    

