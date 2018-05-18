import caffe
import surgery, score, score_vis

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil
import pickle
import time
from pathlib import Path
import parse

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

def get_iter(solver_state):
    format_string = 'solver_iter_{:0>9}.solverstate'
    parsed = parse.parse(format_string, solver_state.parts[-1])
    return int(parsed[0])

results_path = 'results/{}/'.format(sys.argv[1])
snapshot_dir = "snapshot/{}/".format(sys.argv[1])

if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)
# init
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')

p = Path(snapshot_dir)
states = sorted(list(p.glob('*.solverstate')), key=get_iter)

if states:
    solver.restore('/'.join(states[-1].parts))
else:
    voc_net = caffe.Net('../voc-fcn8s/deploy.prototxt', '../voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TRAIN)
    # surgeries
    surgery.transplant(solver.net, voc_net, '_roads');
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/roads/ROADS/ImageSets/Segmentation/val.txt', dtype=str)
train = np.loadtxt('../data/roads/ROADS/ImageSets/Segmentation/train.txt', dtype=str)

niter = train.shape[0]
nepoch = 2

train_loss = []
val_loss = []
train_acc = []
val_acc = []
train_iu = []
val_iu = []

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

with open(results_path + 'results.p', 'wb') as f:
    pickle.dump((train_loss , val_loss, acc, iu), f) 

plt.plot(np.arange(len(train_loss)), train_loss)
plt.grid()
plt.title('Train loss')
plt.show()

plt.plot(np.arange(len(val_loss)), val_loss)
plt.title('Val loss')
plt.grid()
plt.show()

plt.plot(np.arange(len(acc)), acc)
plt.title('Per Class Accuracy')
plt.grid()
plt.show()

plt.plot(np.arange(len(iu)), iu)
plt.grid()
plt.title('Iu')
plt.show()

