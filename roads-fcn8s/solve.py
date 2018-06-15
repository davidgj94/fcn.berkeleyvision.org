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
results_path_train = results_path + 'train/'
results_path_val = results_path + 'val/'
snapshot_dir = "snapshot/{}/".format(sys.argv[1])

if not os.path.exists(results_path):
    os.mkdir(results_path)
    os.mkdir(results_path_train)
    os.mkdir(results_path_val)
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

niter = 90
nepoch = 5

train_loss = []
val_loss = []
train_acc = []
val_acc = []
train_iu = []
val_iu = []

if os.path.exists(results_path + 'results.p'):
    with open(results_path + 'results.p', 'rb') as f:
        train_loss , val_loss, train_acc, val_acc, train_iu, val_iu = pickle.load(f)

for epoch in range(nepoch):
    
    solver.step(niter)
    
    #train_loss_, train_acc_, train_iu_ = score.seg_tests_train(solver, results_path_train + 'iter_{}', train, layer='score')
    val_loss_, val_acc_, val_iu_ = score.seg_tests_val(solver,  results_path_val + 'iter_{}', val, layer='score')
    
    # train_loss.append(train_loss_)
    # train_acc.append(train_acc_)
    # train_iu.append(train_iu_)
    
    val_loss.append(val_loss_)
    val_acc.append(val_acc_)
    val_iu.append(val_iu_)

with open(results_path + 'results.p', 'wb') as f:
    pickle.dump((train_loss , val_loss, train_acc, val_acc, train_iu, val_iu), f)

# def get_class_score(scores, idx):
#     return [el[idx] for el in scores]

# num_epochs = range(len(train_loss))

# # Loss
# plt.plot(num_epochs, train_loss)
# plt.plot(num_epochs, val_loss)
# plt.grid()
# plt.legend(['Train', 'Val'])
# plt.title('Train/Val loss')
# plt.show()

# # Per-Class accuracy
# plt.plot(num_epochs, get_class_score(train_acc, 0))
# plt.plot(num_epochs, get_class_score(val_acc, 0))
# plt.title('Train/Val Class 0 Accuracy')
# plt.legend(['Train', 'Val'])
# plt.grid()
# plt.show()

# plt.plot(num_epochs, get_class_score(train_acc, 1))
# plt.plot(num_epochs, get_class_score(val_acc, 1))
# plt.title('Train/Val Class 1 Accuracy')
# plt.legend(['Train', 'Val'])
# plt.grid()
# plt.show()

# plt.plot(num_epochs, get_class_score(train_acc, 2))
# plt.plot(num_epochs, get_class_score(val_acc, 2))
# plt.title('Train/Val Class 2 Accuracy')
# plt.legend(['Train', 'Val'])
# plt.grid()
# plt.show()

# # Per-Class IU
# plt.plot(num_epochs, get_class_score(train_iu, 0))
# plt.plot(num_epochs, get_class_score(val_iu, 0))
# plt.title('Train/Val Class 0 IU')
# plt.legend(['Train', 'Val'])
# plt.grid()
# plt.show()

# plt.plot(num_epochs, get_class_score(train_iu, 1))
# plt.plot(num_epochs, get_class_score(val_iu, 1))
# plt.title('Train/Val Class 1 IU')
# plt.legend(['Train', 'Val'])
# plt.grid()
# plt.show()

# plt.plot(num_epochs, get_class_score(train_iu, 2))
# plt.plot(num_epochs, get_class_score(val_iu, 2))
# plt.title('Train/Val Class 2 IU')
# plt.legend(['Train', 'Val'])
# plt.grid()
# plt.show()


