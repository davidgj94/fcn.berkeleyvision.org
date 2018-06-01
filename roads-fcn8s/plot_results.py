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

results_path = 'results/{}/'.format(sys.argv[1])
results_path_train = results_path + 'train/'
results_path_val = results_path + 'val/'

if os.path.exists(results_path + 'results.p'):
    with open(results_path + 'results.p', 'rb') as f:
        train_loss , val_loss, train_acc, val_acc, train_iu, val_iu = pickle.load(f)
else:
    print 'No hay resultados'
    sys.exit()
        
def get_class_score(scores, idx):
    return [el[idx] for el in scores]

num_epochs = range(len(train_loss))

# Loss
plt.plot(num_epochs, train_loss)
plt.plot(num_epochs, val_loss)
plt.grid()
plt.legend(['Train', 'Val'])
plt.title('Train/Val loss')
plt.show()

# Per-Class accuracy
plt.plot(num_epochs, get_class_score(train_acc, 0))
plt.plot(num_epochs, get_class_score(val_acc, 0))
plt.title('Train/Val Class 0 Accuracy')
plt.legend(['Train', 'Val'])
plt.grid()
plt.show()

plt.plot(num_epochs, get_class_score(train_acc, 1))
plt.plot(num_epochs, get_class_score(val_acc, 1))
plt.title('Train/Val Class 1 Accuracy')
plt.legend(['Train', 'Val'])
plt.grid()
plt.show()

plt.plot(num_epochs, get_class_score(train_acc, 2))
plt.plot(num_epochs, get_class_score(val_acc, 2))
plt.title('Train/Val Class 2 Accuracy')
plt.legend(['Train', 'Val'])
plt.grid()
plt.show()

# Per-Class IU
plt.plot(num_epochs, get_class_score(train_iu, 0))
plt.plot(num_epochs, get_class_score(val_iu, 0))
plt.title('Train/Val Class 0 IU')
plt.legend(['Train', 'Val'])
plt.grid()
plt.show()

plt.plot(num_epochs, get_class_score(train_iu, 1))
plt.plot(num_epochs, get_class_score(val_iu, 1))
plt.title('Train/Val Class 1 IU')
plt.legend(['Train', 'Val'])
plt.grid()
plt.show()

plt.plot(num_epochs, get_class_score(train_iu, 2))
plt.plot(num_epochs, get_class_score(val_iu, 2))
plt.title('Train/Val Class 2 IU')
plt.legend(['Train', 'Val'])
plt.grid()
plt.show()
