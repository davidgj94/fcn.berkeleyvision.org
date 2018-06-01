import caffe

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
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

_snapshor_dir = "snapshot/"
snapshot_dir = _snapshor_dir + sys.argv[1]
solver = caffe.SGDSolver('solver.prototxt')

p = Path(snapshot_dir)
states = sorted(list(p.glob('*.solverstate')), key=get_iter)
train_loss = np.zeros(len(states))
val_loss = np.zeros(len(states))
nit = 0

for solver_state in states:
    print '/'.join(solver_state.parts)
    solver.restore('/'.join(solver_state.parts))
    train_loss[nit] = solver.net.blobs['loss'].data
    val_loss[nit] = solver.test_nets[0].blobs['loss'].data
    nit += 1
 
plt.plot(range(nit), train_loss) 
plt.title('Train Loss')
plt.show()

plt.plot(range(nit), val_loss) 
plt.title('Val Loss')
plt.show()

plt.plot(range(nit), train_loss, label='train')
plt.plot(range(nit), val_loss, label='val') 
plt.title('Train/Val loss')
plt.legend()
plt.show()



    
