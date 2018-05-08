#!/bin/sh
set -e

CURRENT_DIR=`pwd`

cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:$CAFFE_DIR:`pwd`

cd $CURRENT_DIR

python solve.py
