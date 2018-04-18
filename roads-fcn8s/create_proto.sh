#!/bin/sh
set -e

CURRENT_DIR=`pwd`
CAFFE_DIR="/home/david/projects/caffe/python"

cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:$CAFFE_DIR:`pwd`

cd $CURRENT_DIR

python net.py