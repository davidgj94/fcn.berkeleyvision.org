#!/bin/sh
set -e

# Helper function to download and unpack VOC 2012 dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}"
  fi
  echo "Uncompressing ${FILENAME}"
  uzip ${FILENAME}
}

download_() {
  local BASE_URL=${1}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ..."
    wget -nd -c "${BASE_URL}"
  fi
  
}


CURRENT_DIR=$(pwd)
DATA_DIR="../data"
VOCFCN8S_DIR="../voc-fcn8s"

cd ${DATA_DIR}
BASE_URL="https://drive.google.com/open?id=1hiikZxzCx2zVInQAuo35b1J8jfZaVotK"
FILENAME="roads.zip"
download_and_uncompress ${BASE_URL} ${FILENAME}

cd ${CURRENT_DIR}
cd ${VOCFCN8S_DIR}
BASE_URL="http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel"
download_ ${BASE_URL}

