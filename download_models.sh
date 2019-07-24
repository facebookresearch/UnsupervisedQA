#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
HERE=`dirname "$0"`
HERE_ABS=`( cd "${HERE}" && pwd )`
DATA_DIR=${HERE_ABS}/data
BUCKET="https://dl.fbaipublicfiles.com/UnsupervisedQA"

mkdir -p ${DATA_DIR}

MODEL="subclause_ne_wh_heuristic"
echo "Downloading Model: ${MODEL}"
wget -P ${DATA_DIR} "${BUCKET}/${MODEL}.tar.gz"
tar -zxvf "${DATA_DIR}/${MODEL}.tar.gz" -C ${DATA_DIR}
rm "${DATA_DIR}/${MODEL}.tar.gz"

MODEL="subclause_ne"
echo "Downloading Model: ${MODEL}"
wget -P ${DATA_DIR} "${BUCKET}/${MODEL}.tar.gz"
tar -zxvf "${DATA_DIR}/${MODEL}.tar.gz" -C ${DATA_DIR}
rm "${DATA_DIR}/${MODEL}.tar.gz"


MODEL="sentence_ne"
echo "Downloading Model: ${MODEL}"
wget -P ${DATA_DIR} "${BUCKET}/${MODEL}.tar.gz"
tar -zxvf "${DATA_DIR}/${MODEL}.tar.gz" -C ${DATA_DIR}
rm "${DATA_DIR}/${MODEL}.tar.gz"


MODEL="sentence_np"
echo "Downloading Model: ${MODEL}"
wget -P ${DATA_DIR} "${BUCKET}/${MODEL}.tar.gz"
tar -zxvf "${DATA_DIR}/${MODEL}.tar.gz" -C ${DATA_DIR}
rm "${DATA_DIR}/${MODEL}.tar.gz"


