#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

#
# Initialize tools and data paths
#

# main paths
HERE=`dirname "$0"`
HERE_ABS=`( cd "${HERE}" && pwd )`
UNSUPERVISED_NMT_PATH=${HERE_ABS}/UnsupervisedMT/NMT
TOOLS_PATH=${UNSUPERVISED_NMT_PATH}/tools

cd ${HERE_ABS}

# Download UNMT
if [ ! -d "${UNSUPERVISED_NMT_PATH}" ]; then
  echo "Cloning Unsupervised NMT from GitHub repository..."
  git clone https://github.com/facebookresearch/UnsupervisedMT.git
fi
echo "Unsupervised NMT found in: ${UNSUPERVISED_NMT_PATH}"


# create paths
mkdir -p $TOOLS_PATH

# UnsupervisedMT

# moses
MOSES=$TOOLS_PATH/mosesdecoder


# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext


# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"

# download the spacy model
echo "Downloading spacy model"
python -m spacy download en
