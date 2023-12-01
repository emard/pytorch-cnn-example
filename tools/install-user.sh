#!/bin/sh

# https://pytorch.org/get-started/locally

# for NVIDIA 3090 and CUDA 11.5.x
# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# for NVIDIA 3090 and CUDA 11.6
# pip install torch torchvision

# for new debian sytems
cd /tmp
# cd ~/doc/css/beton/udio-pora-ocvrsli-beton/pytorch-cnn-example
python3 -m virtualenv torch
. torch/bin/activate
# (torch) davor@nuc1:/tmp$
pip install torch torchvision matplotlib opencv-python
