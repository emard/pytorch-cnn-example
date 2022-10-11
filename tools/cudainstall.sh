#!/bin/sh

apt install python3-pil python3-matplotlib python3-opencv python3-typing-extensions
apt install nvidia-driver

# for NVIDIA 3090 and CUDA 11.5.x
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
