#!/bin/sh

apt install python3-pil python3-matplotlib python3-opencv python3-typing-extensions
apt install nvidia-driver

cp xorg.conf.d/intel.conf /etc/X11/xorg.conf.d/intel.conf
# service gdm restart
# nvidia-smi should report
# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1109      G   /usr/lib/xorg/Xorg                  4MiB |
# +-----------------------------------------------------------------------------+



