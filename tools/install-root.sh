#!/bin/sh

# 2023-11-22 testing=trixie
# /etc/apt/sources.list.d/debian-testing.list
# deb http://deb.debian.org/debian/ testing main contrib non-free non-free-firmware
# deb http://security.debian.org/debian-security testing-security main contrib non-free non-free-firmware

apt install python3-full python3-pip python3-virtualenv
apt install python3-pil python3-matplotlib python3-opencv python3-typing-extensions
apt install nvidia-driver

# to accept remote ssh X console:
# cp ssh/sshd_config /etc/ssh/sshd_config
# joe /etc/ssh/sshd_config
# X11Forwarding yes
# X11DisplayOffset 10
# X11UseLocalhost yes

# to fix out of memory with imagemagick
# sh /tmp/en480marker.sh 20231120-1533-2-sharp-seg.png 20231120-1533-2-sharp-seg-mark.png
# convert-im6.q16: cache resources exhausted `20231120-1533-2-sharp-seg.png' @ error/cache.c/OpenPixelCache/4119.
# convert-im6.q16: cache resources exhausted `20231120-1533-2-sharp-seg-mark.png' @ error/cache.c/OpenPixelCache/4119.
# convert-im6.q16: No IDATs written into file `20231120-1533-2-sharp-seg-mark.png' @ error/png.c/MagickPNGErrorHandler/1643.
# cp ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml

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
