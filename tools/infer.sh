#!/bin/sh -e

tools/split-tiles-mirror.sh ${1}
./Infer.py /tmp/airvoid/tile*.png
tools/join-tiles.sh
echo "read original size and scale"
orig_size_scale=$(identify -format "-crop %wx%h+0+0 -density %x -units %U" $1)
echo "crop to original size and set original scale"
mogrify $orig_size_scale /tmp/airvoid/full_seg.png
echo "scale written to /tmp/airvoid/full_seg.png"
