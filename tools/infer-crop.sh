#!/bin/sh -e

tools/split-tiles.sh ${1}
./Infer.py /tmp/airvoid/tile*.png
tools/join-tiles.sh
echo "read original scale"
orig_scale=$(identify -format "-density %x -units %U" $1)
mogrify $orig_scale /tmp/airvoid/full_seg.png
echo "scale written to /tmp/airvoid/full_seg.png"
