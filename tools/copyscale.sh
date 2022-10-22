#!/bin/sh

orig_scale=$(identify -format "-density %x -units %U" $1)
mogrify $orig_scale $2
