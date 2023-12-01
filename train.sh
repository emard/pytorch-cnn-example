#!/bin/sh
. torch/bin/activate
exec nice -n +20 ./train.py $*
