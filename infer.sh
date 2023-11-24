#!/bin/bash
source torch/bin/activate
exec ./infer.py $*
