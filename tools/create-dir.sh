#!/bin/sh

mkdir -p train/saved_models
mkdir -p train/Image
mkdir -p train/Segmented
mkdir -p train/Semantic/1_stone
mkdir -p train/Semantic/2_void
ln -sf train/Image/image000.jpg test-gen.jpg
cd train/Semantic/
ln -sf 1_stone 1
ln -sf 2_void 2






