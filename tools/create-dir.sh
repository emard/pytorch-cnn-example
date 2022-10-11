#!/bin/sh

mkdir -p generated/saved_models
mkdir -p generated/Image
mkdir -p generated/Semantic/1_stone
mkdir -p generated/Semantic/2_void
ln -sf generated/Image/image000.jpg test-gen.jpg
cd generated/Semantic/
ln -sf 1_stone 1
ln -sf 2_void 2






