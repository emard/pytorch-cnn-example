#!/bin/sh

# from flat train set img0.jpg img1.png img2.png
# create directory tree train set

mkdir -p linkedset/Image linkedset/Semantic/1 linkedset/Semantic/2

for file in trainset/*0.*
do
  echo $file
  # filename with extension, without directory
  basefile=$(basename $file)
  # remove 0.jpg extension
  b=$(basename $file 0.jpg)
  # further remove 0.png extension
  b=$(basename $b 0.png)
  # filename without extension 0.jpg 0.png
  echo $b
  # create symlinks
  ln -sf ../../trainset/${basefile} linkedset/Image/${basefile}
  ln -sf ../../../trainset/${b}1.png linkedset/Semantic/1/${basefile}
  ln -sf ../../../trainset/${b}2.png linkedset/Semantic/2/${basefile}
done
