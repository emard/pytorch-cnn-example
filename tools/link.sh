#!/bin/sh

directory=train

mkdir -p ${directory}/Image   ${directory}/Semantic/1_kamen   ${directory}/Semantic/2_rupa
rm    -f ${directory}/Image/* ${directory}/Semantic/1_kamen/* ${directory}/Semantic/2_rupa/* ${directory}/Semantic/1 ${directory}/Semantic/2
ln -sf 1_kamen ${directory}/Semantic/1
ln -sf 2_rupa  ${directory}/Semantic/2

for file in ${directory}/inputset/*0.*
do
  echo $file
  # filename with extension, without directory
  basefile=$(basename $file)
  # remove 0.jpg extension
  b=$(basename $file 0.jpg)
  # further remove 0.png extension
  b=$(basename $b 0.png)
  # filename without extension 0.jpg 0.png
  # echo $b
  # create symlinks
  ln -sf ../inputset/${basefile}  ${directory}/Image/${basefile}
  ln -sf ../../inputset/${b}1.png ${directory}/Semantic/1/${basefile}
  ln -sf ../../inputset/${b}2.png ${directory}/Semantic/2/${basefile}
done
