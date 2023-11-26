#!/bin/sh
filename=$1
outfile=$2
filename_without_extension="${filename%.*}"
extension="${filename##*.}"
# echo $filenoext $extension
../en480img/en480img.py $filename
sh /tmp/en480marker.sh $filename ${filename_without_extension}-mark.$extension
echo Output: ${filename_without_extension}-mark.$extension
