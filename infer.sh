#!/bin/sh
. torch/bin/activate
nice -n +20 ./infer.py $*
# shell copy additional tags
# avoid adding code to complicate infer.py source
for filename in $*
do
filename_without_extension="${filename%.*}"
extension="${filename##*.}"
exiftool -quiet -overwrite_original_in_place -TagsFromFile ${filename} -exif:ImageNumber ${filename_without_extension}-seg.png
done
