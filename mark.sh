#!/bin/sh

for filename in $*
do
filename_without_extension="${filename%.*}"
extension="${filename##*.}"
# echo $filenoext $extension
../en480img/en480img.py $filename
sh /tmp/en480marker.sh $filename ${filename_without_extension}-mark.$extension
exiftool -quiet -overwrite_original_in_place -TagsFromFile ${filename} -exif:ImageNumber ${filename_without_extension}-mark.$extension
echo Output: ${filename_without_extension}-mark.$extension
done
