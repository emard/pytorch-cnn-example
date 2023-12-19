#!/bin/sh
# usage:
# ./setqrcode.sh 12345 file.jpeg
exiftool -quiet -overwrite_original_in_place -ImageNumber=$1 $2
