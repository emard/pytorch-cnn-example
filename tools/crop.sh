#!/bin/sh

# usage ./crop.sh 1000x1000+500+500 img.jpg

ext=$(echo $2 | sed -s "s/.*[.]\(.*\)/\1/g")

dir=$(dirname  $2)
base=$(basename --suffix=.$ext $2)

output=${dir}/${base}-$1.${ext}

convert $2 -crop $1 $output

echo $output
