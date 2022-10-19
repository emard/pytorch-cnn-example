#!/bin/sh

# in case of resources exhausted error:
# find / -name "policy.xml"
# <policy domain="resource" name="disk" value="8GiB"/>

path=/tmp/airvoid
#ext=.png
ext=_seg.png

for first_tile in ${path}/tile_1??_100${ext}
do
  d=$(dirname $first_tile)
  f=$(basename -s _100${ext} $first_tile)
  echo ${d}/${f}
  convert ${d}/${f}_1??${ext} +append ${d}/${f}${ext}
  rm ${d}/${f}_1??${ext}
done
convert ${path}/tile_1??${ext} -append ${path}/full${ext}
rm ${path}/tile_1??${ext}
