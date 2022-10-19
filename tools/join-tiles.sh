#!/bin/sh

# in case of resources exhausted error:
# find / -name "policy.xml"
# <policy domain="resource" name="disk" value="8GiB"/>

path=/tmp/airvoid

for first_tile in ${path}/tile_1??_100.png
do
  d=$(dirname $first_tile)
  f=$(basename -s _100.png $first_tile)
  echo ${d}/${f}
  convert ${d}/${f}_1??.png +append ${d}/${f}.png
  rm ${d}/${f}_1??.png
done
convert ${path}/tile_1??.png -append ${path}/full.png
rm ${path}/tile_1??.png
