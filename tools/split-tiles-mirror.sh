#!/bin/sh

# center split to get only full tiles of wanted size
# discard border tiles smaller than wanted size

# in case of resources exhausted error:
# find / -name "policy.xml"
# /etc/ImageMagick-6/policy.xml
# <policy domain="resource" name="disk" value="8GiB"/>

# wanted tile size
tile_x=900
tile_y=900

path=/tmp/airvoid

rm -rf ${path}
mkdir  ${path}

# calculate larger viewport size that fits splitting to full tiles
crop_size=$(identify -format "%[fx:(w - w % ${tile_x} + ${tile_x})]x%[fx:(h - h % ${tile_y} + ${tile_y})]" $1)
# expand image with right and bottom edge mirrored, split to tiles
# +100 in filename fixes alphabetic sort, similar to leading zeros
convert $1 +repage \
-set option:distort:viewport ${crop_size}+0+0 \
-virtual-pixel Mirror  -filter point  -distort SRT 0 \
+repage \
-crop ${tile_x}x${tile_y} \
-set filename:col "%[fx:page.x/${tile_x}+100]" \
-set filename:row "%[fx:page.y/${tile_y}+100]" \
+repage \
 "${path}/tile_%[filename:row]_%[filename:col].png"
