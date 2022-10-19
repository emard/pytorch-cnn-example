#!/bin/sh
# 0 as x or y describes full size

# in case of resources exhausted error:
# find / -name "policy.xml"
# <policy domain="resource" name="disk" value="8GiB"/>

path=/tmp/airvoid

rm -rf ${path}
mkdir ${path}
convert $1 -crop 900x900 \
-set filename:row "%[fx:page.y/900+100]" \
-set filename:col "%[fx:page.x/900+100]" \
 "${path}/tile_%[filename:row]_%[filename:col].png"
