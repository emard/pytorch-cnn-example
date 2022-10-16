#!/bin/sh
# 1x adapter for 4K HD camera, 2K snapshot

# convert will re-encode jpg and loose detail
#convert $1 -units PixelsPerCentimeter -density 2320 $2

# exiftool doesn't re-encode jpg, keeps detail
exiftool -overwrite_original_in_place -XResolution=2320 -YResolution=2320 -ResolutionUnit=cm $1
