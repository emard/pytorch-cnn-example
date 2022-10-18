#!/bin/sh

# convert/mogrify will re-encode jpg and loose detail
# convert can add scale to .png

#convert $1 -density 2320 -units PixelsPerCentimeter $2
# mogrify is in-place convert
mogrify -density 2320 -units PixelsPerCentimeter $1
#mogrify -density 4800 -units PixelsPerInch $1

# exiftool doesn't re-encode jpg, so it keeps detail
# exiftools can modify existing scale of some .png
# exiftools can't add scale to .png which doesn't have scale

#exiftool -overwrite_original_in_place -XResolution=2320 -YResolution=2320 -ResolutionUnit=cm $1
#exiftool -overwrite_original_in_place -XResolution=4800 -YResolution=4800 -ResolutionUnit=inch $1

# check result with
identify -verbose $1
