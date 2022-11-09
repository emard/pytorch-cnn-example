#!/usr/bin/env python3

# example copy image by iterating
# smaller tiles of equal size

import sys
from PIL import Image, ImageOps

# allow to load large images
Image.MAX_IMAGE_PIXELS = None

# global tile size
width = height = 100

# crop to full tiles and copy
def process_tiles(in_filename, out_filename):
  in_img = Image.open(in_filename)
  (w, h) = in_img.size
  print(in_img.mode)
  #print(w, h)
  # create empty output image
  out_img = Image.new(mode=in_img.mode, size=(w, h))
  xclamp = w-width
  yclamp = h-height
  y = 0
  while y < h:
    if y > yclamp: y = yclamp
    #print()
    #print("y=",y,"x=",end="")
    x = 0
    while x < w:
      if x > xclamp: x = xclamp
      #print(x,end=" ")
      tile = in_img.crop((x,y,x+width,y+height))
      # TODO: semantic segmentation if tile
      out_img.paste(tile, (x,y)) 
      x += width
    y += height
  out_img.save(out_filename)

process_tiles("/tmp/in.jpg", "/tmp/out.jpg")
