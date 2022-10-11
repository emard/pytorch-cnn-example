#!/usr/bin/env python3

# convert color ground truth image
# to semantic BW images, black background and white selected objects

# apt install python3-pil

from PIL import Image, ImageColor, ImageOps
import numpy

def prepimg(crop, name_out):
  img = Image.open("../Pictures/input.jpg")
  img = img.crop(crop)
  img.save("prepared/Image/" + name_out)

  img = Image.open("../Pictures/ground-truth.png")
  img = img.crop(crop)
  # img = img.convert("RGB") # should already be RGB
  pixdata = img.load()
  for y in range(img.size[1]):
    for x in range(img.size[0]):
        if pixdata[x, y][0] < 128 and pixdata[x, y][1] < 128: # detect not cyan by red and green channel
            pixdata[x, y] = (255, 255, 255) # not cyan -> white
        else: # otherwise
            pixdata[x, y] = (  0,   0,   0) # otherwise black
  img = ImageOps.grayscale(img)
  img.save("prepared/Semantic/1_stone/" + name_out)

  img = Image.open("../Slike/ground-truth.png")
  img = img.crop(crop)
  pixdata = img.load()
  for y in range(img.size[1]):
    for x in range(img.size[0]):
        if pixdata[x, y][0] > 127: # detect yellow by red channel
            pixdata[x, y] = (255, 255, 255) # yellow -> white
        else: # otherwise
            pixdata[x, y] = (  0,   0,   0) # otherwise black
  img = ImageOps.grayscale(img)
  img.save("prepared/Semantic/2_void/" + name_out)

for i in range(100):
  # input size
  xi = yi = 650
  # destination size
  x = y = 512
  xr = numpy.random.randint(0,xi-x)
  yr = numpy.random.randint(0,yi-y)
  prepimg( (  xr,  yr, xr+x, yr+y), "image%02d.png" % i)
