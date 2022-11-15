#!/usr/bin/env python3

# how many images to create
count=1

if 0:
  # dimensions of generated images FULL HD
  width=1920
  height=1080
  dpi=100

  # how many voids and stones per image
  void_count=5000
  # poisson distribution
  void_lambda=5
  void_size=1

  stone_count=200
  # poisson distribution
  stone_lambda=5
  stone_size=6

if 1:
  # dimensions of generated images 100x142 mm @ 2400 DPI
  width=9448
  height=13417
  dpi=2400

  # how many voids and stones per image
  void_count=50000
  # poisson distribution
  void_lambda=5
  void_size=2

  stone_count=200
  # poisson distribution
  stone_lambda=5
  stone_size=50


import cv2 
import numpy as np
from PIL import Image # PIL required only to write image with DPI metadata

def circles(xpos,ypos,d,color):
  global image
  n = xpos.size
  for i in range(xpos.size):
    image = cv2.circle(image,(xpos[i],ypos[i]), d[i], color,cv2.FILLED)

black=(0,0,0)
gray=(127,127,127)
white=(255,255,255)
green=(140,160,140)
orange=(0,155,255)
purple=(55,0,55)
cyan=(155,155,0)

def generate(i):
  global image

  # generate arrays of positions and sizes
  xpstone=np.random.randint(width, size=stone_count)
  ypstone=np.random.randint(height, size=stone_count)
  rstone=np.random.poisson(stone_lambda, size=stone_count)*stone_size

  xpvoid=np.random.randint(width, size=void_count)
  ypvoid=np.random.randint(height, size=void_count)
  rvoid=np.random.poisson(void_lambda, size=void_count)*void_size

  # generate color image with everything
  image = np.zeros((height,width,3), np.uint8)
  image[:,:] = purple
  circles(xpstone,ypstone,rstone,green)
  circles(xpvoid,ypvoid,rvoid,orange)
  # image generated, write
  #cv2.imwrite("generated/Image/image%03d.jpg" % i, image)
  image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  image_pil.save("generated/Image/image%03d.jpg" % i, dpi=(dpi,dpi))

  # generate type 1 (stone) annotated image
  image = np.zeros((height,width,3), np.uint8)
  circles(xpstone,ypstone,rstone,white) # draw stones
  # semantic for stones generated, write
  image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
  image_pil.save("generated/Semantic/1_stone/image%03d.png" % i, dpi=(dpi,dpi))

  # generate type 2 (void) annotated image
  image = np.zeros((height,width,3), np.uint8)
  circles(xpvoid,ypvoid,rvoid,white) # draw voids
  circles(xpstone,ypstone,rstone,black) # erase stones
  # semantic for voids generated, write
  image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
  image_pil.save("generated/Semantic/2_void/image%03d.png" % i, dpi=(dpi,dpi))

  # generate ideally segmented image (to compare with output of Infer.py)
  image = np.zeros((height,width,3), np.uint8)
  circles(xpvoid,ypvoid,rvoid,white) # draw voids
  circles(xpstone,ypstone,rstone,gray) # gray stones
  # ideally segmented image, write
  image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
  image_pil.save("generated/Segmented/image%03d.png" % i, dpi=(dpi,dpi))

for i in range(count):
  generate(i)

#cv2.imshow("Green Filled Circle on Image",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(np.random.randint(900, size=10).size)
