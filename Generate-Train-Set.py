#!/usr/bin/env python3

# dimensions of generated images
width=1920
height=1080

import cv2 
import numpy as np

def circles(xpos,ypos,d,color):
  global image
  n = xpos.size
  for i in range(xpos.size):
    image = cv2.circle(image,(xpos[i],ypos[i]), d[i], color,cv2.FILLED)

black=(0,0,0)
white=(255,255,255)
green=(140,160,140)
orange=(0,155,255)
purple=(55,0,55)
cyan=(155,155,0)

def generate(i):
  global image
  # generate image
  image = np.zeros((height,width,3), np.uint8)
  image[:,:] = purple

  n=30
  xpstone=np.random.randint(width, size=n)
  ypstone=np.random.randint(height, size=n)
  rstone=np.random.randint(150, size=n)
  circles(xpstone,ypstone,rstone,green)

  n=200
  xpvoid=np.random.randint(width, size=n)
  ypvoid=np.random.randint(height, size=n)
  rvoid=np.random.randint(20, size=n)
  circles(xpvoid,ypvoid,rvoid,orange)
  # image generated, write
  cv2.imwrite("generated/Image/image%03d.jpg" % i, image)

  image = np.zeros((height,width,3), np.uint8)
  circles(xpvoid,ypvoid,rvoid,white) # draw voids
  circles(xpstone,ypstone,rstone,black) # erase stones
  # semantic for voids generated, write
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  cv2.imwrite("generated/Semantic/2_void/image%03d.png" % i, image)

  image = np.zeros((height,width,3), np.uint8)
  circles(xpstone,ypstone,rstone,white) # draw stones
  # semantic for stones generated, write
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  cv2.imwrite("generated/Semantic/1_stone/image%03d.png" % i, image)

for i in range(100):
  generate(i)

#cv2.imshow("Green Filled Circle on Image",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(np.random.randint(900, size=10).size)
