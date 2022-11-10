#!/usr/bin/env python3

# todo:
# [x] enter list of color images as path/file.jpg or path/file.png files
#     each file will be segmented to grayscale path/file_seg.png file
# [ ] cmdline option to specify model .pth file

import os, sys
from PIL import Image, ImageOps
import cv2
import torch
import torchvision.models.segmentation
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import time

# allow to load large images
Image.MAX_IMAGE_PIXELS = None

SavedModelsFolder = "generated/saved_models"
ListModels=os.listdir(SavedModelsFolder)
# use torch.load with map_location=torch.device('cpu')
#modelPath = "generated/saved_models/760.pth"  # Path to trained model
modelPath = os.path.join(SavedModelsFolder, ListModels[-1]) # latest model
print("trained model:", modelPath)

height=width=900 # should match training
#transformImg = tf.Compose([tf.Resize((height, width)),tf.ToTensor(),tf.Normalize((0.35, 0.35, 0.35),(0.18, 0.18, 0.18))])
transformImg = tf.Compose([tf.ToTensor(),tf.Normalize((0.35, 0.35, 0.35),(0.18, 0.18, 0.18))])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
#Net = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)  # Load net
#Net = torchvision.models.segmentation.deeplabv3_resnet101()
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath,map_location=torch.device(device))) # Load trained model
Net.eval() # Set to evaluation mode

def read_dpi(pilimg):
  if "dpi" in pilimg.info:
    return pilimg.info["dpi"]
  if "jfif_density" in pilimg.info:
    density = pilimg.info["jfif_density"]
    unit_m = (25.4e-3, 25.4e-3)
    if "jfif_unit" in pilimg.info:
      if pilimg.info["jfif_unit"] == 3:
        unit_m = (10.0e-3, 10.0e-3)
    return (density[0]*25.4e-3/unit_m[0], density[1]*25.4e-3/unit_m[1])
  return (2400,2400)

def semantic_segmentation(in_file, out_file):
  save_event = time.time()
  in_img = Image.open(in_file)
  out_img=Image.new(mode='L', size=in_img.size)   # mode='L' creates 8-bit grayscale image
  width_orgin, height_orgin = in_img.size
  print("info", in_img.info)
  in_dpi = read_dpi(in_img)
  print("dpi:", in_dpi)
  torch2pil = tf.ToPILImage()
  # recolor converts indexes 0,1,2 to 8-bit grayscales [0,1,2] -> [0,127,255]
  recolor = numpy.array([0,127,255], dtype=numpy.uint8)

  # for each tile from in_img, apply transform, semantic segmentation and paste to output image
  xclamp = width_orgin-width
  yclamp = height_orgin-height
  y = 0
  while y < height_orgin:
    if y > yclamp: y = yclamp
    #print()
    #print("y=",y,"x=",end="")
    x = 0
    while x < width_orgin:
      if x > xclamp: x = xclamp
      #print(x,end=" ")
      print("tile from origin", x,y)
      tile = in_img.crop((x,y,x+width,y+height))

      # semantic segmentation of the tile
      Img = transformImg(tile)
      mean, std = Img.mean([1,2]), Img.std([1,2])
      # print statistics, if most images deviate too much,
      # change tf.Normalize. They must be same here in Infer.py and in Train.py
      print("mean (ideal = 0,0,0) : ", mean);
      print("std  (ideal = 1,1,1) : ", std);
      Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
      with torch.no_grad():
        Prd = Net(Img)['out']  # Run net
      # Prd = tf.Resize((height_orgin, width_orgin))(Prd[0]) # Resize to origninal size
      Prd = Prd[0] # don't resize, just take out the basic result
      seg = torch.argmax(Prd, 0).cpu().detach().numpy().astype(numpy.uint8) # Get prediction classes
      # color convert and paste result tile to same origin in out_img
      out_img.paste(torch2pil(recolor[seg]), (x,y)) # (x,y) is origin coordinate to paste at
      # save partial result every minute
      if time.time() > save_event:
        save_event = time.time() + 60
        out_img.save(out_file, dpi=in_dpi)
        print("saved", out_file)
      x += width
    y += height

  # save final image to file
  out_img.save(out_file, dpi=in_dpi)

# display 2 images side-by-side
def display2img(path1, path2):
  color = mpimg.imread(path1)
  result = mpimg.imread(path2)
  f, axarr = plt.subplots(1, 2) # Y x X images on the plot
  f.set_size_inches(16, 9)      # Y x X inch size on monitor
  axarr[0].imshow(color)        # show first image
  axarr[1].imshow(result)       # show second image
  # enable this to remove pixel size ticks
  # axarr[0].axis('off'), axarr[1].axis('off')
  plt.show()

input_file_list = sys.argv[1:]
for imagePath in input_file_list:
  print("input  :", imagePath)
  ip_base, ip_ext = os.path.splitext(imagePath)
  resultPath = ip_base + "_seg.png" # file name of result segmented image
  semantic_segmentation(imagePath, resultPath)
  print("output :", resultPath)

#  1 input file  -> display result
# >1 input files -> don't display (only write to files)
#if len(input_file_list) == 1:
#  display2img(imagePath, resultPath)
