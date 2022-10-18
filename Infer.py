#!/usr/bin/env python3

# todo:
# [ ] enter list of color images as path/file.jpg or path/file.png files
#     each file will be segmented to grayscale path/file_seg.png file
# [ ] cmdline option to specify model .pth file

import os, sys
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#SavedModelsFolder = "generated/saved_models"
SavedModelsFolder = "mikroskop/saved_models"
ListModels=os.listdir(SavedModelsFolder)
# use torch.load with map_location=torch.device('cpu')
#modelPath = "generated/saved_models/760.pth"  # Path to trained model
modelPath = os.path.join(SavedModelsFolder, ListModels[-1]) # latest model
print("trained model:", modelPath)

height=width=900 # should match training
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),tf.Normalize((0.35, 0.35, 0.35),(0.18, 0.18, 0.18))])  # tf.Resize((300,600)),tf.RandomRotation(145)])#

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath,map_location=torch.device(device))) # Load trained model
Net.eval() # Set to evaluation mode

def semantic_segmentation(imagePath, resultPath):
  Img = cv2.imread(imagePath) # load test image
  height_orgin, width_orgin, d = Img.shape # Get image original size 
  #plt.imshow(Img[:,:,::-1])  # Show image
  #plt.show()
  Img = transformImg(Img)  # Transform to pytorch
  mean, std = Img.mean([1,2]), Img.std([1,2])
  # print statistics, if most images deviates too much,
  # change tf.Normalize. They must be same here in Infer.py
  # and in Train.py
  print("mean (ideal = 0,0,0) : ", mean);
  print("std  (ideal = 1,1,1) : ", std);

  Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
  with torch.no_grad():
    Prd = Net(Img)['out']  # Run net
  Prd = tf.Resize((height_orgin, width_orgin))(Prd[0]) # Resize to origninal size
  seg = torch.argmax(Prd, 0).cpu().detach().numpy()  # Get prediction classes
  #plt.imshow(seg)  # display image
  #plt.show()
  cv2.imwrite(resultPath, seg*127) # HACK: writes grayscale image with 3 "colors" [0,1,2] -> [0,127,254]

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
if len(input_file_list) == 1:
  display2img(imagePath, resultPath)
