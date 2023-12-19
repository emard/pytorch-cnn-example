#!/usr/bin/env python3
import os, re
import numpy as np
import cv2
import torch
import time
import torchvision.models.segmentation
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Learning_Rate=5e-6 # too large: oscillates, too small: slow convergence
width=height=900 # image width and height, minimum 224 pixels
batchSize=4
save_every=200

TrainFolder="train"

SavedModelFolder=os.path.join(TrainFolder,"saved_models")
ListImages=[]
def updateListImages():
  global ListImages
  ListImages=os.listdir(os.path.join(TrainFolder, "Image")) # Create list of images
updateListImages()
print(len(ListImages), "images")
# display 2 images side-by-side
def display2img(color, result):
  #color = mpimg.imread(path1)
  #result = mpimg.imread(path2)
  f, axarr = plt.subplots(1, 2) # Y*X images on the plot
  f.set_size_inches(16, 9)      # X*Y inch size on monitor
  axarr[0].imshow(color.permute(1, 2, 0))  # show first image
  axarr[1].imshow(result.permute(1, 2, 0)) # show second image
  # enable this to remove pixel size ticks
  # axarr[0].axis('off'), axarr[1].axis('off')
  plt.show()
#----------------------------------------------Transform image-------------------------------------------------------------------
# 4 flip rotations to diversify learning
# tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) -> normalize with mean and stdev for each channel
# tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# with random crop_x crop_y, tf.Resize is not needed
# if sizes of all images are larger or equal than width,height
transformImg=(tf.Compose([tf.ToPILImage(),                                        tf.Resize((height,width)),tf.ToTensor(),tf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),tf.Normalize((0.35, 0.35, 0.35), (0.18, 0.18, 0.18))]),
              tf.Compose([tf.ToPILImage(),tf.functional.hflip,                    tf.Resize((height,width)),tf.ToTensor(),tf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),tf.Normalize((0.35, 0.35, 0.35), (0.18, 0.18, 0.18))]),
              tf.Compose([tf.ToPILImage(),tf.functional.vflip,                    tf.Resize((height,width)),tf.ToTensor(),tf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),tf.Normalize((0.35, 0.35, 0.35), (0.18, 0.18, 0.18))]),
              tf.Compose([tf.ToPILImage(),tf.functional.hflip,tf.functional.vflip,tf.Resize((height,width)),tf.ToTensor(),tf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),tf.Normalize((0.35, 0.35, 0.35), (0.18, 0.18, 0.18))]),
              )
transformAnn=(tf.Compose([tf.ToPILImage(),                                        tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()]),
              tf.Compose([tf.ToPILImage(),tf.functional.hflip,                    tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()]),
              tf.Compose([tf.ToPILImage(),tf.functional.vflip,                    tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()]),
              tf.Compose([tf.ToPILImage(),tf.functional.hflip,tf.functional.vflip,tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()]),
              )

#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(): # load random image and the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image
    #print(ListImages[idx])
    Img=cv2.imread(os.path.join(TrainFolder, "Image", ListImages[idx]))[:,:,0:3]
    image_height, image_width, image_channels = Img.shape
    # randomize wanted crop size between width/2 and width*2
    min_random=width//2
    max_random=width*2
    # clamp max_random to available image size
    if max_random > image_width:
      max_random = image_width
    if max_random > image_height:
      max_random = image_height
    random_width = random_height = np.random.randint(min_random,max_random)
    #print(image_width, image_height)
    crop_x = crop_y = 0
    if image_width > random_width:
      crop_x = np.random.randint(0, image_width  - random_width)
    if image_height > random_height:
      crop_y = np.random.randint(0, image_height - random_height)
    #print(crop_x, crop_y, )
    if crop_x > 0 or crop_y > 0:
      Img    = Img[crop_y:crop_y+random_height, crop_x:crop_x+random_width]
    type1  = cv2.imread(os.path.join(TrainFolder, "Semantic/1", ListImages[idx].replace("jpg","png")),cv2.IMREAD_GRAYSCALE)
    if crop_x > 0 or crop_y > 0:
      type1  = type1[crop_y:crop_y+random_height, crop_x:crop_x+random_width]
    type2  = cv2.imread(os.path.join(TrainFolder, "Semantic/2", ListImages[idx].replace("jpg","png")),cv2.IMREAD_GRAYSCALE)
    if crop_x > 0 or crop_y > 0:
      type2  = type2[crop_y:crop_y+random_height, crop_x:crop_x+random_width]
    AnnMap = np.zeros(Img.shape[0:2],np.float32)
    if type2 is not None: AnnMap[ type2 > 60 ] = 2 # "void"
    if type1 is not None: AnnMap[ type1 > 60 ] = 1 # "stone", overwrites void
    #randflip = 0
    randflip = np.random.randint(0,4)
    Img=transformImg[randflip](Img)
    AnnMap=transformAnn[randflip](AnnMap)
    #display2img(Img, Img)
    return Img,AnnMap
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage()
    return images, ann
#------------- save .pth file (trained model) -------------------
def save_trained_model(itr):
  global saved_filename
  # delete old saved
  if saved_filename != "":
    file_to_remove = os.path.join(SavedModelFolder, saved_filename)
    if os.path.exists(file_to_remove):
      os.remove(file_to_remove)
  saved_filename = str(itr) + ".pth"
  torch.save(Net.state_dict(), os.path.join(SavedModelFolder, saved_filename))
  print("Saved Model", saved_filename)

#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#Net = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()
#Net = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT) # Load net
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights=None) # Load net
#Net = torchvision.models.segmentation.deeplabv3_resnet101()
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
#---------------- Load save state --------------------------------------------------------------
ListSaved=os.listdir(SavedModelFolder) # Create list of saved models
print(ListSaved)
saved_filename = ""
if ListSaved:
  saved_filename = ListSaved[-1]
itr = 0
if saved_filename != "":
   Net.load_state_dict(torch.load(os.path.join(SavedModelFolder, saved_filename), map_location=device))
   print("Loaded Model", saved_filename)
   itr = int(re.sub('[^\d]', '', saved_filename))
   print("Resume iteration", itr)
#----------------Train--------------------------------------------------------------------------
save_event = time.time()
while itr*batchSize < 40000: # Training loop
   images,ann=LoadBatch() # Load taining batch
   images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
   ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
   Pred=Net(images)['out'] # make prediction
   Net.zero_grad()
   criterion = torch.nn.CrossEntropyLoss() # Set loss function
   Loss=criterion(Pred,ann.long()) # Calculate cross entropy loss
   Loss.backward() # Backpropogate loss
   optimizer.step() # Apply gradient descent change to weight
   seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get  prediction classes
   print("Iteration=%5d" % itr, " Loss=%3.0f%%" % (Loss.data.cpu().numpy() * 100))
   if time.time() > save_event:
      save_event = time.time() + 60
      save_trained_model(itr)
      updateListImages()
   itr += 1
save_trained_model(itr)
