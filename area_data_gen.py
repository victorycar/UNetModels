#
# Include Setting
#

import cv2
import pandas
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.draw import line
from tqdm import tqdm

#
# Data Path Setting
#

DATA_PATH_BASE  = "data/"
INPUT_DATA_BASE = DATA_PATH_BASE + "input/"
INPUT_IMAGES    = INPUT_DATA_BASE + "images/"
INPUT_LABELS    = INPUT_DATA_BASE + "labels/"
TRAIN_IMAGES    = INPUT_IMAGES + "train/"
VAL_IMAGES      = INPUT_IMAGES + "val/"
TEST_IMAGES     = INPUT_IMAGES + "test/"
TRAIN_LABELS    = INPUT_LABELS + "bdd100k_labels_images_train.json"
VAL_LABELS      = INPUT_LABELS + "bdd100k_labels_images_val.json"


#
# Settings
#

VAL_LOAD = 10
TRAIN_LOAD = 10
DOWNSCALE = 4

#
# Load Labels into Memory
#

def load_label(path, to_load):
  count = 0
  with open(path) as json_file:  
    data = json.load(json_file)
    
    formatted_data = []
    
    for entry in tqdm(data):
      
      if count > to_load:
        continue
      
      image_name = entry['name']
      labels = entry['labels']
      
      regions = []
      
      for label in labels:
        cat = label['category']
       
        if cat not in 'drivable area':
          continue
        area_type = label['attributes']['areaType']
        
        if area_type not in 'direct':
          continue
        
        polygon = label['poly2d'][0]
        verts  = polygon['vertices']      
        regions.append(verts)
      
      formatted_data.append([image_name, regions])
      count += 1
     
    print("Loaded " + str(len(formatted_data)) + " entries")
    return formatted_data
      
val_labels = load_label(VAL_LABELS, VAL_LOAD)
#train_labels = load_label(TRAIN_LABELS, TRAIN_LOAD)



DOWNSCALE = 1

def label_to_image(label):
  region = label[1]
  image = np.zeros([int(720 / DOWNSCALE),int(1280 / DOWNSCALE),3])
  
  for points in region:
    points = np.array([points], dtype=np.int32)
    image = cv2.fillPoly(image,points, (1.0,1.0,1.0))
  image = cv2.resize(image, (254,126))
  return image



y2 = []
for label in tqdm(val_labels):
  y2.append(label_to_image(label))
print(val_labels[2])
plt.imshow(y2[2])
plt.show()