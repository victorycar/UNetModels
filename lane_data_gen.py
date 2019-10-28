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
import matplotlib.patches as mpatches
from matplotlib.path import Path
from skimage.draw import line, bezier_curve
from tqdm import tqdm

#
# Data Path Setting
#

DATA_PATH_BASE  = "G:/Datasets/Berkly/"
INPUT_DATA_BASE = DATA_PATH_BASE + "input/"
INPUT_IMAGES    = INPUT_DATA_BASE + "images/"
INPUT_LABELS    = INPUT_DATA_BASE + "labels/"
TRAIN_IMAGES    = INPUT_IMAGES + "train/"
VAL_IMAGES      = INPUT_IMAGES + "val/"

TRAIN_LABELS    = INPUT_LABELS + "bdd100k_labels_images_train.json"
VAL_LABELS      = INPUT_LABELS + "bdd100k_labels_images_val.json"


#
# Data State
#

STATE_OFFESET = 0
STATE_INDEX   = 0
#
# Settings
#

VAL_LOAD = 3000
TRAIN_LOAD = 10000
DOWNSCALE = 4
VAL_START = 40
#
# Load Labels into Memory
#



def load_label_file(path):
  with open(path) as json_file:  
    data = json.load(json_file)
    return data

print ("--- LOADING  VALIDATION LABELES --- ")
val_file   = load_label_file(VAL_LABELS)


#
# Parse Labels
#

def parse_label(entry):
  image_name = entry['name']
  labels = entry['labels']
  
  lanes = []
  formatted_data = []
  for label in labels:
    cat = label['category']
    
    if cat not in 'lane':
      continue
    if label['attributes']['laneDirection'] not in "parallel": 
      #print(label['attributes']['laneDirection'] )
      continue

    polygon = label['poly2d'][0]
    verts  = polygon['vertices']   
    types = polygon['types']

    codes = []
    moves = {'L': Path.LINETO,'C': Path.CURVE4}
    codes = [moves[t] for t in types]
    
    codes[0] = Path.MOVETO
    
    lanes.append([codes,verts])
  
  formatted_data.append([image_name, lanes])
  return formatted_data

#
# Image Generation
#
def label_to_image(label):
  lines = label[1]
  image = np.zeros([int(720 / DOWNSCALE),int(1280 / DOWNSCALE),3])
  
  for cur in lines:
    verts = cur[1]
    control = cur[0]
    offset = 1

    if 4 in control:
      path = Path(verts, control)
      patch = mpatches.PathPatch(path, snap=True, lw=10, edgecolor=(1.0,1.0,1.0))
      verts = patch.get_verts()
      offset = 2
    
   

    for i in range(len(verts)- offset ):
      vert = verts[i]

      next_vert = verts[i + 1];
      y1 = int(vert[0] / DOWNSCALE)
      x1 = int(vert[1] / DOWNSCALE)
      y2 = int(next_vert[0] / DOWNSCALE)
      x2 = int(next_vert[1] / DOWNSCALE)
      rr, cc = line(x1,y1,x2,y2)
      rr = np.clip(rr, 0, int(720 / DOWNSCALE) - 2)
      cc = np.clip(cc, 0, int(1280 / DOWNSCALE) -2)
      image[rr     ,cc, :] = 1.0
      image[rr     ,cc - 1, :] = 1.0
      image[rr     ,cc + 1, :] = 1.0
      image[rr - 1 ,cc , :] = 1.0
      image[rr + 1 ,cc , :] = 1.0
      image[rr - 1 ,cc - 1, :] = 1.0
      image[rr + 1 ,cc + 1, :] = 1.0

  image = cv2.resize(image, (254, 126))
  return image

def get_source(path):
  
  image = cv2.imread(path)
  image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
  image = cv2.resize(image, (254,126))
  return image

print ("--- GENERATING VALIDATION DATA ---")

for i in tqdm(range(VAL_LOAD)):
  entry = val_file[i]
  labels = parse_label(entry)
  for label in labels:
    data_entry = [get_source(VAL_IMAGES + label[0]), label_to_image(label)]
    np.save("data/lane/val/bddlane-"+str(STATE_INDEX)+".npy", data_entry);
    STATE_INDEX += 1

print ("--- LOADING TRAINING LABELES --- ")
train_file   = load_label_file(TRAIN_LABELS)

STATE_INDEX = 0
print ("--- GENERATING TRAINING DATA ---")
for i in tqdm(range(TRAIN_LOAD)):
  entry = train_file[i]
  labels = parse_label(entry)
  for label in labels:
    data_entry = [get_source(TRAIN_IMAGES + label[0]), label_to_image(label)]
    np.save("data/lane/train/bddlane-"+str(STATE_INDEX)+".npy", data_entry);
    STATE_INDEX += 1
    