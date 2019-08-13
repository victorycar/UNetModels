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
# Data State
#

STATE_OFFESET = 0
STATE_INDEX   = 0
#
# Settings
#

VAL_LOAD = 4000
TRAIN_LOAD = 20000
DOWNSCALE = 1

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
  
  driveable = []
  alt = []
  formatted_data = []
  for label in labels:
    cat = label['category']
    
    if cat not in 'drivable area':
      continue
    area_type = label['attributes']['areaType']
    #print(area_type)
    
    if area_type in 'direct':
      polygon = label['poly2d'][0]
      verts  = polygon['vertices']
      types = polygon['types']
      closed = polygon["closed"]
      codes = []

      moves = {'L': Path.LINETO,'C': Path.CURVE4}
      codes = [moves[t] for t in types]

      codes[0] = Path.MOVETO
      if closed:
        verts.append(verts[0])
        codes.append(Path.CLOSEPOLY)
      driveable.append([verts, codes])

    else:
      polygon = label['poly2d'][0]
      verts  = polygon['vertices']
      types = polygon['types']
      closed = polygon["closed"]
      codes = []

      moves = {'L': Path.LINETO,'C': Path.CURVE4}
      codes = [moves[t] for t in types]
      codes[0] = Path.MOVETO

      if closed:
        verts.append(verts[0])
        codes.append(Path.CLOSEPOLY)

      alt.append([verts, codes])
  
  
  formatted_data.append([image_name, driveable, alt])
  return formatted_data

#
# Image Generation
#
def label_to_image(label):
  driveable = label[1]
  alt = label[2]
  image = np.zeros([int(720 / DOWNSCALE),int(1280 / DOWNSCALE),3])

  for cur in driveable:
    verts = cur[0]
    control = cur[1]
    path = Path(verts, control)
    patch = mpatches.PathPatch(path)
    points = np.array([patch.get_verts()], dtype=np.int32)
    image = cv2.fillPoly(image,points, (0,1.0,0))
  
  for cur in alt:
    verts = cur[0]
    control = cur[1]
    path = Path(verts, control)
    patch = mpatches.PathPatch(path)
    points = np.array([patch.get_verts()], dtype=np.int32)
    image = cv2.fillPoly(image,points, (1.0,0,0))
  
  image = cv2.resize(image, (254,126))
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
    np.save("data/output/area/val/lane-"+str(STATE_INDEX)+".npy", data_entry);
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
    np.save("data/output/area/train/lane-"+str(STATE_INDEX)+".npy", data_entry);
    STATE_INDEX += 1
