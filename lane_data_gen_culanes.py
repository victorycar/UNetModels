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

BASE_PATH = "G:/Datasets/CULane/"
POINT_COUNT = 50
VAL_COUNT = 6000
TRAIN_COUNT = 30000


IMAGES = glob.glob(BASE_PATH+"/**/**/*.jpg")

IMAGE_SIZE = (720,1280)

test_image = cv2.imread(IMAGES[0])
IMAGE_SIZE = test_image.shape
print(IMAGE_SIZE)

def load_label_file(path):
    with open(path) as contents:
        lines = contents.readlines()
        lanes = []
        for line in lines:
            line = line.replace(" \n", "")
            coords_str = line.split(" ")
            
            coords = []
            true_idx = 0
            for i in range(int(len(coords_str)/2)):
                coords.append((float(coords_str[true_idx]),float(coords_str[true_idx + 1])))
                true_idx += 2

            lanes.append(coords)
       
        return lanes

def split(data, nTrain, nVal):
    offset = 0
    train = []
    val = []
    for i in tqdm(range(nTrain)):
        train.append(data[i])

    for i in tqdm(range(nVal)):
        val.append(data[nTrain + i])

    return (train, val)

def draw(image, points, color):
    for i in range(len(points)- 1 ):

        
        vert = points[i]


        next_vert = points[i + 1];
        if vert[0] < 1:
            continue

        if next_vert[0] < 1:
            continue

        y1 = int(vert[0] / DOWNSCALE)
        x1 = int(vert[1] / DOWNSCALE)
        y2 = int(next_vert[0] / DOWNSCALE)
        x2 = int(next_vert[1] / DOWNSCALE)
        rr, cc = line(x1,y1,x2,y2)
        rr = np.clip(rr, 0, int(IMAGE_SIZE[0] / DOWNSCALE) - 2)
        cc = np.clip(cc, 0, int(IMAGE_SIZE[1] / DOWNSCALE) -2)
        image[rr     ,cc, :] = 1.0
        image[rr     ,cc - 1, :] = 1.0
        image[rr     ,cc + 1, :] = 1.0
        image[rr - 1 ,cc , :] = 1.0
        image[rr + 1 ,cc , :] = 1.0
        image[rr - 1 ,cc - 1, :] = 1.0
        image[rr + 1 ,cc + 1, :] = 1.0
    return image


DOWNSCALE = 2
def parse(image_path):
    
    label_path = image_path.replace(".jpg", ".lines.txt");
    label = load_label_file(label_path)
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 128))
    image = cv2.normalize(image.astype('float'), None,
                          0.0, 1.0, cv2.NORM_MINMAX)

    lane_img = np.zeros([int(IMAGE_SIZE[0] / DOWNSCALE),int(IMAGE_SIZE[1] / DOWNSCALE),3])
    
    for lane in label:
        lane_img = draw(lane_img,lane, (1.0,0,0))
    lane_img = cv2.resize(lane_img, (256, 128))

    return [image, lane_img]


print("LABEL COUNT: " + str(len(IMAGES)))

print("--- SPLITTING  LABELES --- ")
train_data, val_data = split(IMAGES, TRAIN_COUNT, VAL_COUNT);

print("--- PARSING  LABELES --- ")
index = 0
for label in tqdm(train_data):
    data = parse(label)
    np.save("data/lane/train/culane-"+str(index)+".npy", data)
    index += 1


index = 0
for label in tqdm(val_data):
    data = parse(label)
    np.save("data/lane/val/culane-"+str(index)+".npy", data)
    index += 1
