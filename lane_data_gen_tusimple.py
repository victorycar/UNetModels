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

BASE_PATH = "D:/Datasets/TuSimple/"
LABEL_FILES = ["label_data_0601.json",
               "label_data_0531.json", "label_data_0601.json"]
POINT_COUNT = 50
VAL_COUNT = 300
TRAIN_COUNT = 800


def load_label_file(path):
    with open(path) as contents:
        lines = contents.readlines()
        lanes = []
        for line in lines:
            data = json.loads(line)
            lanes.append(data)
        return lanes


def split_labels(data, nTrain, nVal):
    offset = 0
    train = []
    val = []
    for i in tqdm(range(nTrain)):
        train.append(data[i])

    for i in tqdm(range(nVal)):
        val.append(data[nTrain + i])

    return (train, val)

DOWNSCALE = 4
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
        rr = np.clip(rr, 0, int(720 / DOWNSCALE) - 2)
        cc = np.clip(cc, 0, int(1280 / DOWNSCALE) -2)
        image[rr     ,cc, :] = 1.0
        image[rr     ,cc - 1, :] = 1.0
        image[rr     ,cc + 1, :] = 1.0
        image[rr - 1 ,cc , :] = 1.0
        image[rr + 1 ,cc , :] = 1.0
        image[rr - 1 ,cc - 1, :] = 1.0
        image[rr + 1 ,cc + 1, :] = 1.0
    return image


def parse_labels(data):
    filename = BASE_PATH + data["raw_file"]
    
    image = cv2.imread(filename)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (254, 126))
    image = cv2.normalize(image.astype('float'), None,
                          0.0, 1.0, cv2.NORM_MINMAX)
    
    lanes = data["lanes"]
    h_samples = data["h_samples"]

    lane_1 = []
    lane_2 = []
    lane_3 = []
    lane_4 = []

    for i in range(POINT_COUNT):
        x1 = lanes[0][i]
        x2 = lanes[1][i]
        try:
            x3 = lanes[2][i]
        except:
            x3 = -2
            
        try:
             x4 = lanes[3][i]
        except:
             x4 = -2

        y = h_samples[i]
        
        lane_1.append((x1 , y ))
        lane_2.append((x2, y))
        lane_3.append((x3 , y))
        lane_4.append((x4, y))


    lane_img = np.zeros([int(720 / DOWNSCALE),int(1280 / DOWNSCALE),3])

    lane_img = draw(lane_img,lane_1, (1.0,0,0))
    lane_img = draw(lane_img,lane_2, (0,1.0,0))
    lane_img = draw(lane_img,lane_3, (0,1.0,0))
    lane_img = draw(lane_img,lane_4, (0,0,1.0))

    lane_img = cv2.resize(lane_img, (254, 126))

    return [image, lane_img]


print("--- LOADING  LABELES --- ")
labels = []
for label_file in tqdm(LABEL_FILES):
    labels = labels + load_label_file(BASE_PATH + label_file)
print("LABEL COUNT: " + str(len(labels)))
print("--- SPLITTING  LABELES --- ")
(train_labels, val_labels) = split_labels(labels, TRAIN_COUNT, VAL_COUNT)

print("--- PARSING  LABELES --- ")
index = 0
for label in tqdm(train_labels):
    data = parse_labels(label)
    np.save("data/lane/train/tusimplelane-"+str(index)+".npy", data)
    index += 1


index = 0
for label in tqdm(val_labels):
    data = parse_labels(label)
    np.save("data/lane/val/tusimplelane-"+str(index)+".npy", data)
    index += 1
