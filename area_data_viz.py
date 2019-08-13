import cv2
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
PATH_BASE = "data/output/area/"


files = glob.glob(PATH_BASE + "/val/*.npy")

for file in files:

    entry = np.load(file)
    x = entry[0]
    y = entry[1]

    overlayed = x + y 
  
    plt.imshow(overlayed)
    plt.show();
