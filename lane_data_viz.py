import cv2
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
PATH_BASE = "data/lane/"


files = glob.glob(PATH_BASE + "/train/*.npy")

for file in files:

    entry = np.load(file, allow_pickle=True)
    source = entry[0]
    lanes = entry[1]
   
    result = source + lanes;

    print("Source Max: " + str(np.max(source)))
    print("Source Min: " + str(np.min(source)))
    print("lanes Max: " + str(np.max(lanes)))
    print("lanes Min: " + str(np.min(lanes)))
    
    cv2.imshow("lanes",lanes)
    cv2.imshow("result",result)
    
    cv2.waitKey(100)
