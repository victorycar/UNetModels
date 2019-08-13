import cv2
import pandas
import numpy as np
import keras
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as keras
from skimage.draw import line
from tqdm import tqdm_notebook as tqdm
from keras_tqdm import TQDMNotebookCallback

from models import *
PATH_TEST = "data/input/images/test/"
PATH_BASE = "data/output/lane/"

model = unet_small("weights/lane-small.hdf5")

def predict_model(image):
    #files = glob.glob(PATH_TEST + "*.jpg")
    #image = cv2.imread(files[index])
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (256,128))
    test = np.array([image])
    test = test.reshape(len(test),128,256,3)
    lanes = model.predict(test, verbose=0)
    return lanes[0]

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    lanes = predict_model(frame)
    lanes = cv2.resize(lanes, (1280,720))
    lanes = np.clip(lanes, a_min = 0.3, a_max =1.0) 
    lanes *= 255
    lanes = cv2.normalize(lanes.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)
    lanes = cv2.cvtColor(lanes, cv2.COLOR_RGB2GRAY)
    
    (_, contours, _) = cv2.findContours(lanes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # biggest area
   
    for target in contours:
        
        # cv2.drawContours(frame, [target], -1, [0, 200, 0], -1) # debug
        # just example of fitting
        x = target[:, :, 0].flatten()
        y = target[:, :, 1].flatten()
        poly = np.poly1d(np.polyfit(x, y, 5))
        for _x in range(min(x), max(x), 20): # too lazy for line/curve :)
            cv2.circle(frame, (_x, int(poly(_x))), 3, [0, 255, 0])

    
    # Display the resulting frame
   

    numpy_horizontal_concat = np.concatenate((frame, lanes), axis=1)
    cv2.imshow("Result", numpy_horizontal_concat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()