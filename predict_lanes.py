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
#model = unet_small("weights/lane-small.hdf5")
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

cap = cv2.VideoCapture("drive-1.mp4")
#out = cv2.VideoWriter('output.mp4', -1, 30, (960,960))
#raw = cv2.VideoWriter('raw.mp4', -1, 20, (1280,720))
frame_n = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        continue;

    frame = cv2.resize(frame, (1280,720))
    lanes = predict_model(frame)
    
    
   
    lanes[lanes > 0.5] = 1.0
    lanes[lanes < 0.5] = 0
    
    lanes = cv2.resize(lanes, (1280,720))
   
    
    lanes *= 255
    lanes = cv2.normalize(lanes.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)
    lanes = cv2.cvtColor(lanes, cv2.COLOR_RGB2GRAY)
  
    (_, contours, _) = cv2.findContours(lanes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # biggest area
   
    for target in contours:
        if cv2.contourArea(target) < 2500:
            target = []
            #cv2.drawContours(frame, [target], -1, [0, 200, 0], -1) # debug
        else:
            x = target[:, :, 0].flatten()
            y = target[:, :, 1].flatten()
            poly = np.poly1d(np.polyfit(x, y, 5))
            for _x in range(min(x), max(x), 5): # too lazy for line/curve :)
                cv2.circle(frame, (_x, int(poly(_x))), 4, [255, 0, 0])
      
    
    # Display the resulting frame
    lanes = cv2.cvtColor(lanes, cv2.COLOR_GRAY2RGB)
   
    merged = np.concatenate((frame, lanes), axis=0)
    merged = cv2.resize(merged, (960,960))
    cv2.imshow("Result", merged)
    #out.write(merged)
    #raw.write(frame)
   # frame_n += 1

   #if frame_n > (24 * 60):
   # out.release()
    #out = cv2.VideoWriter('output.mp4', -1, 30, (960,960))
   # frame_n = 0;

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
raw.release()
cv2.destroyAllWindows()