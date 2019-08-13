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


model = unet_mid("weights/area-mid.hdf5")
#model = unet_small("weights/lane-small.hdf5")
def predict_model(image):
    #files = glob.glob(PATH_TEST + "*.jpg")
    #image = cv2.imread(files[index])
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (256,128))
    test = np.array([image])
    test = test.reshape(len(test),128,256,3)
    regions = model.predict(test, verbose=0)
    return regions[0]

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
    regions = predict_model(frame)
    regions = cv2.cvtColor(regions, cv2.COLOR_BGR2RGB)
    kernel = np.ones((2,2),np.uint8)
    regions = cv2.morphologyEx(regions, cv2.MORPH_CLOSE, kernel)
    regions = cv2.resize(regions, (1280,720))
    regions *= 255
    regions = cv2.normalize(regions.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)

    driveable_mask = np.copy(regions);
    driveable_mask[:, :, 0] = 0
    driveable_mask[:, :, 2] = 0
    driveable_mask[driveable_mask > (0.5 * 255)] = 255
    driveable_mask[driveable_mask < (0.5 * 255)] = 0
    driveable_mask = cv2.cvtColor(driveable_mask, cv2.COLOR_RGB2GRAY)
    
    (_, contours, _) = cv2.findContours(driveable_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) is not 0:    
        c = max(contours, key = cv2.contourArea)
        cv2.drawContours(frame, [c], -1, [0, 200, 0, 0.5], -1) # debug
    
    merged = np.concatenate((frame, regions), axis=0)
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
cv2.destroyAllWindows()