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


lane_model = unet_small("weights/lane-small.hdf5")
area_model = unet_mid("weights/area-mid.hdf5")

def predict(model,image):
    #files = glob.glob(PATH_TEST + "*.jpg")
    #image = cv2.imread(files[index])
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (256,128))
    test = np.array([image])
    test = test.reshape(len(test),128,256,3)
    result = model.predict(test, verbose=0)
    return result[0]

import numpy as np
import cv2
out = cv2.VideoWriter('output.mp4', -1, 20, (960,960))
raw = cv2.VideoWriter('raw.mp4', -1, 20, (1280,720))
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    raw.write(frame)

    regions = predict(area_model,frame)
    regions = cv2.cvtColor(regions, cv2.COLOR_BGR2RGB)
    regions[regions > 0.5 ] = 1.0
    regions[regions < 0.5 ] = 0
    kernel = np.ones((5,5),np.uint8)
    #regions = cv2.morphologyEx(regions, cv2.MORPH_CLOSE, kernel)
    regions = cv2.resize(regions, (1280,720))
    regions *= 255
    regions = cv2.normalize(regions.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)

    driveable_mask = np.copy(regions);
    driveable_mask[:, :, 0] = 0
    driveable_mask[:, :, 2] = 0
    driveable_mask = cv2.cvtColor(driveable_mask, cv2.COLOR_RGB2GRAY)
    
    display = np.zeros((720,1280,3), np.uint8)

    (_, contours, _) = cv2.findContours(driveable_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(display, [c], -1, [0, 200, 0, 20], -1) # debug
    M = cv2.moments(driveable_mask)
 
    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cX < 520:
            cv2.putText(display,  "Turn Left: " + str(abs(640 - cX) ), (100,200),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        elif  cX > 760:
            cv2.putText(display,  "Turn Right: " + str(abs(640 - cX) ), (100,200),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            pass
        else:
            cv2.putText(display,  "Straight: " + str(abs(640 - cX) ), (100,200),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            pass
        # put text and highlight the center
        cv2.circle(display, (cX, cY), 5, (255, 255, 255), -1)
    except :
        pass
    
    alt_mask = np.copy(regions);
    alt_mask[:, :, 0] = 0
    alt_mask[:, :, 1] = 0
    alt_mask = cv2.cvtColor(alt_mask, cv2.COLOR_RGB2GRAY)

    (_, contours, _) = cv2.findContours(alt_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:  
        cv2.drawContours(display, [c], -1, [0, 0, 255, 0.5], -1) # debug
    
    
    dst = cv2.addWeighted(frame, 1.0, display, 0.3, 0)
    merged = np.concatenate((dst, regions), axis=0)
    merged = cv2.resize(merged, (960,960))
    out.write(merged)
    cv2.imshow("Result", merged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
out.release()
raw.release()
cap.release()
cv2.destroyAllWindows()