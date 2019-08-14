import glob
import os
import sys
import numpy as np
import carla
import random
import cv2
import pandas
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import glob

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
import keras
import tensorflow as tf


from models import *

model = unet_mid("weights/area-mid.hdf5")

def predict_model(image):
    #files = glob.glob(PATH_TEST + "*.jpg")
    #image = cv2.imread(files[index])
    

    image = cv2.resize(image, (256,128))
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    test = np.array([image])
  
    test = test.reshape(len(test),128,256,3)
    regions = model.predict(test, verbose=1)
    return regions[0]
        
    

def get_center(image):
    regions = predict_model(image)
    regions = cv2.cvtColor(regions, cv2.COLOR_BGR2RGB)
    regions[regions > 0.5 ] = 1.0
    regions[regions < 0.5 ] = 0
    kernel = np.ones((5,5),np.uint8)
    regions = cv2.morphologyEx(regions, cv2.MORPH_CLOSE, kernel)
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
    print(driveable_mask.shape)
    moment_mask = driveable_mask[0:600, 0:1280]
    M = cv2.moments(moment_mask)
    cX = 0
    cY = 0
    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
        # put text and highlight the center
        cv2.circle(display, (cX, cY), 5, (255, 255, 255), -1)
    except:
        pass

    dst = cv2.addWeighted(image, 1.0, display, 0.3, 0)

    merged = np.concatenate((dst, regions), axis=0)
    merged = cv2.resize(merged, (960,960))
    cv2.imshow("Result", merged)

    return (cX,cY)

client = carla.Client('localhost', 2000)
client.set_timeout(30.0) # seconds
client.reload_world()
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))

spawn_points = world.get_map().get_spawn_points()
car = world.spawn_actor(vehicle_bp, spawn_points[0])

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_bp.set_attribute('fov', '110')
# Set the time in seconds between sensor captures
camera_bp.set_attribute('sensor_tick', '0.1')
# Provide the position of the sensor relative to the vehicle.
camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
# Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=car)

car.apply_control(carla.VehicleControl(throttle=0.5))


lastImage = None
def proc(image):
    global lastImage
    lastImage = image

sensor.listen(lambda image: proc(image))
i = 0
while True:
    if lastImage is not None:
        data = np.array(lastImage.raw_data)
        data = data.reshape((lastImage.height, lastImage.width,4))
        data = cv2.cvtColor(data,cv2.COLOR_RGBA2RGB)
        data = data.reshape((lastImage.height, lastImage.width,3))
        center = get_center(data)

        x = center[0]
        centerLine = 640
        error = centerLine - x
        error = error/640
        error = -error
        p = 0.15
        steer = 0
        if abs(error) > 0.01:
            steer = error * p

        car.apply_control(carla.VehicleControl(steer=steer, throttle=0.5))
      
        plt.scatter(i,steer)
        plt.pause(0.05)
        print(steer)
        i += 1
        cv2.waitKey(1)
plt.show()