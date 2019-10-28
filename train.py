import cv2
import pandas
import numpy as np
import keras
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as keras
from skimage.draw import line
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
from models import *
PATH_TEST = "data/test/"
PATH_BASE = "data/lane/"
VAL_COUNT = 4000 
TRAIN_COUNT = 16000 
BATCH = 8
IMAGE_SIZE_W = 256
IMAGE_SIZE_H = 128
NAME = "lane-mid"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




def lane_generatoe(files, batch_size=4):

    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=files,
                                       size=batch_size)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            entry = np.load(input_path)
           # input = entry[0]
          

            input = cv2.resize(entry[0], (IMAGE_SIZE_W,IMAGE_SIZE_H))
            output = cv2.resize(entry[1], (254,126))

            batch_input += [input]
            batch_output += [output]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)

print("LOADING MODEL")

model = unet_mid()


try:
    model = multi_gpu_model(model)
except:
    print("~NOT USING MUTLI GPU")
    pass
def predict_model(index=0):
    files = glob.glob(PATH_TEST + "*.jpg")
    image = cv2.imread(files[index])
    
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (IMAGE_SIZE_W,IMAGE_SIZE_H))
    test = np.array([image])
    test = test.reshape(len(test),IMAGE_SIZE_H,IMAGE_SIZE_W,3)
    lanes = model.predict(test, verbose=1)

    y = lanes[0]
    x = cv2.resize(test[0], (IMAGE_SIZE_W - 2,IMAGE_SIZE_H - 2))
    x = np.array([x])
    x = x.reshape(len(x),IMAGE_SIZE_H - 2,IMAGE_SIZE_W - 2,3)
    #y *= 255
    #y = cv2.normalize(y.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)
    combined = x[0] + (y)
    merged = np.concatenate((combined, y), axis=0)
    return merged


class Predict(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
      lanes = predict_model()
      plt.imshow(lanes)
      plt.savefig('results/'+NAME+'_' + str(epoch)+'.png')
      return


model_checkpoint_best = ModelCheckpoint('weights/'+NAME+'.hdf5', monitor='loss',verbose=1,save_best_only=True)

predict_cb = Predict()

from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log/'+NAME+"/",
                        
                         batch_size=BATCH,
                         update_freq="batch"
                         )


trainFiles = glob.glob(PATH_BASE + "/train/*.npy")
valFiles = glob.glob(PATH_BASE + "/val/*.npy")

trainGen = lane_generatoe(trainFiles,BATCH)
valGen = lane_generatoe(valFiles, BATCH)

trainFiles = trainFiles[:TRAIN_COUNT]
valFiles = valFiles[:VAL_COUNT]

H = model.fit_generator(
	trainGen,
	steps_per_epoch=TRAIN_COUNT // BATCH,
	validation_data=valGen,
	validation_steps=VAL_COUNT // BATCH,
    epochs=200,
    verbose=1, 
    callbacks=[ model_checkpoint_best, predict_cb])
