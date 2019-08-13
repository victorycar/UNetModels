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
from keras_tqdm import TQDMNotebookCallback
from datetime import datetime
from models import *
PATH_TEST = "data/input/images/test/"
PATH_BASE = "data/output/area/"
VAL_COUNT = 4000
TRAIN_COUNT = 20000
BATCH = 16


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
            input = entry[0]
            output = entry[1]

            input = cv2.resize(input, (256,128))

            batch_input += [input]
            batch_output += [output]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)

print("LOADING MODEL")

model = unet_mid('weights/area-mid.hdf5')
try:
    model = multi_gpu_model(model)
except:
    print("~NOT USING MUTLI GPU")
    pass
def predict_model(index=0):
    files = glob.glob(PATH_TEST + "*.jpg")
    image = cv2.imread(files[index])
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (256,128))
    test = np.array([image])
    test = test.reshape(len(test),128,256,3)
    lanes = model.predict(test, verbose=1)

    y = lanes[0]
    x = cv2.resize(test[0], (254,126))
    x = np.array([x])
    x = x.reshape(len(x),126,254,3)
    combined = x[0] + (y*2)
    return combined


class Predict(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
      lanes = predict_model()
      plt.imshow(lanes)
      plt.savefig('weights/area-mid-' + str(epoch)+'.png')
      return


model_checkpoint_best = ModelCheckpoint('weights/area-mid.hdf5', monitor='loss',verbose=1,save_best_only=True)

predict_cb = Predict()

from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log/area-mid/',
                         write_images=True,
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
    epochs=10,
    verbose=1, 
    callbacks=[ model_checkpoint_best, predict_cb,tbCallBack])
