import numpy as np # linear algebra
import scipy as scipy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import dill

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2
from tqdm import tqdm

from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
import math

from keras.callbacks import ModelCheckpoint


x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print(label_map)

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (64, 64)))
    y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

split = 35000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

import math

def add_unet_block(depth, width, drop_constant, act, previous_layer):
    prev = previous_layer
    toConcatLater = list()

    for cur_depth in range(depth):
        if cur_depth != 0:
            prev = MaxPooling2D(pool_size=(2, 2))(prev)
        conv_width = int(width * math.pow(2, cur_depth))
        batch = BatchNormalization()(prev)
        conv1 = Conv2D(conv_width, (3, 3), activation=act, padding='same')(batch)
        drop1 = Dropout(0.3)(conv1)
        conv2 = Conv2D(conv_width, (3, 3), activation=act, padding='same')(drop1)
        drop2 = Dropout(0.3)(conv2)
        prev = drop2
        toConcatLater.append(conv2)
    prev = UpSampling2D(size=(2, 2))(prev)
    for cur_depth in reversed(range(depth - 1)):
        merged = concatenate([prev, toConcatLater[cur_depth]], axis=3)
        drop = Dropout(drop_constant)(merged)
        conv_width = int(width * math.pow(2, cur_depth))
        batch = BatchNormalization()(drop)
        conv1 = Conv2D(conv_width, (3, 3), activation=act, padding='same')(batch)
        drop3 = Dropout(0.3)(conv1)
        conv2 = Conv2D(conv_width, (3, 3), activation=act, padding='same')(drop3)
        drop4 = Dropout(0.3)(conv2)
        if cur_depth != 0:
            up = UpSampling2D(size=(2, 2))(drop4)
            prev = up
        else:
            prev = drop4
    return prev

input_layer = Input(shape=(64, 64, 3))
unet_out = add_unet_block(3, 32, 0.4, 'relu', input_layer)
flat = Flatten()(unet_out)
dense1 = Dense(128, activation='relu')(flat)
drop = Dropout(0.5)(dense1)
dense2 = Dense(17, activation='sigmoid')(drop)
unet = Model(inputs=[input_layer], outputs=dense2)

unet.summary()

from keras.optimizers import Adam


checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

epochs_arr = [20, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs_num in zip(learn_rates, epochs_arr):
    adam = Adam(lr=learn_rate)
    unet.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                        optimizer=adam,
                        metrics=['accuracy'])
    unet.fit(x_train, y_train, batch_size=128, epochs=epochs_num, verbose=1,
            validation_data=(x_valid, y_valid), callbacks=[checkpointer])

unet.save('unet_model_with_batchnorm.h5')