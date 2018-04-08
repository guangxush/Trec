# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

import numpy as np
import os

# fix random seed for reproducibility
np.random.seed(7)
# define the raw dataset

if not os.path.exists('models'):
    os.makedirs('models')

nb_epoch = 30
batch_size = 2048
status_maxlen = 8
nb_classes = 4

# input factors
tid, grid, tStamp, dst, direc, distance, wth, FX = [], [], [], [], [], [], [], []

# test input
test_tid, test_grid, test_tStamp, test_dst, test_direc, test_distance, test_wth, test_FX = [], [], [], [], [], [], [], []

# result classification
target = []
test_target = []

# read file
fr = open('train.txt', 'rb')
test_data = open('test.txt', 'rb')

# generate train
for line in fr:
    tmp = line.strip().split(',')
    tid.append((tmp[0]))
    grid.append(tmp[1])
    tStamp.append(tmp[2])
    dst.append(tmp[3])
    direc.append(tmp[4])
    distance.append(tmp[5])
    wth.append(tmp[6])
    FX.append(tmp[7])

    target.append(tmp[8])

# add index
fulList = tid + grid + tStamp + dst + direc + distance + wth + FX

ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))

n_status = len(ful2idx.keys())

# to index
to_idx = lambda x: [ful2idx[word] for word in x]

tid_array = to_idx(tid)
grid_array = to_idx(grid)
tStamp_array = to_idx(tStamp)
dst_array = to_idx(dst)
direc_array = to_idx(direc)
distance_array = to_idx(distance)
wth_array = to_idx(wth)
FX_array = to_idx(FX)

in_array = np.column_stack((tid_array, grid_array, tStamp_array, dst_array, direc_array, distance_array, wth_array, FX_array))
in_array = np.reshape(in_array, (in_array.shape[0], status_maxlen, 1))
in_array = in_array / float(n_status)
target_array = np.asarray(target, dtype='int16')
target_array = np_utils.to_categorical(target_array, nb_classes)

# generate test
for line in test_data:
    tmp = line.strip().split(',')
    test_tid.append(tmp[0])
    test_grid.append(tmp[1])
    test_tStamp.append(tmp[2])
    test_dst.append(tmp[3])
    test_direc.append(tmp[4])
    test_distance.append(tmp[5])
    test_wth.append(tmp[6])
    test_FX.append(tmp[7])

    test_target.append(tmp[8])

test_tid_array = to_idx(test_tid)
test_grid_array = to_idx(test_grid)
test_tStamp_array = to_idx(test_tStamp)
test_dst_array = to_idx(test_dst)
test_direc_array = to_idx(test_direc)
test_distance_array = to_idx(test_distance)
test_wth_array = to_idx(test_wth)
test_FX_array = to_idx(test_FX)

test_in_array = np.column_stack((test_tid_array, test_grid_array, test_tStamp_array, test_dst_array, test_direc_array, test_distance_array, test_wth_array, test_FX_array))
test_in_array = np.reshape(test_in_array, (test_in_array.shape[0], status_maxlen, 1))
test_in_array = test_in_array / float(n_status)
test_target_array = np.asarray(test_target, dtype='int16')
test_target_array = np_utils.to_categorical(test_target_array, nb_classes)

# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(in_array.shape[1], 1)))
model.add(Dense(target_array.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(in_array, target_array, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
          validation_data=(test_in_array, test_target_array))
# summarize performance of the model
scores = model.evaluate(test_in_array, test_target_array, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

# demonstrate some model predictions
