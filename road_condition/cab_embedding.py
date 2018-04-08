import numpy as np
import os
import sys

from collections import Counter
from keras.layers import Input, Embedding, Dense, Reshape, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

if not os.path.exists('models'):
    os.makedirs('models')

nb_epoch = 500
batch_size = 2048
status_maxlen = 8
n_embed_dims = 10
nb_classes = 4

# input factors
tid, grid, tStamp, dst, direc, dis, wth, days = [], [], [], [], [], [], [], []

# test input
test_tid, test_grid, test_tStamp, test_dst, test_direc, test_dis, test_wth, test_days = [], [], [], [], [], [], [], []

# result classification
target = []
test_target = []

# read file
fr = open(sys.argv[1], 'rb')
test_data = open(sys.argv[2], 'rb')

# for line in test_data:
#     tmp = ','.join(line.strip().split(',')[1:]) + '\n'
#     new_test_data.write(tmp)
#
# for line in fr:
#     tmp = ','.join(line.strip().split(',')[1:]) + '\n'
#     new_fr.write(tmp)


# generate train
for line in fr:
    tmp = line.strip().split(',')
    tid.append(tmp[0])
    grid.append(tmp[1])
    tStamp.append(tmp[2])
    dst.append(tmp[3])
    direc.append(tmp[4])
    dis.append(tmp[5])
    wth.append(tmp[6])
    days.append(tmp[7])

    target.append(tmp[8])

# add index
fulList = tid + grid + direc + tStamp + dst + dis + wth + days

ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))

# to index
to_idx = lambda x: [ful2idx[word] for word in x]

tid_array = to_idx(tid)
grid_array = to_idx(grid)
direc_array = to_idx(direc)
tStamp_array = to_idx(tStamp)
dst_array = to_idx(dst)
dis_array = to_idx(dis)
wth_array = to_idx(wth)
days_array = to_idx(days)

in_array = np.column_stack((tid_array, grid_array, tStamp_array, dst_array, direc_array, dis_array, wth_array, days_array))
target_array = np.asarray(target, dtype='int8')
target_array = np_utils.to_categorical(target_array, nb_classes)

# generate test
for line in test_data:
    tmp = line.strip().split(',')
    test_tid_array = to_idx(test_tid)
    test_tid.append(tmp[0])
    test_grid.append(tmp[1])
    test_tStamp.append(tmp[2])
    test_dst.append(tmp[3])
    test_direc.append(tmp[4])
    test_dis.append(tmp[5])
    test_wth.append(tmp[6])
    test_days.append(tmp[7])

    test_target.append(tmp[8])

test_tid_array = to_idx(test_tid)
test_grid_array = to_idx(test_grid)
test_direc_array = to_idx(test_direc)
test_tStamp_array = to_idx(test_tStamp)
test_dst_array = to_idx(test_dst)
test_dis_array = to_idx(test_dis)
test_wth_array = to_idx(test_wth)
test_days_array = to_idx(test_days)

test_in_array = np.column_stack((test_tid_array, test_grid_array, test_tStamp_array, test_dst_array, test_direc_array, test_dis_array, test_wth_array, test_days_array))
test_target_array = np.asarray(test_target, dtype='int8')
test_target_array = np_utils.to_categorical(test_target_array, nb_classes)

n_status = len(ful2idx.keys())

input_status = Input(shape=(status_maxlen,), dtype='int32')
input_embedding = Embedding(n_status, n_embed_dims)(input_status)
flatten = Reshape((status_maxlen * n_embed_dims,))(input_embedding)
hidden_layer1 = Dense(500, activation='relu')(flatten)
hidden_layer1_dropout = Dropout(0.25)(hidden_layer1)
# hidden_layer2 = Dense(512, activation='relu')(hidden_layer1_dropout)
# hidden_layer3 = Dense(256, activation='relu')(hidden_layer2)
# hidden_layer4 = Dense(128, activation='relu')(hidden_layer3)
# hidden_layer4_dropout = Dropout(0.5)(hidden_layer4)
output_predict = Dense(nb_classes, activation='softmax')(hidden_layer1_dropout)

model = Model(input=[input_status], output=[output_predict])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/weigts.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0,
                             save_best_only=True, mode='auto')
callback_list = [checkpoint]

model.fit(in_array, target_array, nb_epoch=nb_epoch, verbose=1, batch_size=batch_size, callbacks=callback_list,
          validation_data=[test_in_array, test_target_array])

prob = model.predict(test_in_array, batch_size=batch_size)
target_classes = [np.argmax(class_list) for class_list in prob]
print Counter(target_classes).most_common(100)
