# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
from collections import Counter
from keras.layers import Input, Embedding, Dense, Reshape, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from util.dataset import load_data_v3, make_submission

nb_epoch = 50
batch_size = 2048
status_maxlen = 6
n_embed_dims = 10
nb_classes = 2823

def embedding_mlp(n_status):
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
    return model

if __name__ == '__main__':
    in_array, target_array, test_in_array, test_target_array, n_status = load_data_v3(data_path='old_data')
    model = embedding_mlp(n_status)
    checkpoint = ModelCheckpoint('models/weigts.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0,
                                 save_best_only=True, mode='auto')
    callback_list = [checkpoint]
    model.fit(in_array, target_array, nb_epoch=nb_epoch, verbose=1, batch_size=batch_size, callbacks=callback_list,
              validation_data=[test_in_array, test_target_array])
    prob = model.predict(test_in_array, batch_size=batch_size)
    target_classes = [np.argmax(class_list) for class_list in prob]
    print(Counter(target_classes).most_common(100))
