# -*- coding: utf-8 -*-
from util.dataset import load_data_v5
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
from sklearn import preprocessing as pp
import numpy as np
import os

if __name__ == '__main__':
    in_array, target_array, test_in_array, test_target_array = load_data_v5(in_array, target_array, test_in_array,
                                                                            test_target_array)
    X_test = scaler.transform(test_in_array)
    # auto encoder
    input_gps = Input(shape=(in_size,), )
    encoded = Dense(out_size, activation='relu')(input_gps)
    '''
    decoded = Dense(in_size, activation='sigmoid')(encoded)
    autoEncoder = Model(input=input_gps, output=decoded)
    autoEncoder.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    autoEncoder.fit(X, X, nb_epoch=nb_epoch, shuffle=True, validation_data=(X_test, X_test))
    '''
    # create and fit the model
    hidden_layer1 = Dense(500, activation='relu')(encoded)
    hidden_layer_dropout = Dropout(0.05)(hidden_layer1)
    output_predict = Dense(nb_classes, activation='softmax')(hidden_layer_dropout)
    model = Model(input=[input_gps], output=[output_predict])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(in_array, target_array, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
                     validation_data=(test_in_array, test_target_array))
    # summarize performance of the model
    scores = model.evaluate(test_in_array, test_target_array, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))
    '''
    fl = open('mlp_loss.txt', 'w')
    for loss in hist.history['loss']:
        fl.write(repr(loss) + '\n')
    fa = open('mlp_acc.txt', 'w')
    for acc in hist.history['acc']:
        fa.write(repr(acc) + '\n')
    '''
    # demonstrate some model predictions