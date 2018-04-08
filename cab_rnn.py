# -*- coding: utf-8 -*-
from util.dataset import load_data_v5
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN as RNN
from keras.utils import np_utils
import numpy as np
import os
nb_epoch = 100
batch_size = 2048
# fix random seed for reproducibility
np.random.seed(7)
# define the raw dataset
if not os.path.exists('models'):
    os.makedirs('models')

def rnn(in_array, target_array):
    # create and fit the model
    model = Sequential()
    model.add(RNN(512, input_shape=(in_array.shape[1], 1)))
    model.add(Dense(target_array.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    in_array, target_array, test_in_array, test_target_array = load_data_v5(in_array, target_array, test_in_array,
                                                                            test_target_array)
    model = rnn(in_array, target_array)
    hist = model.fit(in_array, target_array, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
                     validation_data=(test_in_array, test_target_array))
    # summarize performance of the model
    scores = model.evaluate(test_in_array, test_target_array, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))
    fl = open('rnn_loss.txt', 'w')
    for loss in hist.history['loss']:
        fl.write(repr(loss) + '\n')
    fa = open('rnn_acc.txt', 'w')
    for acc in hist.history['acc']:
        fa.write(repr(acc) + '\n')