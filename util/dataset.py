# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandas.core.frame import DataFrame
import os
import numpy as np
import pandas as pd
import datetime
import time


def load_data_v1(data_path):
    attr_name = ['taxi_id', 'point', 'duration', 'time', 'duration', 'distance']
    train = pd.read_csv('train.txt', header=None)
    train_set = train.values[:,[0, 1, 2, 3, 4, 5, 6]]
    print(train_set[0])

    test = pd.read_csv('test.txt', header=None)
    test_set = test.values[:, [0, 1, 2, 3, 4, 5, 6]]
    print(test_set[0])
    dataset = train.values[:, [0, 1, 2, 3, 4, 5]]

    samples = list()
    for sample in dataset:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            sample_dict[attr_name[index]] = attr
        samples.append(sample_dict)
    h = FeatureHasher(n_features=2048)
    h.fit(samples)
    X_train = list()
    y_train = list()

    X_test = list()
    y_test = list()
    for sample in train_set:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            attr = str(attr)
            if index == 6:
                y_train.append(int(attr))
                continue
            sample_dict[attr_name[index]] = attr
        X_train.append(sample_dict)

    for sample in test_set:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            attr = str(attr)
            if index == 6:
                y_test.append(int(attr))
                continue
            sample_dict[attr_name[index]] = attr
        X_test.append(sample_dict)

    X_train = h.transform(X_train).toarray()
    X_test = h.transform(X_test).toarray()
    print(X_train[0])
    print(X_test[0])
    print(X_train.shape)
    print(X_test.shape)

    y_train = np.asarray(y_train, dtype='int16')
    y_test = np.asarray(y_test, dtype='int16')
    nb_classes = np.max(y_train) + 1
    nb_test_classes = np.max(y_test) + 1
    print('nb_classes: ', nb_classes)
    print('nb_test_classes: ', nb_test_classes)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print(y_train.shape)
    print(y_test.shape)

def make_submission(result_path, results, model_name):
    submit_lines = ['test_id,count\n']
    for index, line in enumerate(results):
        submit_lines.append(','.join([str(index), str(line[0])]) + '\n')
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    result_file_name = 'result_' + model_name + '_' + timestamp + '.csv'
    with open(os.path.join(result_path, result_file_name), mode='w') as result_file:
        result_file.writelines(submit_lines)


if __name__ == '__main__':
    load_data_v1(data_path='../data/')