# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandas.core.frame import DataFrame
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from keras.utils import np_utils
import os
import numpy as np
import pandas as pd
import datetime
import time

def load_data_v1(data_path):
    attr_name = ['taxi_id', 'point', 'duration', 'time', 'duration', 'distance']
    # 训练集数据
    train = pd.read_csv(os.path.join(data_path,'train.txt'), header=None)
    train_set = train.values[:,[0, 1, 2, 3, 4, 5, 6]]
    dataset = train.values[:, [0, 1, 2, 3, 4, 5]]
    print(train_set[0])

    # 测试集数据
    test = pd.read_csv(os.path.join(data_path,'test.txt'), header=None)
    test_set = test.values[:, [0, 1, 2, 3, 4, 5, 6]]
    print(test_set[0])

    # 测试集中除去最后一列数据存放于列表中，以出租车ID为主键
    samples = list()
    for sample in dataset:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            sample_dict[attr_name[index]] = attr
        samples.append(sample_dict)

    h = FeatureHasher(n_features=2048)
    h.fit(samples)

    # 训练集数据转换成x,y列表
    x_train = list()
    y_train = list()
    for sample in train_set:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            attr = str(attr)
            if index == 6:
                y_train.append(int(attr))
                continue
            sample_dict[attr_name[index]] = attr
        x_train.append(sample_dict)

    # 测试集数据转换成x,y列表
    x_test = list()
    y_test = list()
    for sample in test_set:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            attr = str(attr)
            if index == 6:
                y_test.append(int(attr))
                continue
            sample_dict[attr_name[index]] = attr
        x_test.append(sample_dict)

    x_train = h.transform(x_train).toarray()
    x_test = h.transform(x_test).toarray()
    print(x_train[0])
    print(x_test[0])
    print(x_train.shape)
    print(x_test.shape)

    y_train = np.asarray(y_train, dtype='int16')
    y_test = np.asarray(y_test, dtype='int16')

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print(y_train.shape)
    print(y_test.shape)

    # return x_train, y_train, x_dev, y_dev, x_test
    return x_train, y_train, x_test, y_test, x_test

# csv加载数据的方式
def load_data_v2(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataset = train_dataframe.values

    x_train = train_dataset[:, 0:-2]
    y_train = train_dataset[:, -2].reshape(-1, 1).astype('float32')

    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev = dev_dataset[:, 0:-2]
    y_dev = dev_dataset[:, -2].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test = test_dataset

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


def make_submission(result_path, results, model_name):
    submit_lines = ['test_id,count\n']
    for index, line in enumerate(results):
        submit_lines.append(','.join([str(index), str(line[0])]) + '\n')
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    result_file_name = 'result_' + model_name + '_' + timestamp + '.csv'
    with open(os.path.join(result_path, result_file_name), mode='w') as result_file:
        result_file.writelines(submit_lines)


if __name__ == '__main__':
    load_data_v1(data_path='../old_data/')