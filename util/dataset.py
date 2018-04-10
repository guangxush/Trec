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

# embedding_mlp数据处理
def load_data_v3(data_path):
    # 输入数据参数
    tid, grid, direction, tStamp, dur, dis = [], [], [], [], [], []
    # 测试输入参数
    test_tid, test_grid, test_direction, test_tStamp, test_dur, test_dis = [], [], [], [], [], []
    # 结果
    target = []
    test_target = []
    train_data = open(os.path.join(data_path,'train.txt'), 'rb')
    test_data = open(os.path.join(data_path,'test.txt'), 'rb')
    # 读取输入数据，按照列分割
    for line in train_data:
        temp = line.strip().split(',')
        tid.append(temp[0])
        grid.append(temp[1])
        direction.append(temp[2])
        tStamp.append(temp[3])
        dur.append(temp[4])
        dis.append(temp[5])
        target.append(temp[6])
    # 将输入数据转换成字典
    fulList = tid + grid + direction + tStamp + dur + dis
    ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))
    to_idx = lambda x: [ful2idx[word] for word in x]
    tid_array = to_idx(tid)
    grid_array = to_idx(grid)
    direction_array = to_idx(direction)
    tStamp_array = to_idx(tStamp)
    dur_array = to_idx(dur)
    dis_array = to_idx(dis)
    in_array = np.column_stack((tid_array, grid_array, direction_array, tStamp_array, dur_array, dis_array))
    target_array = np.asarray(target, dtype='int16')
    target_array = np_utils.to_categorical(target_array, nb_classes)
    # 生成测试集数据
    for line in test_data:
        temp = line.strip().split(',')
        test_tid.append(temp[0])
        test_grid.append(temp[1])
        test_direction.append(temp[2])
        test_tStamp.append(temp[3])
        test_dur.append(temp[4])
        test_dis.append(temp[5])
        test_target.append(temp[6])
    test_tid_array = to_idx(test_tid)
    test_grid_array = to_idx(test_grid)
    test_direction_array = to_idx(test_direction)
    test_tStamp_array = to_idx(test_tStamp)
    test_dur_array = to_idx(test_dur)
    test_dis_array = to_idx(test_dis)
    test_in_array = np.column_stack((test_tid_array, test_grid_array, test_direction_array, test_tStamp_array, test_dur_array, test_dis_array))
    test_target_array = np.asarray(test_target, dtype='int16')
    test_target_array = np_utils.to_categorical(test_target_array, nb_classes)
    n_status = len(ful2idx.keys()) # 得到分类数目
    return in_array, target_array, test_in_array, test_target_array, n_status

# xgboost/rf数据读取
def load_data_v4(data_path):
    # 训练集输入数据特征
    tid, grid, direction, tStamp, dur, dis = [], [], [], [], [], []
    # 测试集输入数据特征
    test_tid, test_grid, test_direction, test_tStamp, test_dur, test_dis = [], [], [], [], [], []
    # 结果分类
    target = []
    test_target = []
    print('start reading files ...')
    # 读取数据集
    fr = open(os.path.join(data_path,'train.txt'), 'rb')  # train.txt
    test_data = open(sys.argv[2], 'rb')  # test.txt
    # 训练数据处理
    for line in fr:
        # 输入特征
        tmp = line.strip().split(',')
        tid.append(tmp[0])
        grid.append(tmp[1])
        direction.append(tmp[2])
        tStamp.append(tmp[3])
        dur.append(tmp[4])
        dis.append(tmp[5])
        # 输出特征
        target.append(int(tmp[6]))
    # 添加索引
    fulList = tid + grid + direction + tStamp + dur + dis
    ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))
    # 编号
    to_idx = lambda x: [ful2idx[word] for word in x]
    tid_array = to_idx(tid)
    grid_array = to_idx(grid)
    direction_array = to_idx(direction)
    tStamp_array = to_idx(tStamp)
    dur_array = to_idx(dur)
    dis_array = to_idx(dis)
    in_array = np.column_stack((tid_array, grid_array, direction_array, tStamp_array, dur_array, dis_array))
    target_array = np.array(target)
    print('training data is ready ...')
    # 测试集数据处理
    for line in test_data:
        tmp = line.strip().split(',')
        test_tid.append(tmp[0])
        test_grid.append(tmp[1])
        test_direction.append(tmp[2])
        test_tStamp.append(tmp[3])
        test_dur.append(tmp[4])
        test_dis.append(tmp[5])
        test_target.append(int(tmp[6]))
    test_tid_array = to_idx(test_tid)
    test_grid_array = to_idx(test_grid)
    test_direction_array = to_idx(test_direction)
    test_tStamp_array = to_idx(test_tStamp)
    test_dur_array = to_idx(test_dur)
    test_dis_array = to_idx(test_dis)
    test_in_array = np.column_stack(
        (test_tid_array, test_grid_array, test_direction_array, test_tStamp_array, test_dur_array, test_dis_array))
    test_target_array = np.array(test_target)
    print('testing data is ready ...')
    return in_array, target_array, test_in_array, test_target_array

# cnn/rnn/lstm神经网络数据读取
def load_data_v5(data_path):
    # 测试输入数据
    tid, grid, direction, tStamp = [], [], [], []
    # 测试数据输入
    test_tid, test_grid, test_direction, test_tStamp = [], [], [], []
    # 结果分类
    target = []
    test_target = []
    # 读取文件
    fr = open(os.path.join(data_path,'400_7Days.txt'), 'rb')
    test_data = open(os.path.join(data_path,'400_test.txt'), 'rb')
    # 生成训练集数据
    for line in fr:
        tmp = line.strip().split(',')
        tid.append((tmp[0]))
        grid.append(tmp[1])
        direction.append(tmp[2])
        tStamp.append(tmp[3])
        target.append(tmp[4])
    # 生成索引
    fulList = tid + grid + direction + tStamp
    ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))
    n_status = len(ful2idx.keys())
    to_idx = lambda x: [ful2idx[word] for word in x]
    tid_array = to_idx(tid)
    grid_array = to_idx(grid)
    direction_array = to_idx(direction)
    tStamp_array = to_idx(tStamp)

    in_array = np.column_stack((tid_array, grid_array, direction_array, tStamp_array))
    in_array = np.reshape(in_array, (in_array.shape[0], status_maxlen, 1))
    in_array = in_array / float(n_status)
    target_array = np.asarray(target, dtype='int16')
    target_array = np_utils.to_categorical(target_array, nb_classes)

    # 生成测试集数据
    for line in test_data:
        tmp = line.strip().split(',')
        test_tid.append(tmp[0])
        test_grid.append(tmp[1])
        test_direction.append(tmp[2])
        test_tStamp.append(tmp[3])
        test_target.append(tmp[4])

    test_tid_array = to_idx(test_tid)
    test_grid_array = to_idx(test_grid)
    test_direction_array = to_idx(test_direction)
    test_tStamp_array = to_idx(test_tStamp)
    status_maxlen = 4
    nb_classes = 1823
    test_in_array = np.column_stack((test_tid_array, test_grid_array, test_direction_array, test_tStamp_array))
    test_in_array = np.reshape(test_in_array, (test_in_array.shape[0], status_maxlen, 1))
    test_in_array = test_in_array / float(n_status)
    test_target_array = np.asarray(test_target, dtype='int16')
    test_target_array = np_utils.to_categorical(test_target_array, nb_classes)
    # print(target_array.shape[1])
    return in_array, target_array, test_in_array, test_target_array

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