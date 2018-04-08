# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import xgboost as xgb
from util.dataset import load_data_v4
nb_classes = 2823
#merror: 0.893915

def boost_method(in_array, target_array, test_in_array, test_target_array):
    xg_train = xgb.DMatrix(in_array, label=target_array)
    xg_test = xgb.DMatrix(test_in_array, label=test_target_array)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 4
    param['silent'] = 0
    param['nthread'] = 16
    param['num_class'] = 2823
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 2
    print 'start training ...'
    bst = xgb.train(param, xg_train, num_round, watchlist)
    # get prediction
    print 'start testing ...'
    pred = bst.predict(xg_test)
    return pred


if __name__ == '__main__':
    in_array, target_array, test_in_array, test_target_array = load_data_v4('old_data')
    pred = boost_method(in_array=in_array, target_array=target_array, test_in_array=test_in_array,
                 test_target_array=test_target_array)
    print ('predicting, classification error=%f' % (
                sum(int(pred[i]) != test_target_array[i] for i in range(len(test_target_array))) / float(
            len(test_target_array))))

