# -*- coding: utf-8 -*-
# random forest method
from sklearn.ensemble import RandomForestClassifier as RFC
from util.dataset import load_data_v4

if __name__ == '__main__':
    in_array, target_array, test_in_array, test_target_array = load_data_v5(in_array, target_array, test_in_array,
                                                                            test_target_array)
    rf.fit(in_array, target_array)
    # get prediction
    print('start testing ...')
    pred = rf.predict(test_in_array)
    print('predicting, classification error=%f' % (
                sum(int(pred[i]) != test_target_array[i] for i in range(len(test_target_array))) / float(
            len(test_target_array))))