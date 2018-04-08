import os, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

#acc: 0.723386

# input attributes
tid, grid, tStamp, dst, direc, dis, wth, days = [], [], [], [], [], [], [], []

# test input
test_tid, test_grid, test_tStamp, test_dst, test_direc, test_dis, test_wth, test_days = [], [], [], [], [], [], [], []

# result classes
target = []
test_target = []

print 'start reading files ...'

# read file
fr = open(sys.argv[1], 'rb') # train.txt
test_data = open(sys.argv[2], 'rb') # test.txt

# prepare train data
for line in fr:
	# input
	tmp = line.strip().split(',')
	tid.append(tmp[0])
	grid.append(tmp[1])
	tStamp.append(tmp[2])
	dst.append(tmp[3])
	direc.append(tmp[4])
	dis.append(tmp[5])
	wth.append(tmp[6])
	days.append(tmp[7])
	# output
	target.append(int(tmp[8]))

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
target_array = np.array(target)

print 'training data is ready ...'

# prepare test data
for line in test_data:
    tmp = line.strip().split(',')
    test_tid.append(tmp[0])
    test_grid.append(tmp[1])
    test_tStamp.append(tmp[2])
    test_dst.append(tmp[3])
    test_direc.append(tmp[4])
    test_dis.append(tmp[5])
    test_wth.append(tmp[6])
    test_days.append(tmp[7])

    test_target.append(int(tmp[8]))

test_tid_array = to_idx(test_tid)
test_grid_array = to_idx(test_grid)
test_direc_array = to_idx(test_direc)
test_tStamp_array = to_idx(test_tStamp)
test_dst_array = to_idx(test_dst)
test_dis_array = to_idx(test_dis)
test_wth_array = to_idx(test_wth)
test_days_array = to_idx(test_days)

test_in_array = np.column_stack((test_tid_array, test_grid_array, test_tStamp_array, test_dst_array, test_direc_array, test_dis_array, test_wth_array, test_days_array))
test_target_array = np.array(test_target)

print 'testing data is ready ...'

# random forest method                                             
                                                                        
print 'start training ...'                                              

rf = RFC(verbose=1)

rf.fit(in_array, target_array)
                                                                        
# get prediction                                                        
                                                                        
print 'start testing ...'                                               
                                                                        
pred = rf.predict(test_in_array)

print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_target_array[i] for i in range(len(test_target_array))) / float(len(test_target_array)) ))
