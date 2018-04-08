import pandas as pd
import numpy as np

'''
原始数据集体积较大 
采用pandas进行处理
按taxiID进行分割 
分割为较小的txt
'''
def devide(files):
    data=pd.read_csv(files,error_bad_lines=False,header=None)
    data=data.dropna(axis=0)#丢弃空行print(data)
    data.columns = ['t', 'nt', 'name', 'id', 'x','y','d','s','u','nu','w','z']
    data['id']=data['id'].astype(np.int32)
    data=data.sort_values(by=['id','nt'])  #按taxiID和时间进行排序
    data=data[data['id']>=10000]
    #筛除id异常的数据
    #按taxiID分割生成csv
    #print(data)

    groups = data.groupby(data['id'])
    for group in groups:
        group[1].to_csv('raw_gps'+'/'+str(group[0]) + '.csv', index=False)

devide('20160301.txt')
