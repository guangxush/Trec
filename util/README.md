# util文件说明

## new_devide.py
- 按taxiID将原始数据文件切割为成小csv文件，每个csv对应一个taxiID的某一天原始数据
-
  ```python new_devide.py 'data.txt'``` 其中data.txt代表某天原始数据txt文件

## dig.py
- 对每一出租车计算其寻客情况数据，数据写入pre.txt中
- 
  ```python dig.py```

## g_train_data.py
- 对pre.txt数据进一步处理，生成标准data的格式

## my_Time.py
- 对数据中的时间项进行标准化处理

## direction.py
- 对数据中的寻客方向进行标准化处理

## geo_dis.py
- 对数据中的经纬度距离进行计算和标准化处理

## dataset.py
- 将已处理的数据（data）转换成机器学习所需的数据格式
-
  <pre>python dataset.py 'data_path'</pre>
  其中datapath参数代表已处理数据的文件路径如data

