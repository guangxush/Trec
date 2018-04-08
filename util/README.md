# util文件说明

## new_devide.py
- 按taxiID将原始数据文件切割为成小csv文件，每个csv对应一个taxiID的某一天原始数据
-
  ```python new_devide.py 'data.txt'```   
  其中data.txt代表某天原始数据txt文件

## dig.py
- 对每一出租车计算其寻客情况数据，数据写入pre.txt中
- 
  ```python dig.py```

## g_train_data.py
- 对pre.txt数据进一步处理，生成标准data的格式
- ```python g_train_data.py pre.txt ceil_size```  
其中pre.txt为粗糙的寻客数据，ceil_size为网格处理的单元网格大小 选为100  
结果将生成input100.txt---标准的预处理数据格式
|taxiID|出发点网格|方向|出发时间|寻客时间|目标网格变化|目标网格ID|
|-|-|-|-|-|-|-|
|10066|194-193|a|2.0|1094|-15x0y|18253|

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

