# Trec
**TRec: A Taxi Recommender System for Finding Passengers via Deep Neural Networks**

## Requirement
  Python 2.7/3.5<br/>
  Keras 2.x<br/>
  sklearn<br/>
  pandas<br/>
  numpy<br/>
  Tensorflow

## Run
代码运行方式如下：
* 训练过程：
  <pre>python mlp.py  train</pre>
* 测试过程：
  <pre>python mlp.py  test</pre>

## 测试结果

|模型/方法|Tranin Acc|Dev Acc|备注说明|
|:--:|:--:|:--:|:--:|
|mlp|0.00|0.00|多层神经网络|

## 文件组织方式
|文件名称|文件描述|
|:--:|:--|
|data文件夹|存放处理好的训练集数据及测试集数据|
|logs文件夹|存放训练日志|
|models文件夹|存放训练好的模型|
|raw_data文件夹|存放原始16年3、4月份的出租车数据|
|util文件夹|存放数据预处理的代码|
|README.md|项目描述|




