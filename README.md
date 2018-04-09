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
|mlp|0.628|0.000|[512, 256, 128] neurons|
|XGBoost|0.211|0.000|max_depth: 40, eta: 0.1, silent: 0|
|RF|0.117|0.000|max_depth: 30, n_estimators: 10,min_samples_split: 2|
|SVM|---|0.000|C: 1.0, kernel: rbf, degree: 3|

## 文件组织方式
|文件名称|文件描述|
|:--:|:--|
|data文件夹|存放处理好的训练集数据及测试集数据|
|logs文件夹|存放训练日志|
|models文件夹|存放训练好的模型|
|raw_data文件夹|存放原始16年3、4月份的出租车数据|
|util文件夹|存放数据预处理的代码|
|README.md|项目描述|




