# 路况预测
## 使用xgboost,lstm,cnn,rnn,svm,rf,mlp,embedding-mlp方法预测路况的拥堵程度

## 道路路况预测

## 数据格式

## 所用模型

## 测试结果

|模型/方法|Tranin Acc|Dev Acc|备注说明|
|:--:|:--:|:--:|:--:|
|mlp|0.753|0.000|[512, 256, 128] neurons|
|XGBoost|0.739|0.000|max_depth: 40, eta: 0.1, silent: 0|
|RF|0.723|0.000|max_depth: 30, n_estimators: 10,min_samples_split: 2|
|SVM|0.705|0.000|C: 1.0, kernel: rbf, gamma: 1/n features|
