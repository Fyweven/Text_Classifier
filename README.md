# Text_Classifier
Some deep learning model for Text Classifier by Pytorch


Models/ 下有四个深度学习 分类模型

Text_CNN.py 为： Yoon Kim在论文(2014 EMNLP) Convolutional Neural Networks for Sentence Classification

Text_LSTM.py 为：LSTM 对文本编码，然后分类。 并实现了LSTM +  Maxpool和self-attention的方法

Text_RCNN.py 为： 第一层网络LSTM， 之后接CNN编码，最后MLP分类

CNN_Inception.py 为：一个借鉴谷歌的GoogLeNet 的Inception思想， 实现的分类模型


scripts/ 下为一些数据的处理脚本，训练测试数据的加载。 数据格式为：txt+'\t'+label


utils/ 下有一些超参数的设置，中文字符清洗等


train/ 为模型的训练，评估等
