#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

## 导入逻辑回归模型函数
from sklearn.linear_model import LogisticRegression

##Demo演示LogisticRegression分类

## 构造数据集
import glob
postxt = sys.argv[1]
negtxt = sys.argv[2]

possamples = open(postxt,'r').readlines()
negsamples = open(negtxt,'r').readlines()

#possamples = glob.glob(os.path.join(posdir,'*.npy'))
#negsamples = glob.glob(os.path.join(negdir,'*.npy'))
x_features = []
y_label = []
for sample in possamples:
    sample = sample.strip()
    feature = np.squeeze(np.load(sample))
    x_features.append(list(feature))
    y_label.append(1)
for sample in negsamples:
    sample = sample.strip()
    feature = np.squeeze(np.load(sample))
    x_features.append(list(feature))
    y_label.append(0)
x_features = np.array(x_features)
y_label = np.array(y_label)

print("x_features="+str(x_features.shape))
print("y_label="+str(y_label.shape))

#x_features = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
#y_label = np.array([0, 0, 0, 1, 1, 1])

## 调用逻辑回归模型
lr_clf = LogisticRegression()

## 用逻辑回归模型拟合构造的数据集
lr_clf = lr_clf.fit(x_features, y_label) #其拟合方程为 y=w*x+b

## 查看其对应模型的w
print('the weight of Logistic Regression:',lr_clf.coef_)
np.save('smilew.npy',lr_clf.coef_)

## 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:',lr_clf.intercept_)


