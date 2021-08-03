#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## 3层卷积神经网络simpleconv3定义
## 包括3个卷积层，3个BN层，3个ReLU激活层，3个全连接层

class simpleconv3(nn.Module):
    ## 初始化函数
    def __init__(self,nclass):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2) #输入图片大小为3*48*48，输出特征图大小为12*23*23，卷积核大小为3*3，步长为2
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2) #输入图片大小为12*23*23，输出特征图大小为24*11*11，卷积核大小为3*3，步长为2
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2) #输入图片大小为24*11*11，输出特征图大小为48*5*5，卷积核大小为3*3，步长为2
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200) #输入向量长为48*5*5=1200，输出向量长为1200
        self.fc2 = nn.Linear(1200 , 128) #输入向量长为1200，输出向量长为128
        self.fc3 = nn.Linear(128 , nclass) #输入向量长为128，输出向量长为nclass，等于类别数

    ## 前向函数
    def forward(self, x):
        ## relu函数，不需要进行实例化，直接进行调用
        ## conv，fc层需要调用nn.Module进行实例化
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    import torch
    x = torch.randn(1,3,48,48)
    model = simpleconv3(2)
    y = model(x)
    print(model)
