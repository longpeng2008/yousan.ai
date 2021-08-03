#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import sys
import numpy as np
import cv2
import os
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F

## 全局变量
## sys.argv[1] 权重文件
## sys.argv[2] 图像文件夹

testsize = 48 ##测试图大小
from net import simpleconv3
net = simpleconv3(2) ## 定义模型
net.eval() ## 设置推理模式，使得dropout和batchnorm等网络层在train和val模式间切换
torch.no_grad() ## 停止autograd模块的工作，以起到加速和节省显存

## 载入模型权重
modelpath = sys.argv[1] 
net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))

## 定义预处理函数
data_transforms =  transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

## 读取3通道图片，并扩充为4通道tensor
imagepath = sys.argv[2]
image = Image.open(imagepath)
imgblob = data_transforms(image).unsqueeze(0)

## 获得预测结果predict，得到预测的标签值label
predict = net(imgblob)
index = np.argmax(predict.detach().numpy())
## print(predict)
## print(index)

if index == 0:
    print('the predict of '+sys.argv[2]+' is '+str('none'))
else:
    print('the predict of '+sys.argv[2]+' is '+str('smile'))

