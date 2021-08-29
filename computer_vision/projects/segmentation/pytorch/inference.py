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
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import cv2
import sys
import torch.nn.functional as F
import numpy as np

data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

modelpath = sys.argv[1] #模型目录
net = torch.load(modelpath,map_location='cpu')
net.eval() #设置为推理模式，不会更新模型的k，b参数

imagepaths = os.listdir(sys.argv[2]) #测试图片目录
torch.no_grad() #停止autograd模块的工作，加速和节省显存

for imagepath in imagepaths:
    image = cv2.imread(os.path.join(sys.argv[2],imagepath)) #读取图像
    image = cv2.resize(image,(224,224),interpolation=cv2.INTER_NEAREST)
    imgblob = data_transforms(image).unsqueeze(0) #填充维度，从3维到4维
    predict = F.softmax(net(imgblob)).cpu().data.numpy().copy() #获得原始网络输出，多通道
    predict = np.argmax(predict, axis=1) #得到单通道label
    result = np.squeeze(predict) #降低维度，从4维到3维
    print(np.max(result)) 
    result = (result*127).astype(np.uint8) #灰度拉伸，方便可视化

    resultimage = image.copy()
    for y in range(0,result.shape[0]): 
        for x in range(0,result.shape[1]):
            if result[y][x] == 127:
                resultimage[y][x] = (0,0,255) 
            elif result[y][x] == 254:
                resultimage[y][x] = (0,255,255) 

    combineresult = np.concatenate([image,resultimage],axis=1)
    cv2.imwrite(os.path.join(sys.argv[3],imagepath),combineresult) #写入新的目录
