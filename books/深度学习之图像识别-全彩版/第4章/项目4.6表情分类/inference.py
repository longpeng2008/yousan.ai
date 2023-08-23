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
import torchvision
from torchvision import datasets, models, transforms
import time
from PIL import Image
import torch.nn.functional as F

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


## 全局变量
## sys.argv[1] 权重文件
## sys.argv[2] 图像文件夹

testsize = 48 ##测试图大小
from net import simpleconv3
net = simpleconv3(4) ## 定义模型
net.eval() ## 设置推理模式，使得dropout和batchnorm等网络层在train和val模式间切换
torch.no_grad() ## 停止autograd模块的工作，以起到加速和节省显存

## 载入模型权重
modelpath = sys.argv[1] 
net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))

## 定义预处理函数
data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

#一次测试一个文件
imagepaths = os.listdir(sys.argv[2])
for imagepath in imagepaths:
    im = cv2.imread(os.path.join(sys.argv[2],imagepath),1)
    try:
        rects = cascade.detectMultiScale(im, 1.3,5)
        x,y,w,h =rects[0]
        rect=dlib.rectangle(x,y,x+w,y+h)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    except:
        continue ###没有检测到人脸

    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    
    for i in range(48,67):
        x = landmarks[i,0]
        y = landmarks[i,1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    
    roiwidth = xmax - xmin
    roiheight = ymax - ymin
    
    roi = im[ymin:ymax,xmin:xmax,0:3]
    
    if roiwidth > roiheight:
        dstlen = 1.5*roiwidth
    else:
        dstlen = 1.5*roiheight
    
    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight
    
    newx = xmin
    newy = ymin
    
    imagerows,imagecols,channel = im.shape
    if newx >= diff_xlen/2 and newx + roiwidth + diff_xlen/2 < imagecols:
        newx  = newx - diff_xlen/2;
    elif newx < diff_xlen/2:
        newx = 0;
    else:
        newx =  imagecols - dstlen;
    
    if newy >= diff_ylen/2 and newy + roiheight + diff_ylen/2 < imagerows:
        newy  = newy - diff_ylen/2;
    elif newy < diff_ylen/2:
        newy = 0;
    else:
        newy =  imagerows - dstlen;
    
    roi = im[int(newy):int(newy+dstlen),int(newx):int(newx+dstlen),0:3]
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    roiresized = cv2.resize(roi,(testsize,testsize))
    imgblob = data_transforms(roiresized).unsqueeze(0)
    imgblob.requires_grad = False
    torch.no_grad()
    predict = F.softmax(net(imgblob))
    print(predict)
    index = np.argmax(predict.detach().numpy())
   
    im_show = cv2.imread(os.path.join(sys.argv[2],imagepath),1)
    im_h,im_w,im_c = im_show.shape
    pos_x = int(newx+dstlen)
    pos_y = int(newy+dstlen)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(im_show,(int(newx),int(newy)),(int(newx+dstlen),int(newy+dstlen)),(0,255,255), 2)
    if index == 0:
        cv2.putText(im_show, 'none', (pos_x,pos_y), font, 1.2, (0,0,255), 1)
    elif index == 1:
        cv2.putText(im_show, 'pouting', (pos_x,pos_y), font, 1.2, (0,0,255), 1)
    elif index == 2:
        cv2.putText(im_show, 'smile', (pos_x,pos_y), font, 1.2, (0,0,255), 1)
    elif index == 3:
        cv2.putText(im_show, 'open', (pos_x,pos_y), font, 1.2, (0,0,255), 1)
    cv2.namedWindow('result',0)
    cv2.imshow('result',im_show)
    cv2.imwrite(os.path.join('results',imagepath),im_show)
    cv2.waitKey(0) 
