#coding:utf8
import sys
import os
import time
import numpy as np
import cv2
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

class simpleconv3(nn.Module):
    def __init__(self,nclass):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 5 * 5 , 1024)
        self.fc2 = nn.Linear(1024 , 128)
        self.fc3 = nn.Linear(128 , nclass)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 128 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## 载入人脸检测与关键点检测模型
PREDICTOR_PATH = "../../../models/mouth/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='../../../mouth/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

## 调用 cascade.detectMultiScale 人脸检测器和 Dlib 的关键点检测算法 predictor 获得关键点结果
def get_landmarks(im,rects):
    x,y,w,h =rects[0]
    rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h)) # 获取人脸的四个属性值，左上角坐标 x,y 、高宽 w、h
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def getlipfromimage(im, landmarks):
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    # 根据最外围的关键点获取包围嘴唇的最小矩形框
    # 68 个关键点是从
    # 左耳朵0 -下巴-右耳朵16-左眉毛（17-21）-右眉毛（22-26）-左眼睛（36-41）
    # 右眼睛（42-47）-鼻子从上到下（27-30）-鼻孔（31-35）
    # 嘴巴外轮廓（48-59）嘴巴内轮廓（60-67）
    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    #print("xmin=", xmin)
    #print("xmax=", xmax)
    #print("ymin=", ymin)
    #print("ymax=", ymax)

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    roi = im[int(newy):int(newy) + int(dstlen), int(newx):int(newx) + int(dstlen), 0:3]
    return roi,int(newy),int(newx),dstlen

## 载入预训练模型并设置为推理模型
modelpath = "../../../models/mouth/model.pt"
testsize = 48
torch.no_grad()
## 设置推理图像大小，与预处理函数
data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
## 类定义
class Mouth:
    def __init__(self):
        self.modelpath = modelpath
        self.testsize = testsize
        self.net = simpleconv3(4)
        self.net.eval()
        self.net.load_state_dict(torch.load(self.modelpath,map_location=lambda storage,loc: storage))
        self.data_transforms = data_transforms
    def process(self,im):
        ## 获得嘴唇区域
        try:
            rects = cascade.detectMultiScale(im, 1.3,5)
            landmarks = get_landmarks(im,rects)
            roi,offsety,offsetx,dstlen = getlipfromimage(im, landmarks)
            print(roi.shape)
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            inferenceimage = cv2.resize(roi,(self.testsize,self.testsize),interpolation=cv2.INTER_NEAREST)
            imgblob = self.data_transforms(inferenceimage).unsqueeze(0)
            ## 获得分类结果
            predict = F.softmax(self.net(imgblob)).cpu().data.numpy().copy()
            print(predict)
            index = np.argmax(predict)
            return str(index)
        except:
            return "-1"

## 创建类实例
ms = Mouth()
def mouth(img_path):
    im = cv2.imread(img_path,1)
    print("img.shape"+str(im.shape))
    result = ms.process(im)
    print("result="+str(result))
    return result

if __name__== '__main__':
    ms = Mouth()
    images = os.listdir(sys.argv[1])
    for image in images:
        img = cv2.imread(os.path.join(sys.argv[1],image))
        result = ms.process(img)
        print(result)
        
