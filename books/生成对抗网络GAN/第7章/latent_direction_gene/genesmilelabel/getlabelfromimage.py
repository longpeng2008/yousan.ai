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
import shutil

import torchvision
from torchvision import datasets, models, transforms

latent = 'w' ## 'z' or 'w'
debug = False ## 显示分类结果

## 载入人脸检测与关键点检测模型
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

smilenpyfile = open('smilenpy.txt','w') ##存储微笑属性向量
nonenpyfile = open('nonenpy.txt','w')
othernpyfile = open('othernpy.txt','w')

# 调用 cascade.detectMultiScale 人脸检测器和 Dlib 的关键点检测算法 predictor 获得关键点结果
def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h) # 获取人脸的四个属性值，左上角坐标 x,y 、高宽 w、h
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

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
modelpath = 'model.pt'
from net import simpleconv3
net = simpleconv3(4)
net.eval()
net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))
torch.no_grad()

## 设置推理图像大小，与预处理函数
testsize = int(48)
data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

## 对图像进行分类
def testimages(rootdir):
    list_dirs = os.walk(rootdir)
    for root, dirs, files in list_dirs:
        for f in files:
            fileid = f.split('.')[0]
            filepath = os.path.join(root, f)
            try:
                im = cv2.imread(filepath, 1)
                rects = cascade.detectMultiScale(im, 1.3,5)
                x,y,w,h =rects[0]
                rect=dlib.rectangle(x,y,x+w,y+h)
                landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
            except:
                continue ###没有检测到人脸
        
            ## 获得嘴唇区域
            roi,offsety,offsetx,dstlen = getlipfromimage(im, landmarks)
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            
            ## 图像预处理
            inferenceimage = cv2.resize(roi,(testsize,testsize),interpolation=cv2.INTER_NEAREST)
            imgblob = data_transforms(inferenceimage).unsqueeze(0)
        
            ## 获得分类结果
            predict = F.softmax(net(imgblob)).cpu().data.numpy().copy()
            index = np.argmax(predict)
   
            ## 显示分类结果
            im_h,im_w,im_c = im.shape
            pos_x = int(offsetx+dstlen)
            pos_y = int(offsety+dstlen)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if debug:
                cv2.namedWindow('roi',0)
                cv2.imshow('roi',roi)
                print(str(predict)+str(index))
                cv2.rectangle(im,(int(offsetx),int(offsety)),(int(offsetx+dstlen),int(offsety+dstlen)),(0,255,255), 2)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    break

            if index == 2:
                smilenpyfile.write(filepath.replace('.png','.npy').replace('images',latent)+'\n')

            elif index == 0:
                nonenpyfile.write(filepath.replace('.png','.npy').replace('images',latent)+'\n')
            else:
                othernpyfile.write(filepath.replace('.png','.npy').replace('images',latent)+'\n')


if __name__ == '__main__':
     testimages(sys.argv[1])   
