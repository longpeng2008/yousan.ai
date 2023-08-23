#coding:utf8

# Copyright 2023 longpeng2008. All Rights Reserved.
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

modelpath = sys.argv[1]
net = torch.load(modelpath,map_location='cpu')
net.eval()

imagepaths = os.listdir(sys.argv[2])
torch.no_grad()
for imagepath in imagepaths:
    image = cv2.imread(os.path.join(sys.argv[2],imagepath))
    image = cv2.resize(image,(160,160),interpolation=cv2.INTER_NEAREST)
    imgblob = data_transforms(image).unsqueeze(0)
    predict = F.softmax(net(imgblob)).cpu().data.numpy().copy()
    predict = np.argmax(predict, axis=1)
    result = np.squeeze(predict)
    print(np.max(result))
    result = (result*127).astype(np.uint8)

    resultimage = image.copy()
    for y in range(0,result.shape[0]): 
        for x in range(0,result.shape[1]):
            if result[y][x] == 127:
                resultimage[y][x] = (0,0,255) 
            elif result[y][x] == 254:
                resultimage[y][x] = (0,255,255) 

    cv2.imwrite(os.path.join(sys.argv[3],imagepath),resultimage)
