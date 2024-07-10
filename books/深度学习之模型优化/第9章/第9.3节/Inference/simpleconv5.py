#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## 简单模型定义
class simpleconv5(nn.Module):
    def __init__(self,nclass):
        super(simpleconv5,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, nclass)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = nn.AvgPool2d(7)(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    model = simpleconv5(20)
    y = model(x)
