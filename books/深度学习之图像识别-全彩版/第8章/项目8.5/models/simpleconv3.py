#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import numpy as np

## 简单模型定义
class simpleconv3(nn.Module):
    def __init__(self,nclass):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, nclass)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.AvgPool2d(6)(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

## 简单更小模型定义
class simpleconv3small(nn.Module):
    def __init__(self,nclass):
        super(simpleconv3small,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, nclass)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.AvgPool2d(6)(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    x = Variable(torch.randn(1,3,48,48))
    model = simpleconv3(4)
    y = model(x)
    g = make_dot(y)
    g.view()
