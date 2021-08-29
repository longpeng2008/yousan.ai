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
from torch import nn
class simpleNet5(nn.Module):
    def __init__(self):
        super(simpleNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, 3, 2, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.classifier = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x):       
        out = self.conv1(x)     
        out = self.conv2(out)     
        out = self.conv3(out)   
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.deconv1(out) 
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    net = simpleNet5()
    sample = net(img)
    print(sample.shape)
