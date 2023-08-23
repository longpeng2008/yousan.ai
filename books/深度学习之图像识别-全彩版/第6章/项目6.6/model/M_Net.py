"""
Matting network : M-Net
Author: Zhengwei Li
Date  : 2018/12/24
"""

import torch
import torch.nn as nn
## encoder + decoder
class M_net(nn.Module):
    def __init__(self, classes=2):
        super(M_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder
        # ---------------------
        # stride=2
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(6, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # stride=4
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        # stride=8
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # stride=16
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # stride=8
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.deconv_1 = nn.ConvTranspose2d(128, 128, 5, 2, 2, 1, bias=False)

        # stride=4
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 5, 2, 2, 1, bias=False)

        # stride=2
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.deconv_3 = nn.ConvTranspose2d(32, 32, 5, 2, 2, 1, bias=False)

        # stride=1
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.deconv_4 = nn.ConvTranspose2d(16, 16, 5, 2, 2, 1, bias=False)

        self.conv = nn.Conv2d(16, 1, 5, 1, 2, bias=False)


    def forward(self, input):
        # encoder
        x = self.en_conv_bn_relu_1(input)
        x = self.max_pooling_1(x)

        x = self.en_conv_bn_relu_2(x)
        x = self.max_pooling_2(x)

        x = self.en_conv_bn_relu_3(x)
        x = self.max_pooling_3(x)

        x = self.en_conv_bn_relu_4(x)
        x = self.max_pooling_4(x)

        # decoder
        x = self.de_conv_bn_relu_1(x)
        x = self.deconv_1(x)
        x = self.de_conv_bn_relu_2(x)
        x = self.deconv_2(x)

        x = self.de_conv_bn_relu_3(x)
        x = self.deconv_3(x)

        x = self.de_conv_bn_relu_4(x)
        x = self.deconv_4(x)

        # raw alpha pred
        out = self.conv(x)

        return out 





