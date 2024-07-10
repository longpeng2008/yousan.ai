#coding:utf8
import torch
from torch import nn

class Simpleconv5(nn.Module):
    def __init__(self,nclass=2,inplanes=32,kernel=3):
        super(Simpleconv5, self).__init__()
        self.inplanes = inplanes
        self.kernel = kernel
        self.pad = self.kernel // 2
        ## 卷积模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=self.kernel, stride=2, padding=self.pad),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=self.kernel, stride=2, padding=self.pad),
            nn.BatchNorm2d(self.inplanes*2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.inplanes*2, self.inplanes*4, kernel_size=self.kernel, stride=2, padding=self.pad),
            nn.BatchNorm2d(self.inplanes*4),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.inplanes*4, self.inplanes*8, kernel_size=self.kernel, stride=2, padding=self.pad),
            nn.BatchNorm2d(self.inplanes*8),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.inplanes*8, self.inplanes*16, kernel_size=self.kernel, stride=2, padding=self.pad),
            nn.BatchNorm2d(self.inplanes*16),
            nn.ReLU(True),
        )

        self.classifier = nn.Linear(self.inplanes*16, nclass)

    def forward(self, x):       
        out = self.conv1(x)     
        out = self.conv2(out)     
        out = self.conv3(out)   
        out = self.conv4(out)
        out = self.conv5(out)
        out = nn.AvgPool2d(7)(out)
        out = out.view(out.size(0), -1) 
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    img = torch.randn(1, 3, 224 ,224)
    net = Simpleconv5(nclass=20,inplanes=12,kernel=3)
    sample = net(img)
    print(sample.shape)
