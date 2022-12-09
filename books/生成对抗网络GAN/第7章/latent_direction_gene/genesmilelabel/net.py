import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        #print "bn1 shape",x.shape
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 128 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    from visualize import  make_dot
    x = Variable(torch.randn(1,3,48,48))
    model = simpleconv3(4)
    y = model(x)
    g = make_dot(y)
    g.view()
