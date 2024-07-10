#coding:utf8
from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable 
import models
from models import *
from dataset import Dataset
from tensorboardX import SummaryWriter

## 训练配置参数
parser = argparse.ArgumentParser(description='PyTorch KD simpleconv3 training')
parser.add_argument('--inplanes', type=int, default=32, metavar='N',
                    help='input planes (default: 32)')
parser.add_argument('--kernel', type=int, default=3, metavar='N',
                    help='the kernel size for training (default: 3)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=int, default=30, metavar='N',
                    help='stepsize to adjust lr (default: 30)')
parser.add_argument('--lamda', type=float, default=0.5,
                    help='KL loss weight (default: 0.5)') ##损失权重系数
parser.add_argument('--T', type=float, default=5.0,
                    help='knowledge distillation temperature (default: 5)') ##蒸馏温度系数
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--tmodelpath', type=str, default='models/tmodel.pth', 
                    help='teacher model path')
parser.add_argument('--save', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save student model (default: current directory)')
parser.add_argument('--logdir', default='log-simple', type=str, metavar='M',
                    help='the logdir (default: log-simple)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

writer = SummaryWriter(args.logdir)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

## 训练配置
image_size = 256 ##图像缩放大小
crop_size = 224 ##图像裁剪大小
nclass = 20 ##类别
## 载入训练好的teacher模型
print("=> loading teacher model checkpoint ".format(args.tmodelpath))
tmodel = torch.load(args.tmodelpath)
tmodel.eval()
inplanes = args.inplanes # 输入通道数
kernel = args.kernel # 卷积核大小
smodel = Simpleconv5(nclass=nclass,inplanes=inplanes,kernel=kernel)

if args.cuda:
    tmodel.cuda()
    smodel.cuda()

## 数据读取与预处理方法
data_dir = './data'
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(crop_size,scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val': transforms.Compose([
            transforms.Scale(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
    }

# 构建Dataset实例
train_data = Dataset(data_dir=data_dir, filelist="data/GHIM-20/train_shuffle.txt", transform=data_transforms['train'])
val_data = Dataset(data_dir=data_dir, filelist="data/GHIM-20/val_shuffle.txt", transform=data_transforms['val'])
print('num of train data:',train_data.__len__())
print('num of val data:',val_data.__len__())

## 创建数据指针，设置batch大小，shuffle，多进程数量
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=4,shuffle=False,num_workers=1)
dataloaders = {}
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader

## 优化方法
optimizer = optim.SGD(smodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.2) ## 每step_size个epoch，学习率衰减
kd_fun = nn.KLDivLoss(reduce=True)

for epoch in range(1,args.epochs+1):
    print("epoch ",epoch)
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            smodel.train(True)
        else:
            smodel.train(False)

        running_loss_clc = 0.0
        running_loss_kd = 0.0
        running_loss = 0.0
        running_acc = 0.0
        num_batch = 0

        for data, target in dataloaders[phase]:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if phase == 'train':
               optimizer.zero_grad()
            output_s = smodel(data)
            output_t = tmodel(data)
            _, preds = torch.max(output_s.data, 1)

            s_max = F.log_softmax(output_s / args.T, dim=1)
            t_max = F.softmax(output_t / args.T, dim=1)
            batch_size = target.shape[0]
            loss_kd = kd_fun(s_max, t_max) ## KL散度,实现为logy-x，输入第一项必须是对数形式
            loss_clc = F.cross_entropy(output_s, target) ##分类loss
            loss = (1 - args.lamda) * loss_clc + args.lamda * args.T * args.T * loss_kd

            running_acc += torch.sum(preds == target).item()
            running_loss += loss.data.item()
            running_loss_clc += loss_clc.data.item()
            running_loss_kd += loss_kd.data.item()
            num_batch += 1

            if phase == 'train':
                loss.backward()
                optimizer.step()

        epoch_loss = running_loss / num_batch
        epoch_loss_clc = running_loss_clc / num_batch
        epoch_loss_kd = running_loss_kd / num_batch
        epoch_acc = running_acc / len(dataloaders[phase].dataset)

        if phase == 'train':
            writer.add_scalar('data/trainloss', epoch_loss, epoch)
            writer.add_scalar('data/trainloss_clc', epoch_loss_clc, epoch)
            writer.add_scalar('data/trainloss_kd', epoch_loss_kd, epoch)
            writer.add_scalar('data/trainacc', epoch_acc, epoch)
            print('\nTrain set: Epoch: {}, Average loss: {:.4f}'.format(epoch,epoch_loss))
        else:
            writer.add_scalar('data/testloss', epoch_loss, epoch)
            writer.add_scalar('data/testloss_clc', epoch_loss_clc, epoch)
            writer.add_scalar('data/testloss_kd', epoch_loss_kd, epoch)
            writer.add_scalar('data/testacc', epoch_acc, epoch)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f})\n'.format(
        epoch_loss, running_acc, len(dataloaders[phase].dataset),epoch_acc))

        if epoch % 10 == 0:
            torch.save(smodel, os.path.join(args.save,'model_epoch_{}.pth'.format(epoch)))

writer.close()
