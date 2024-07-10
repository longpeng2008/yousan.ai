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
parser = argparse.ArgumentParser(description='PyTorch simpleconv3 training')
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
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--model', default='simple', type=str, metavar='M',
                    help='the network (default: simpleconv5)')
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
image_size = 256 # 图像缩放大小
crop_size = 224 # 图像裁剪大小
nclass = 20 # 类别
inplanes = args.inplanes # 输入通道数
kernel = args.kernel # 卷积核大小
if args.model == 'simple':
    model = Simpleconv5(nclass=nclass,inplanes=inplanes,kernel=kernel)
else:
    model = torchvision.models.vgg16_bn(pretrained=True).cuda()

    # for param in model.parameters():
        # param.requires_grad = False
    # model.classifier[6] = torch.nn.Sequential(torch.nn.Linear(4096,20))
    # for param in model.classifier[6].parameters():
        # param.requires_grad = True

if args.cuda:
    model.cuda()

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
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.2) ## 每step_size个epoch，学习率衰减

for epoch in range(1,args.epochs+1):
    print("epoch ",epoch)
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0
        running_acc = 0.0

        n = 0
        for data in dataloaders[phase]:
            imgs, labels = data
            imgs, labels = imgs.to(device).float(), labels.to(device)
            output = model(imgs)
            _, preds = torch.max(output.data, 1)
            loss = F.cross_entropy(output, labels)
            running_loss += loss.data.item()
            running_acc += (torch.sum(preds == labels).item())

            n += 1

            optimizer.zero_grad()
            if phase == 'train':
            # 梯度置0，反向传播，参数更新
                loss.backward()
                optimizer.step()

        epoch_loss = running_loss / n
        
        if phase == 'train':
            epoch_acc = running_acc / train_data.__len__()
            writer.add_scalar('data/trainloss', epoch_loss, epoch)
            writer.add_scalar('data/trainacc', epoch_acc, epoch)
            print('train loss='+str(epoch_loss))
            print('train acc='+str(epoch_acc))
        else:
            epoch_acc = running_acc / val_data.__len__()
            writer.add_scalar('data/valloss', epoch_loss, epoch)
            writer.add_scalar('data/valacc', epoch_acc, epoch)
            print('val loss='+str(epoch_loss))
            print('val acc='+str(epoch_acc))

    if epoch % 10 == 0:
        torch.save(model, os.path.join(args.save,'model_epoch_{}.pth'.format(epoch)))

writer.export_scalars_to_json('./all_scalars_{}.json'.format(args.model))
writer.close()
