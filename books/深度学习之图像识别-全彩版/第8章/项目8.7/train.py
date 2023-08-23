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
from torchvision import datasets, transforms
from torch.autograd import Variable 
import models
from models import *

## 训练配置参数
parser = argparse.ArgumentParser(description='PyTorch KD simpleconv3 training')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lamda', type=float, default=0.5,
                    help='KL loss weight (default: 0.5)') ##损失权重系数
parser.add_argument('--T', type=float, default=5.0,
                    help='knowledge distillation temperature (default: 5)') ##蒸馏温度系数
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--tmodelpath', type=str, default='models/tmodel.pth.tar', 
                    help='teacher model path')
parser.add_argument('--save', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save student model (default: current directory)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

## 训练配置
image_size = 60 ##图像缩放大小
crop_size = 48 ##图像裁剪大小
nclass = 4 ##类别
tmodel = simpleconv3(nclass)
tmodel.eval()
smodel = simpleconv3small(nclass)

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

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=64,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'val']}
train_loader = dataloaders['train']
test_loader = dataloaders['val']

## 优化方法
optimizer = optim.SGD(smodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
kd_fun = nn.KLDivLoss(reduce=True)

## 载入训练好的teacher模型
print("=> loading teacher model checkpoint ".format(args.tmodelpath))
tcheckpoint = torch.load(args.tmodelpath,map_location=lambda storage,loc: storage)
tmodel.load_state_dict(tcheckpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(args.tmodelpath, tcheckpoint['epoch'], tcheckpoint['best_prec1']))
print("opti="+str(tcheckpoint['optimizer']['param_groups']))

## 训练函数
from tensorboardX import SummaryWriter
writer = SummaryWriter(args.save)
def process(epoch,data_loader,istrain=True):
    if istrain:
        smodel.train()
    else:
        smodel.eval()

    running_loss_clc = 0.0
    running_loss_kd = 0.0
    running_loss = 0.0
    running_acc = 0.0
    num_batch = 0

    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if istrain:
           optimizer.zero_grad()
        output_s = smodel(data)
        output_t = tmodel(data)
        _, preds = torch.max(output_s.data, 1)

        s_max = F.log_softmax(output_s / args.T, dim=1)
        t_max = F.softmax(output_t / args.T, dim=1)
        batch_size = target.shape[0]
        loss_kd = kd_fun(s_max, t_max) ##KL散度,实现为logy-x，输入第一项必须是对数形式
        #loss_kd = kd_fun(s_max, t_max) / batch_size ##KL散度,实现位logy-x，输入第一项必须是对数形式
        loss_clc = F.cross_entropy(output_s, target) ##分类loss
        loss = (1 - args.lamda) * loss_clc + args.lamda * args.T * args.T * loss_kd

        running_acc += torch.sum(preds == target).item()
        running_loss += loss.data.item()
        running_loss_clc += loss_clc.data.item()
        running_loss_kd += loss_kd.data.item()
        num_batch += 1

        if istrain:
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / num_batch
    epoch_loss_clc = running_loss_clc / num_batch
    epoch_loss_kd = running_loss_kd / num_batch
    epoch_acc = running_acc / len(data_loader.dataset)
    if istrain == True:
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
        epoch_loss, running_acc, len(data_loader.dataset),epoch_acc))

    return epoch_acc

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

## 训练
best_prec1 = 0.
print("args.start_epoch="+str(args.start_epoch))

for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]: ##学习率变更，50%，75%的2次epoch，学习率乘以0.1，param_groups包括[{'params','lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'},{……}]
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    prec_train = process(epoch,train_loader,istrain=True)
    prec1 = process(epoch,test_loader,istrain=False)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': smodel.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
