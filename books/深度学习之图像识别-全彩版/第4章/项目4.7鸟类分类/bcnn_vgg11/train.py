#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
from bcnn import BCNN
from mydataset import BirdDataset

## 使用tensorboardX进行可视化
from tensorboardX import SummaryWriter
writer = SummaryWriter('logs') ## 创建一个SummaryWriter的示例，默认目录名字为runs

## 训练主函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  ## 设置为训练模式
            else:
                model.train(False)  ## 设置为验证模式

            running_loss = 0.0 ##损失变量
            running_accs = 0.0 ##精度变量
            number_batch = 0 ##

            ## 从dataloaders中获得数据
            for data in dataloaders[phase]:
                inputs, labels = data 
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad() ##清空梯度
                outputs = model(inputs) ##前向运行
                _, preds = torch.max(outputs.data, 1) ##使用max()函数对输出值进行操作，得到预测值索引
                loss = criterion(outputs, labels) ##计算损失
                if phase == 'train':
                    loss.backward() ##误差反向传播
                    optimizer.step() ##参数更新

                running_loss += loss.data.item()
                running_accs += torch.sum(preds == labels).item()
                number_batch += 1

            ## 得到每一个epoch的平均损失与精度
            epoch_loss = running_loss / number_batch
            epoch_acc = running_accs / dataset_sizes[phase]
            
            ## 收集精度和损失用于可视化
            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    writer.close()
    return model

if __name__ == '__main__':

    image_size = 256 ##缩放图像大小
    crop_size = 224 ##图像裁剪大小，即训练输入大小
    nclass = 200 ##分类类别数
    model = BCNN() ##创建模型
    data_dir = './data/CUB_200_2011/images/' ##数据目录
    
    ## 模型缓存接口
    if not os.path.exists('models'):
        os.mkdir('models')

    ## 检查GPU是否可用，如果是使用GPU，否使用CPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    print(model)

    ## 创建数据预处理函数，训练预处理包括随机裁剪缩放、随机翻转、归一化，验证预处理包括中心裁剪，归一化
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([image_size,image_size]),
            transforms.RandomSizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize([crop_size,crop_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
    }

    # 构建MyDataset实例
    train_data = BirdDataset(data_dir=data_dir, filelist="data/train_shuffle.txt", transform=data_transforms['train'])
    val_data = BirdDataset(data_dir=data_dir, filelist="data/val_shuffle.txt", transform=data_transforms['val'])
    image_datasets = {}
    image_datasets['train'] = train_data
    image_datasets['val'] = val_data

    ## 创建数据指针，设置batch大小，shuffle，多进程数量
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],batch_size=32,shuffle=True,num_workers=4)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],batch_size=4,shuffle=False,num_workers=1)

    ## 获得数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    ## 优化目标使用交叉熵，优化方法使用带动量项的SGD，学习率迭代策略为step，每隔100个epoch，变为原来的0.1倍
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    model = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=step_lr_scheduler,
                           num_epochs=300)

    torch.save(model.state_dict(),'models/model.pt')
