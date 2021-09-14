#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from net import Generator
from net import Discriminator 
import os, sys
import shutil   

from tensorboardX import SummaryWriter
writer = SummaryWriter('logs') ## 创建一个SummaryWriter的示例，默认目录名字为runs

if os.path.exists("out"):
    print("移除现有 out 文件夹！")
    if sys.platform.startswith("win"):
        shutil.rmtree("./out")
    else:
        os.system("rm -r ./out")          

print("创建 out 文件夹！")
os.mkdir("./out")

# 设置一个随机种子，方便进行可重复性实验
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

## 基本参数配置
# 数据集所在路径
dataroot = "mouth/"
# 数据加载的进程数
workers = 0
# Batch size 大小
batch_size = 64
# 图片大小
image_size = 64
# 图片的通道数
nc = 3
# 噪声向量维度
nz = 100
# 生成器特征图单位
ngf = 64
# 判别器特征图单位
ndf = 64

# 损失函数
criterion = nn.BCELoss()
# 真假标签
real_label = 1.0
fake_label = 0.0
# 是否使用GPU训练
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# 创建生成器与判别器
netG = Generator().to(device)
netD = Discriminator().to(device)
# G和D的优化器，使用Adam
# Adam学习率与动量参数
lr = 0.0003
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 缓存生成结果
img_list = []
# 损失变量
G_losses = []
D_losses = []
iters = 0

## 读取数据
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 多GPU训练
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 总epochs
num_epochs = 100
## 模型缓存接口
if not os.path.exists('models'):
    os.mkdir('models')
print("Starting Training Loop...")
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
for epoch in range(num_epochs):
    lossG = 0.0
    lossD = 0.0
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 训练真实图片
        netD.zero_grad()
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_data).view(-1)
        # 计算真实图片损失，梯度反向传播
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## 训练生成图片
        # 产生latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 使用G生成图片
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        # 计算生成图片损失，梯度反向传播
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # 累加误差，参数更新
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 给生成图赋标签
        # 对生成图再进行一次判别
        output = netD(fake).view(-1)
        # 计算生成图片损失，梯度反向传播
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 存储损失
        lossG = lossG + errG.item()
        lossD = lossD + errD.item()

        # 对固定的噪声向量，存储生成的结果
        if (iters % 20 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):

            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()

            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            i = vutils.make_grid(fake, padding=2, normalize=True)
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(np.transpose(i, (1, 2, 0)))
            plt.axis('off')  # 关闭坐标轴
            plt.savefig("out/%d_%d.png" % (epoch, iters))
            plt.close(fig)
        iters += 1

    writer.add_scalar('data/lossG', lossG, epoch)
    writer.add_scalar('data/lossD', lossD, epoch)

torch.save(netG.state_dict(),'models/netG.pt')





