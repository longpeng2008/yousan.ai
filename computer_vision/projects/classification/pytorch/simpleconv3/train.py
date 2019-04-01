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
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from net import simpleconv3
from tensorboardX import SummaryWriter

writer = SummaryWriter()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
           
            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    return model

if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(48),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val': transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
    }

    data_dir = './da/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=16,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()
    
    modelclc = simpleconv3()
    print(modelclc)
    if use_gpu:
        modelclc = modelclc.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(modelclc.parameters(), lr=0.1, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    modelclc = train_model(model=modelclc,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=500)
    
    os.mkdir('models')
    torch.save(modelclc.state_dict(),'models/model.ckpt')
