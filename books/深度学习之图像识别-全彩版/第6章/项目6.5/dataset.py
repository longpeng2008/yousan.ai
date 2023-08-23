#coding:utf8

# Copyright 2023 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import random
class SegDataset(Dataset):
    def __init__(self, traintxt, imagesize, cropsize,transform=None):
        self.images = []
        self.labels = []

        lines = open(traintxt,'r').readlines()
        for line in lines:
            imagepath,labelpath = line.strip().split(' ')
            self.images.append(imagepath)
            self.labels.append(labelpath)

        self.imagesize = imagesize
        self.cropsize = cropsize

        assert len(self.images) == len(self.labels)
        self.transform  = transform
        self.samples = []
        for i in range(len(self.images)):
            self.samples.append((self.images[i],self.labels[i]))

    def __getitem__(self, item):
        img_path, label_path = self.samples[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.imagesize,self.imagesize),interpolation=cv2.INTER_NEAREST)
        label = cv2.imread(label_path, 0)
        label = cv2.resize(label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)

        randoffsetx = np.random.randint(self.imagesize - self.cropsize)
        randoffsety = np.random.randint(self.imagesize - self.cropsize)

        img = img[randoffsety:randoffsety + self.cropsize, randoffsetx:randoffsetx + self.cropsize]
        label = label[randoffsety:randoffsety + self.cropsize, randoffsetx:randoffsetx + self.cropsize]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    img = cv2.imread('test.png', 0)
    img = cv2.resize(img, (16, 16))
    hot = np.eye(2)[img]
    print(hot.transpose(2, 0, 1).shape)
