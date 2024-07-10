# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @date       : 2022-3-29
# #author     : longpeng
# @brief      : 数据集Dataset定义
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

random.seed(1)

## 数据集定义
class Dataset(Dataset):
    def __init__(self, data_dir, filelist="train.txt", transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param filelist: str, 图片路径与标签txt
        :param transform: torch.transform，数据预处理
        """
        self.filelist = filelist
        self.data_dir = data_dir
        self.image_labels = self._get_img_label()  # image_labels存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        imgpath, label = self.image_labels[index]
        img = Image.open(imgpath).convert('RGB') # 0~255

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.image_labels) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.image_labels)

    def _get_img_label(self):
        image_labels = []
        datas = open(self.filelist).readlines()
        for data in datas:
            imgpath,label = data.strip().split(' ')
            image_labels.append((os.path.join(self.data_dir,imgpath),int(label)))

        return image_labels

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_dir = os.path.join(BASE_DIR, "data")
    image_size = 256 # 图像缩放大小
    crop_size = 224 # 图像裁剪大小
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
    dset = Dataset(data_dir=path_dir, filelist="data/GHIM-20/train_shuffle.txt", transform=data_transforms['train'])
    print(dset.__len__())
    dset = Dataset(data_dir=path_dir, filelist="data/GHIM-20/val_shuffle.txt", transform=data_transforms['val'])
    print(dset.__len__())
    print(dset.__getitem__(0)[0].shape)




