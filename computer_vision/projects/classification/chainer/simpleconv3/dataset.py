#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
import glob
from chainer.datasets import tuple_dataset
class Dataset():
    def __init__(self, path, width=60, height=60):
        channels = 3
        path = glob.glob('./mouth/*')
        pathsAndLabels = []
        index = 0
        for p in path:
            print(p + "," + str(index))
            pathsAndLabels.append(np.asarray([p, index]))
            index = index + 1
        allData = []
        for pathAndLabel in pathsAndLabels:
            path = pathAndLabel[0]
            label = pathAndLabel[1]
            imagelist = glob.glob(path + "/*")
            for imgName in imagelist:
                allData.append([imgName, label])
        allData = np.random.permutation(allData)
        imageData = []
        labelData = []

        for pathAndLabel in allData:
            #print(pathAndLabel[0])
            img = Image.open(pathAndLabel[0])
            img = img.resize((width, height))
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))

            threshold = np.int32(len(imageData)/8*7)
            self.train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
            self.test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
    def get_dataset(self):
        return self.train, self.test
a= Dataset("mouth")
a.get_dataset()

