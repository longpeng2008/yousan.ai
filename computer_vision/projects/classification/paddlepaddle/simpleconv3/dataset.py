#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
from multiprocessing import cpu_count
import paddle.v2 as paddle

class Dataset:
    def __init__(self,cropsize,resizesize):
        self.cropsize = cropsize
        self.resizesize = resizesize

    def train_mapper(self,sample):
        img, label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img, self.resizesize, self.cropsize, True)
        #print "train_mapper",img.shape,label
        return img.flatten().astype('float32'), label

    def test_mapper(self,sample):
        img, label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img, self.resizesize, self.cropsize, False)
        #print "test_mapper",img.shape,label
        return img.flatten().astype('float32'), label

    def train_reader(self,train_list, buffered_size=1024):
        def reader():
            with open(train_list, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                #print "len of train dataset=",len(lines)
                for line in lines:
                    img_path, lab = line.strip().split(' ')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.train_mapper, reader,
                                          cpu_count(), buffered_size)

    def test_reader(self,test_list, buffered_size=1024):
        def reader():
            with open(test_list, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                print "len of val dataset=",len(lines)
                for line in lines:
                    img_path, lab = line.strip().split(' ')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.test_mapper, reader,
                                          cpu_count(), buffered_size)

