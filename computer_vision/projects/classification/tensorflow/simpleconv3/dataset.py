#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import numpy as np

class ImageData:
    def read_txt_file(self):
        self.img_paths = []
        self.labels = []
        for line in open(self.txt_file, 'r'):
            items = line.split(' ')
            self.img_paths.append(items[0])
            self.labels.append(int(items[1]))

    def __init__(self, txt_file, batch_size, num_classes,
                 image_size,buffer_scale=100):
        self.image_size = image_size
        self.batch_size = batch_size
        self.txt_file = txt_file ##txt list file,stored as: imagename id
        self.num_classes = num_classes
        buffer_size = batch_size * buffer_scale

        # 读取图片
        self.read_txt_file()
        self.dataset_size = len(self.labels) 
        print "num of train datas=",self.dataset_size
        # 转换成Tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # 创建数据集
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        print "data type=",type(data)
        data = data.map(self.parse_function)
        data = data.repeat(1000)
        data = data.shuffle(buffer_size=buffer_size)

        # 设置self data Batch
        self.data = data.batch(batch_size)
        print "self.data type=",type(self.data)
    
    def augment_dataset(self,image,size):
        distorted_image = tf.image.random_brightness(image,
                                               max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
        return float_image
    
    def parse_function(self, filename, label):
        label_ = tf.one_hot(label, self.num_classes)
        img = tf.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype = tf.float32)
        img = tf.random_crop(img,[self.image_size[0],self.image_size[1],3])
        img = tf.image.random_flip_left_right(img)
        img = self.augment_dataset(img,self.image_size)
        return img, label_


