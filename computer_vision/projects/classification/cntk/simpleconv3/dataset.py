#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import os
import sys
import cv2
import numpy as np
import cntk as C
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

image_height = 60
image_width  = 60
num_channels = 3
num_classes  = 2

import cntk.io.transforms as transform 
def create_reader(map_file,train):
    print("Reading map file:", map_file)

    if not os.path.exists(map_file):
        raise RuntimeError("no training file.")

    # 定义好transform
    transforms = []
    if train:
        transforms += [
            transform.crop(crop_type='randomside', side_ratio=0.8) 
        ]
    transforms += [
        transform.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
    ]
    # 解析
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features = C.io.StreamDef(field='image', transforms=transforms), 
        labels   = C.io.StreamDef(field='label', shape=num_classes)
    )))

# Create the train and test readers
reader_train = create_reader('./train.txt',True)
reader_test = create_reader('./val.txt',False)

    
