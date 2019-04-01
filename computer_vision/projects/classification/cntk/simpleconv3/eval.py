#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
from simpleconv3 import simpleconv3
import matplotlib.pyplot as plt
import os
import sys
import cv2
import numpy as np
import cntk as C
from PIL import Image
from dataset import *
model_file = sys.argv[1]
image_list = sys.argv[2]
model = C.load_model(model_file)

count = 0
acc = 0
imagepaths = open(image_list,'r').readlines()
for imagepath in imagepaths:
    imagepath,label = imagepath.strip().split('\t')
    im = Image.open(imagepath)
    print imagepath
    print "im size",im.size
    image_data = np.array(im,dtype=np.float32)
    image_data = cv2.resize(image_data,(image_width,image_height))
    image_data = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
    output = model.eval({model.arguments[0]:[image_data]})[0]
    print output
    print label,np.argmax(np.squeeze(output))
    if str(label) == str(np.argmax(np.squeeze(output))):
        acc = acc + 1
    count = count + 1
print "acc=",float(acc) / float(count)
