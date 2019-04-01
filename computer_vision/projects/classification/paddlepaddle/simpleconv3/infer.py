#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import sys
import numpy as np
import paddle.v2 as paddle
from PIL import Image
from net import simplenet
import cv2

class testmodel:
    def __init__(self):
        paddle.init(use_gpu=False, trainer_count=2)

    def get_parameters(self):
        with open("./snaps/model.tar", 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
        return parameters

    def get_TestData(self,resizesize,path):
        def load_images(resizesize,file):
            img = paddle.image.load_image(file)
            img = paddle.image.simple_transform(img,resizesize,resizesize,False)
            img = img.flatten().astype('float32')
            return img

        test_data = []
        test_data.append((load_images(resizesize,path),))
        return test_data

    def to_prediction(self, out, parameters, test_data):
        probs = paddle.infer(output_layer=out,
                             parameters=parameters,
                             input=test_data)
        #print probs
        lab = np.argsort(-probs)
        label = lab[0][0]
        return label


if __name__ == "__main__":
    tester = testmodel()
    # 开始预测
    type_size = 2
    cropsize = 48
    datadim = 3 * cropsize * cropsize
    out = simplenet(datadim,type_size)
    parameters = tester.get_parameters()

    imagedir = sys.argv[1]
    images = os.listdir(imagedir)

    predicts = np.zeros((type_size,1))
    for image in images:
        imagepath = os.path.join(imagedir,image)
        image = tester.get_TestData(cropsize,imagepath)
        label = tester.to_prediction(out=out, parameters=parameters, test_data=image)
        predicts[label] += 1
    
    '''
    lines = open(sys.argv[1],'r').readlines()
    for line in lines:
        line = line.strip()
        imagepath,label = line.split(' ')
        image = tester.get_TestData(cropsize,imagepath)
        label = tester.to_prediction(out=out, parameters=parameters, test_data=image)
        predicts[label] += 1
    '''
    print predicts
