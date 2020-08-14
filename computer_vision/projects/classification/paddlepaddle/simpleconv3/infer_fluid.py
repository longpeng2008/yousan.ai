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
import os
import cv2
# coding=utf-8
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.v2 as paddle
from paddle.fluid.initializer import NormalInitializer
from paddle.fluid.param_attr import ParamAttr
from visualdl import LogWriter
from net_fluid import simplenet

if __name__ == "__main__":
    # 开始预测
    type_size = 2
    testsize = 48

    imagedir = sys.argv[1]
    images = os.listdir(imagedir)
        
    save_dirname = "./models/299"
    exe = fluid.Executor(fluid.CPUPlace())
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program,feed_target_names,fetch_targets] = fluid.io.load_inference_model(save_dirname,exe)

        predicts = np.zeros((type_size,1))
        for image in images:
            imagepath = os.path.join(imagedir,image)
            img = paddle.image.load_image(imagepath)
            img = paddle.image.simple_transform(img,testsize,testsize,False)
            img = img[np.newaxis,:]

            #print img.shape

            results = np.argsort(-exe.run(inference_program,feed={feed_target_names[0]:img},
                    fetch_list=fetch_targets)[0])
            label = results[0][0]
            predicts[label] += 1
    
    print predicts
