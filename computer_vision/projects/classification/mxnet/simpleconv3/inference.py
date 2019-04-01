#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
#coding=utf8
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import json
from common import find_mxnet
import mxnet as mx
import random
import importlib
def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

epoch = int(sys.argv[1]) #check point step
gpu_id = int(sys.argv[2]) #GPU ID for infer
prefix = sys.argv[3]
imgdir = sys.argv[4]
ctx = mx.gpu(gpu_id)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
#mx.viz.plot_network(sym).view()

sym  = mx.symbol.SoftmaxOutput(data = sym, name = 'softmax')

imgs = os.listdir(imgdir)
for imgname in imgs:
        img_full_name = os.path.join(imgdir,imgname)
        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img = np.float32(img)
        rows, cols = img.shape[:2]

        resize_width = 48
	resize_height = 48
        
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
        h, w, _ = img.shape

        img_crop = img[0:h, 0:w]
        img_crop = np.swapaxes(img_crop, 0, 2)
        img_crop = np.swapaxes(img_crop, 1, 2)  # change to r,g,b order
        img_crop = img_crop[np.newaxis, :]

        arg_params["data"] = mx.nd.array(img_crop, ctx)
        arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        probs = exe.outputs[0].asnumpy()
        print "test image",imgname,probs

