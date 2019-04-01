#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================

import mxnet as mx

def get_symbol(num_classes, **kwargs):
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=12, kernel=(3,3), stride=(2,2), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv1_scale = conv1_bn
    relu1 = mx.symbol.Activation(name='relu1', data=conv1_scale , act_type='relu')
    
    conv2 = mx.symbol.Convolution(name='conv2', data=relu1 , num_filter=24, kernel=(3,3), stride=(2,2), no_bias=True)
    conv2_bn = mx.symbol.BatchNorm(name='conv2_bn', data=conv2 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv2_scale = conv2_bn
    relu2 = mx.symbol.Activation(name='relu2', data=conv2_scale , act_type='relu')

    conv3 = mx.symbol.Convolution(name='conv3', data=relu2 , num_filter=48, kernel=(3,3), stride=(2,2), no_bias=True)
    conv3_bn = mx.symbol.BatchNorm(name='conv3_bn', data=conv3 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
    conv3_scale = conv3_bn
    relu3 = mx.symbol.Activation(name='relu3', data=conv3_scale , act_type='relu')

    pool = mx.symbol.Pooling(name='pool', data=relu3 , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    fc = mx.symbol.Convolution(name='fc', data=pool , num_filter=num_classes, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
    flatten = mx.symbol.Flatten(data=fc, name='flatten')
    softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')
    return softmax

if __name__ == "__main__":
    net = get_symbol(2)
    net.save('simpleconv3-symbol.json')

