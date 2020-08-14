#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import paddle.fluid as fluid


def simplenet(input):
    # 定义卷积块 
    conv1 = fluid.layers.conv2d(input=input, num_filters=12,stride=2,padding=1,filter_size=3,act="relu")
    bn1 = fluid.layers.batch_norm(input=conv1)
    conv2 = fluid.layers.conv2d(input=bn1, num_filters=24,stride=2,padding=1,filter_size=3,act="relu")
    bn2 = fluid.layers.batch_norm(input=conv2)
    conv3 = fluid.layers.conv2d(input=bn2, num_filters=48,stride=2,padding=1,filter_size=3,act="relu")
    bn3 = fluid.layers.batch_norm(input=conv3)
    fc1 = fluid.layers.fc(input=bn3, size=128, act=None)
    return fc1,conv1
