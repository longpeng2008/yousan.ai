#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import paddle.v2 as paddle

def simplenet(datadim, type_size):
    # 获取输入数据模式
    image = paddle.layer.data(name="image",
                              type=paddle.data_type.dense_vector(datadim))

    conv1 = paddle.layer.img_conv(name="conv1",input=image,filter_size=(3,3),num_channels=3,stride=(2,2),num_filters=12,padding=1)
    bn1 = paddle.layer.batch_norm(name="bn1",input=conv1)
    conv2 = paddle.layer.img_conv(name="conv2",input=bn1,filter_size=(3,3),num_channels=12,stride=(2,2),num_filters=24,padding=1)
    bn2 = paddle.layer.batch_norm(name="bn2",input=conv2)
    conv3 = paddle.layer.img_conv(name="conv3",input=bn2,filter_size=(3,3),num_channels=24,stride=(2,2),num_filters=48,padding=1)
    bn3 = paddle.layer.batch_norm(name="bn3",input=conv3)
    fc1 = paddle.layer.fc(input=bn3, size=128, act=paddle.activation.Linear())
    bn4 = paddle.layer.batch_norm(input=fc1,
                                 act=paddle.activation.Relu(),
                                 layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn4, size=2, act=paddle.activation.Linear())
    out = paddle.layer.fc(input=fc2,
                          size=type_size,
                          act=paddle.activation.Softmax())
    return out
