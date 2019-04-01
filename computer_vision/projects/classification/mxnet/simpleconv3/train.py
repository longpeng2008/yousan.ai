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
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit
import mxnet as mx

import os, urllib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simple conv3 net",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 1)
    parser.set_defaults(image_shape='3,48,48', num_epochs=1000,
                        lr=.001, wd=0)
    args = parser.parse_args()

    # define simpleconv3
    net = mx.sym.load('models/simple-conv3-symbol.json')
    print "net",net

    # train
    fit.fit(args        = args,
            network     = net,
            data_loader = data.get_rec_iter)
