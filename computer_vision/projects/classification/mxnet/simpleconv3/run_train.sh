#!/usr/bin/env bash
# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
export MXNET_CPU_WORKER_NTHREADS=160
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python train.py --gpus 0 \
    --data-train data/train.txt \
    --model-prefix 'models/simple-conv3' \
    --batch-size 80 --num-classes 2 --num-examples 900 2>&1 | tee log.txt

