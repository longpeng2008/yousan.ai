#!/bin/bash

# first fine tune the last layer only

../bilinear_cub/caffe-20160312/build/tools/caffe train \
	-model "ft_last_layer.prototxt" \
	-solver "ft_last_layer.solver" \
	-weights "VGG_ILSVRC_16_layers.caffemodel" \
	-gpu 1 2>&1 | tee log.txt
	
