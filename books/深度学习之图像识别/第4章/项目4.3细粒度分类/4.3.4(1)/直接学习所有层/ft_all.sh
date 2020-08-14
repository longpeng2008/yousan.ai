#!/bin/bash

# first fine all layers

../bilinear_cub/caffe-20160312/build/tools/caffe train \
	-model "ft_all.prototxt" \
	-solver "ft_all.solver" \
	-weights "VGG_ILSVRC_16_layers.caffemodel" \
	-gpu 5 2>&1 | tee log.txt
	
