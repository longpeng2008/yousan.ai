#!/bin/bash

# first fine all layers

../bilinear_cub/caffe-20160312/build/tools/caffe train \
	-model "ft_all.prototxt" \
	-solver "ft_all.solver" \
	-weights "ft_last_layer_iter_60000.caffemodel" \
	-gpu 0 2>&1 | tee log.txt
	
