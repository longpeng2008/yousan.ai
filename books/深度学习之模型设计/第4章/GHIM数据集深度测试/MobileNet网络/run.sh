#!/bin/bash

# first fine tune the last layer only

../../../../caffe/build/tools/caffe train \
	-model "mobilenet_train.prototxt" \
	-solver "mobilenet_solver.prototxt" \
	-weights "mobilenet.caffemodel" \
	-gpu 1,2,3 2>&1 | tee log_mobilenet.txt
	
