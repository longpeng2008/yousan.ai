#########################################################################
# Author: longpeng2008to2012@gmail.com
#########################################################################
SOLVER=./solver.prototxt
caffe/build/tools/caffe train -solver $SOLVER -gpu 0 2>&1 | tee log.txt 
#/home/longpeng/opts/1_Caffe_Long/build/tools/caffe train -solver $SOLVER -gpu 0 2>&1 | tee log.txt 
