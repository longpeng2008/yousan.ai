export LD_LIBRARY_PATH=/home/longpeng/projects/portrait_segmentation/original/caffe/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

SOLVER=./mobilenet_solver.prototxt
WEIGHTS=./mobilenet.caffemodel
/home/longpeng/projects/6_hair/caffe/build/tools/caffe train -solver $SOLVER -weights $WEIGHTS -gpu 0,1 2>&1 | tee log.txt
