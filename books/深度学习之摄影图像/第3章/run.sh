SOLVER=./mobilenet_solver.prototxt
../1_Caffe_Long/build/tools/caffe train -weights pretrained_model.caffemodel -solver $SOLVER -gpu 0 2>&1 | tee log.txt
