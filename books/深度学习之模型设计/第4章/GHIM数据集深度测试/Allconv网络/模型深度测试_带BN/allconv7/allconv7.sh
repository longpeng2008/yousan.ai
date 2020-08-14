export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
../../../../caffe/build/tools/caffe train \
	-solver "allconv7.solver" \
	-gpu 2,3,6,7 2>&1 | tee log.txt
	
