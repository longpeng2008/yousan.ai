../../../caffe/build/tools/caffe train \
	-solver "allconv7.solver" \
	-gpu 1,2,3,4 2>&1 | tee log.txt
	
