../../../caffe/build/tools/caffe train \
	-solver "allconv6.solver" \
	-gpu 0,1,2,3 2>&1 | tee log.txt
	
