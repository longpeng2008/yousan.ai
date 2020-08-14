./caffe/build/tools/caffe train \
	-solver "allconv7.solver" \
	-gpu 0 2>&1 | tee log.txt
	
