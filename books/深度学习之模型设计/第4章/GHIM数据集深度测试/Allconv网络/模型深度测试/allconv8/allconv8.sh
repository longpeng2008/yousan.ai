../../../../caffe/build/tools/caffe train \
	-solver "allconv8.solver" \
	-gpu 4,5,6,7 2>&1 | tee log.txt
	
