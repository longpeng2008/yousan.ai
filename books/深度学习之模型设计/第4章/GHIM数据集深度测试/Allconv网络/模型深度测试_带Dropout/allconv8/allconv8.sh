/root/anaconda3/envs/py2cv3/bin/caffe train \
	-solver "allconv8.solver" \
	-gpu 0 2>&1 | tee log.txt
	
