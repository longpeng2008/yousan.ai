export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
/home/longpeng/project/Books/segmentation/Caffe_Long/build/tools/caffe train \
	-solver "allconv5.solver" \
	-gpu 0 2>&1 | tee log.txt
	
