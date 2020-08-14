/home/long.peng/opts/Caffe_Long/build/tools/caffe train \
	-solver "solver.prototxt" \
        -weights "nin__iter_40000.caffemodel" \
	-gpu 4,5,6,7 2>&1 | tee log.txt
	
