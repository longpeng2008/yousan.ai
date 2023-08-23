for image in ../datas/*.jpg
do
	arr=$(echo $image | tr "/" "\n")
	for x in $arr
	do
		filename=$x
	done

	brr=$(echo $filename | tr "." "\n")
	brrs=( $brr )
	fileid=${brrs[0]}
	./simpleconv3 caffe_int8.param caffe_int8.bin $image result/""$fileid"".jpg
	#./simpleconv3 caffe.param caffe.bin $image result/""$fileid"".jpg
done
