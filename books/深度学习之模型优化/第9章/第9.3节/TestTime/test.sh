for imgdir in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  for image in ../GHIM-20/""$imgdir""/*.jpg
  do
  	arr=$(echo $image | tr "/" "\n")
  	for x in $arr
  	do
  		filename=$x
  	done
  
  	brr=$(echo $filename | tr "." "\n")
  	brrs=( $brr )
  	fileid=${brrs[0]}
  	#./simpleconv5 ../models/simpleconv5_int8.param ../models/simpleconv5_int8.bin $image results_int8/""$fileid"".jpg
  	./simpleconv5 ../models/simpleconv5.param ../models/simpleconv5.bin $image results/""$fileid"".jpg
  done
done
