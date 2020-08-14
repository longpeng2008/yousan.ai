i=0
dir=$1
resultdir=$2
app=$3
for file in $dir""* 
do 
	arr=$(echo $file | tr "/" "\n")
	for x in $arr
	do
		filename=$x
	done

	brr=$(echo $filename | tr "." "\n")
	brrs=( $brr )
	fileid=${brrs[0]}

	num=${#brrs[@]}  
	index=$(expr $num - 1)
	fileformat=${brrs[index]}
	
	echo file=""$file
	echo fileid=""$fileid
	echo fileformat=""$fileformat

	if [ $fileformat == jpeg -o $fileformat == png -o $fileformat == jpg -o $fileformat == bmp ] ;
    	then
        #echo "good"
		i=$(expr $i + 1)
		resultfile=$resultdir""$i""$app"".$fileformat
		echo file=""$file"",resultfile=""$resultfile
		mv "$file" "$resultfile"
	else
		echo $file""not good
    fi
done

echo 执行删除""$dir""*
#rm $dir""*
echo 执行mv""$resultdir""*
mv $resultdir""* $dir
