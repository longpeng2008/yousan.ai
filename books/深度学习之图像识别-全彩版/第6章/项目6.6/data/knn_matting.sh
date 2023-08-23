
ROOT=./

for line in $(cat ./train_list.txt)
do 
    echo "Getting alpha for image : ${line} ..."

    python3 knn_matting.py --dataroot=$ROOT/ --img=${line}
done

