#python3 train.py --inplanes 32 --kernel 5 --model simple --save checkpoint-simpleconv7_32_5 --logdir log-simpleconv7_32_5 --epochs 150 --step_size 40

python3 train.py --model vgg --save checkpoint-vgg16 --logdir log-vgg16 --epochs 20 --step_size 10
