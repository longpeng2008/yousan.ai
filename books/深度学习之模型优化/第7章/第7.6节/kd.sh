#python3 kd.py --lamda 0.5 --T 3 --inplanes 12 --kernel 3 --save checkpoint_0.5_3 --tmodelpath simple32_5/model_epoch_140.pth  --logdir log_kd_0.5_3 --epochs 150 --step_size 40

python3 kd.py --lamda 0.5 --T 3 --inplanes 32 --kernel 5 --save vgg_0.5_3_simple32 --tmodelpath vgg16/model_epoch_20.pth --logdir log_kd_0.5_3_vgg --epochs 150 --step_size 40

