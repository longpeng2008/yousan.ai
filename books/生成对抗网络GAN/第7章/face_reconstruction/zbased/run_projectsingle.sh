mkdir results/w
for i in 1 2
do
    python3 projector.py --ckpt checkpoints/stylegan-1024px-new.model --files single_reconstruct/images/""$i --step 1000 --results single_reconstruct/projects1024/1000 --noise_regularize 1000 --size 1024 > log""$i""_1000.txt
done
