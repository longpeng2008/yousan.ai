## 年龄编辑
# for scale in -0.1 -0.05 -0.03 -0.01 0.01 0.03 0.05 0.1
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于w的重建/projects1024_new/10000/ --size 1024 --direction latent_directions/agew.npy --directionscale $scale --results projects/基于w的重建/projects1024_new/10000/edit_age --zlatent 0
# done

## 表情编辑
# for scale in -0.05 -0.03 0.03 0.05
#do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于w的重建/projects1024_new/10000/ --size 1024 --direction latent_directions/smilew.npy --directionscale $scale --results projects/基于w的重建/projects1024_new/10000/edit_smile --zlatent 0
# done

## 性别编辑
# for scale in -0.1 -0.05 0.05 0.1
# do
	#python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于w的重建/projects1024_new/10000/  --size 1024 --direction latent_directions/genderw.npy --directionscale $scale --results projects/基于w的重建/projects1024_new/10000/edit_gender --zlatent 0
# done

