## 基于向量Z的表情编辑
# for scale in -0.5 -0.3 -0.1 0.1 0.3 0.5
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/z/  --size 1024 --direction latent_directions/smilez.npy --directionscale $scale --results projects/基于z的重建/projects1024_new/10000/z/edit_smile --zlatent 1
# done

## 基于向量W的表情编辑
# for scale in -0.05 -0.03 0.03 0.05
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/w/  --size 1024 --direction latent_directions/smilew.npy --directionscale $scale --results projects/基于z的重建/projects1024_new/10000/w/edit_smile --zlatent 0
# done

## 基于向量Z的年龄编辑
# for scale in -0.5 -0.3 -0.1 0.1 0.3 0.5
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/z/  --size 1024 --direction latent_directions/agez.npy --directionscale $scale --results projects/基于z的重建/projects1024_new/10000/z/edit_age --zlatent 1
# done

## 基于向量W的年龄编辑
# for scale in -0.03 -0.01 0.01 0.03
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/w/  --size 1024 --direction latent_directions/agew.npy --directionscale $scale --results projects/基于z的重建/projects1024_new/10000/w/edit_age --zlatent 0
# done

## 基于向量Z的性别编辑
# for scale in -0.5 -0.3 0.3 0.5
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/z/  --size 1024 --direction latent_directions/genderz.npy --directionscale $scale --results projects/基于z的重建/projects1024_new/10000/z/edit_gender --zlatent 1
# done

## 基于向量W的性别编辑
# for scale in -0.1 -0.05 0.05 0.1
# do
	# python3 testeditsingle.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/w/ --size 1024 --direction latent_directions/genderw.npy --directionscale $scale --results projects/基于z的重建/projects1024_new/10000/w/edit_gender --zlatent 0
# done
