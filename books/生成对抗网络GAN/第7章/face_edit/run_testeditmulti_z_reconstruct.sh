for scale in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python3 testeditmulti.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/w/ --results 基于z的重建/projects1024_new/10000/w --size 1024 --image1 projects/基于z的重建/projects1024_new/10000/w/star.npy --image2 projects/基于z的重建/projects1024_new/10000/w/edit_smile/yellow_smilew_-0.03.npy --image3 projects/基于z的重建/projects1024_new/10000/w/edit_smile/yellow_smilew_0.03.npy --lamda $scale --zlatent 0

    python3 testeditmulti.py --ckpt checkpoints/stylegan-1024px-new.model --files projects/基于z的重建/projects1024_new/10000/w/ --results 基于z的重建/projects1024_new/10000/w --size 1024 --image1 projects/基于z的重建/projects1024_new/10000/w/woman1.npy --image2 projects/基于z的重建/projects1024_new/10000/w/edit_smile/yellow_smilew_0.03.npy --image3 projects/基于z的重建/projects1024_new/10000/w/edit_smile/yellow_smilew_-0.03.npy --lamda $scale --zlatent 0
done
