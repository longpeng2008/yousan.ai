#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================

import argparse
import math
import os
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from model import StyledGenerator

## 生成图片
def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def make_noise(device,size):
    noises = []
    step = int(math.log(size, 2)) - 2
    for i in range(step + 1):
            size = 4 * 2 ** i
            noises.append(torch.randn(1, 1, size, size, device=device))
    return noises

    parser.add_argument(
        "--zlatent", type=int, default=1, help="zlatent or wlatent"
    )

if __name__ == "__main__":
    device = "cpu"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--files", type=str, help="path to image files to be projected"
    )

    parser.add_argument(
        "--image1", type=str, help="path to image1 to be mixed"
    )
    parser.add_argument(
        "--image2", type=str, help="path to image1 to be mixed"
    )
    args = parser.parse_args()

    ## 载入模型
    netG = StyledGenerator(512)
    netG.load_state_dict(torch.load(args.ckpt,map_location=device)["g_running"], strict=False)
    netG.eval()
    netG = netG.to(device)
    step = int(math.log(args.size, 2)) - 2

    ## 载入向量
    npy1 = args.files+args.image1
    npy2 = args.files+args.image2
    latent_in_1 = torch.from_numpy(np.load(npy1))
    latent_in_2 = torch.from_numpy(np.load(npy2))
    if len(latent_in_1.shape) == 1:
        latent_in_1 = latent_in_1.unsqueeze(0)
    if len(latent_in_2.shape) == 1:
        latent_in_2 = latent_in_2.unsqueeze(0)
   
    ranges=[[0,1],[2,3],[4,8]] ##分别交换粗粒度，中等粒度，精细粒度特征
    for i in range(0,3):
        img_gen = netG([latent_in_1,latent_in_2], step=step,mixing_range=(ranges[i][0],ranges[i][1])) ##生成的图片
        batch, channel, height, width = img_gen.shape
        img_ar = make_image(img_gen)
        print("img_ar.shape"+str(img_ar.shape))
        img_name = args.files+'/mixes/'+'mix_'+args.image1.split('.')[0]+'_'+args.image2.split('.')[0]+'_'+str(i)+'.jpg'
        pil_img = Image.fromarray(img_ar.squeeze(0))
        pil_img.save(img_name)


