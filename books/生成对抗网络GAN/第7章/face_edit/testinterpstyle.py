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
from torchvision import utils
from PIL import Image
import numpy as np
import glob

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
    parser.add_argument(
        "--zlatent", type=int, default=1, help="zlatent or wlatent"
    )

    args = parser.parse_args()

    ## 载入模型
    from model import StyledGenerator

    netG = StyledGenerator(512)
    netG.load_state_dict(torch.load(args.ckpt,map_location=device)["g_running"], strict=False)
    netG.eval()
    netG = netG.to(device)
    step = int(math.log(args.size, 2)) - 2

    npy1 = args.files+args.image1
    npy2 = args.files+args.image2
    latent_in_1 = torch.from_numpy(np.load(npy1))
    latent_in_2 = torch.from_numpy(np.load(npy2))
    if len(latent_in_1.shape) == 1:
        latent_in_1 = latent_in_1.unsqueeze(0)
    if len(latent_in_2.shape) == 1:
        latent_in_2 = latent_in_2.unsqueeze(0)
    
    samples = 10
    for i in range(0,samples+1):
        lamda = float(i)/float(samples)
        ## 样式混合，z=lamda*z1+(1-lamda)*z2
        new_latent = lamda*latent_in_1 + (1-lamda)*latent_in_2

        if args.zlatent == 1: ##基于Z的混合，z=lamda*z1+(1-lamda)*z2
            print("zlatent based")
            img_name = args.files+'/interps/'+'interps_z_'+args.image1.split('.')[0]+'_'+args.image2.split('.')[0]+'_'+str(i)+'.jpg'
            img_gen = netG([new_latent], step=step) ##生成的图片
        else:##基于W的混合，w=lamda*w1+(1-lamda)*w2
            img_name = args.files+'/interps/'+'interps_w_'+args.image1.split('.')[0]+'_'+args.image2.split('.')[0]+'_'+str(i)+'.jpg'
            print("wlatent based")
            img_gen = netG([latent_in_1],mean_style=new_latent,step=step,style_weight=0) ##生成的图片

        img_ar = make_image(img_gen)
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save(img_name) ##存储图片


