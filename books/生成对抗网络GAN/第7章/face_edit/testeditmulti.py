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
        "--results", type=str, help="path to results files to be stored"
    )
    parser.add_argument(
        "--lamda", type=float, help="weight"
    )
    parser.add_argument(
        "--image1", type=str, help="image1"
    )
    parser.add_argument(
        "--image2", type=str, help="image2"
    )
    parser.add_argument(
        "--image3", type=str, help="image3"
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

    ## 构建噪声输入
    lamda = args.lamda
    latent1 = torch.from_numpy(np.load(args.image1))
    latent2 = torch.from_numpy(np.load(args.image2))
    latent3 = torch.from_numpy(np.load(args.image3))

    if len(latent1.shape) == 1:
        latent1 = latent1.unsqueeze(0)
    if len(latent2.shape) == 1:
        latent2 = latent2.unsqueeze(0)
    if len(latent3.shape) == 1:
        latent3 = latent3.unsqueeze(0)

    latent = latent1 + float(lamda)*(latent2-latent3)
    latent.to(device)

    if args.zlatent == 1:
        print("zlatent based")
        img_gen = netG([latent], step=step) ##生成的图片
        img_name = os.path.join(args.image1.replace('.npy','_multiedit_zbased_'+str(lamda)+'.jpg'))
    else:
        print("wlatent based")
        img_gen = netG([latent],mean_style = latent, step=step, style_weight=0) ##生成的图片
        img_name = os.path.join(args.image1.replace('.npy','_multiedit_wbased_'+str(lamda)+'.jpg'))

    img_ar = make_image(img_gen)
    pil_img = Image.fromarray(img_ar[0])
    pil_img.save(img_name) ##存储图片



