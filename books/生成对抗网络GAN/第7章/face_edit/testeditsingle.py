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
        "--direction", type=str, help="direction file to be read"
    )
    parser.add_argument(
        "--directionscale", type=float, help="direction scale"
    )
    parser.add_argument(
        "--zlatent", type=int, default=1, help="zlatent or wlatent"
    )
    args = parser.parse_args()

    from model import StyledGenerator
    netG = StyledGenerator(512)
    netG.load_state_dict(torch.load(args.ckpt,map_location=device)["g_running"], strict=False)
    netG.eval()
    netG = netG.to(device)
    step = int(math.log(args.size, 2)) - 2

    ## 载入方向向量
    direction = np.load(args.direction)
    directiontype = args.direction.split('/')[-1].split('.')[0]
    editscale = args.directionscale

    npys = glob.glob(args.files+"*.npy")
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    for npyfile in npys:
        latent = torch.from_numpy(np.load(npyfile)) ##获得与图像对应的z向量
        if len(latent.shape) == 1:
            latent = latent.unsqueeze(0)
        print("latent.shape="+str(latent.shape))
        img_name = os.path.join(npyfile.replace('.npy','_'+directiontype+'_'+str(editscale)+'.jpg'))
        if args.zlatent == 1:
            print("zlatent based")
            latent = latent + torch.from_numpy((editscale*direction[0]).astype(np.float32)) ##计算Z方向偏移量
            latent.to(device) ## 输入z向量+基于方向向量的编辑，很容易更改输入人脸的属性
            img_gen = netG([latent], step=step) ##生成的图片
            np.save(os.path.join(args.results,img_name.split('/')[-1].replace('.jpg','.npy')),latent) #存储Z向量
        else:
            print("wlatent based")
            w = torch.from_numpy((editscale*direction[0]).astype(np.float32)).to(device) ##计算W方向偏移量
            latent.to(device) ## 输入z向量
            img_gen = netG([latent], mean_style = latent + w, step=step, style_weight=0) ##生成结果只由mean_style决定
            np.save(os.path.join(args.results,img_name.split('/')[-1].replace('.jpg','.npy')),latent+w) #存储W向量

        img_ar = make_image(img_gen)
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save(os.path.join(args.results,img_name.split('/')[-1])) ##存储图片


