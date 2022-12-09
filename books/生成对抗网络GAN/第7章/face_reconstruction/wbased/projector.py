#coding:utf8
import argparse
import math
import os
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

import lpips
from model import StyledGenerator

## TV噪声正则化
def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

## 噪声归一化
def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)

## 计算学习率
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

## 合并latent向量和噪声
def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


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

## 生成与图像大小相等的噪声
def make_noise(device,size):
    noises = []
    step = int(math.log(size, 2)) - 2
    for i in range(step + 1):
            size = 4 * 2 ** i
            noises.append(torch.randn(1, 1, size, size, device=device))
    return noises

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )

    ## 学习率参数
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")


    ## 噪声相关参数，噪声水平，噪声衰减，噪声正则化
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=10000,
        help="weight of the noise regularization",
    )
    
    ## MSE损失
    parser.add_argument("--mse", type=float, default=1, help="weight of the mse loss")
    
    ## 迭代次数
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")

    ## 重建图像
    parser.add_argument(
        "--files", type=str, help="path to image files to be projected"
    )

    ## 重建结果
    parser.add_argument(
        "--results", type=str, help="path to results files to be stored"
    )

    args = parser.parse_args()

    ## 计算latent向量的平均次数
    n_mean_latent = 10000

    ## 最小尺寸
    resize = min(args.size, 256)

    ## 预处理函数
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    ## 投影的人脸图片,将图片处理成一个batch
    imgs = []
    imgfiles = os.listdir(args.files)
    for imgfile in imgfiles:
        img = transform(Image.open(os.path.join(args.files,imgfile)).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    ## 载入模型
    netG = StyledGenerator(512)
    netG.load_state_dict(torch.load(args.ckpt,map_location=device)["g_running"], strict=False)
    netG.eval()
    netG = netG.to(device)
    step = int(math.log(args.size, 2)) - 2
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = netG.style(noise_sample)

        latent_mean = latent_out.mean(0) ## 计算平均W向量
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    ## 感知损失计算
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )
    ## 构建噪声输入
    noises_single = make_noise(device,args.size)

    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    ## 使用平均W向量来初始化W向量
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    ## 构建进度条
    pbar = tqdm(range(args.step+1))
    latent_path = []

    ## 优化学习W向量
    for i in pbar:
        t = i / args.step ## range(0,1)
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        ## 噪声衰减 = 标准差*噪声幅度*
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        print("noise_strength="+str(noise_strength))

        latent_n = latent_noise(latent_in, noise_strength.item())
        latent_n.to(device)

        img_gen = netG([torch.zeros(imgs.shape[0], 512).to(device)], mean_style = latent_n, noise=noises, step=step) ##生成的图片

        batch, channel, height, width = img_gen.shape

        ## 在不超过256的分辨率上计算损失
        if height > 256:
            factor = height // 256
            
            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])
        
        p_loss = percept(img_gen, imgs).sum()  ## 感知损失
        n_loss = noise_regularize(noises)      ## 噪声损失
        print("noises.shape:"+str(len(noises)))

        mse_loss = F.mse_loss(img_gen, imgs)   ## MSE损失

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss
        print("p_loss="+str(p_loss))
        print("n_loss="+str(n_loss))
        print("mse_loss="+str(mse_loss))
        print("loss="+str(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 10 == 0:
            latent_path.append(latent_in.detach().clone())

        '''
        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; loss: {loss.item():.4f}; lr: {lr:.4f}"
            )
        )
        '''

    img_gen = netG([torch.zeros(imgs.shape[0], 512).to(device)], mean_style = latent_n, noise=noises, step=step) ##生成的图片
    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(imgfiles):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        print("i="+str(i)+"; len of imgs:"+str(len(img_gen)))
        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
        }

        img_name = os.path.join(args.results,input_name)
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)
        np.save(os.path.join(args.results,input_name.split('.')[0]+'.npy'),latent_in[i].cpu().detach().numpy())

    ## json结果存储为pt文件
    filename = "project.pt"
    torch.save(result_file, filename)
