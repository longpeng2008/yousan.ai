import argparse
import time

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import cv2
import os
from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_dir', type=str, help='test low resolution image dir')
parser.add_argument('--result_dir', type=str, help='result image dir')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

images = os.listdir(opt.image_dir)
cv2.namedWindow("sr result", 0)
for imagepath in images:
    image = Image.open(os.path.join(opt.image_dir,imagepath)) ## 读取图片
    imageblob = ToTensor()(image).unsqueeze(0) ## 转化为tensor并扩充维度
    if TEST_MODE:
        imageblob = imageblob.cuda()

    start = time.clock()
    out = model(imageblob) ## 模型推理
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu()) ## 将结果转为图片

    out_img.save(os.path.join(opt.result_dir,imagepath))

    ## 将原图进行4倍上采样，与结果图进行拼接
    cv2.imshow("sr result",cv2.cvtColor(np.concatenate((cv2.resize(np.asarray(image),dsize=(0,0),fx=4,fy=4),np.asarray(out_img)),axis=1),cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
cv2.destroyAllWindows()