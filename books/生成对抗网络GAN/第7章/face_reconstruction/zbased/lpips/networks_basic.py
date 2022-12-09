#coding:utf8
from __future__ import absolute_import

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from skimage import color
from IPython import embed
from . import pretrained_networks as pn

import lpips as util

## 计算tensor平均值
def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

## 上采样，H和W使用相同的尺度因子
def upsample(in_tens, out_H=64): 
    in_H = in_tens.shape[2]
    scale_factor = 1.*out_H/in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)

## 用于学习感知准则的网络
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type ##选择的基准网络
        self.pnet_tune = pnet_tune ##是否进行微调
        self.pnet_rand = pnet_rand ##是否使用随机参数，否则使用预训练模型
        self.spatial = spatial ##是否对不同空间位置单独进行计算指标
        self.lpips = lpips ##lpips因子
        self.version = version ##版本，新版本有缩放
        self.scaling_layer = ScalingLayer() ##尺度缩放

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns) ##特征数量

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        ## v0.0没有缩放，v0.1有输入缩放
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        ## 计算两幅图像的特征
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        ## 遍历特征层，计算L2归一化后的特征差
        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        ## 计算指标，当为lpips时，有需要学习的1*1层
        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else: ##平均值
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val

## 尺度缩放层
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


## 1x1的卷积层
class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

## BCE排序输出层
class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

## BCE排序损失
class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

## 不需要网络，直接计算L2或则DSSIM
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

## 不需要网络，直接计算L2距离
class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

## 不需要网络，直接计算DSSIM距离
class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = util.dssim(1.*util.tensor2im(in0.data), 1.*util.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
