#coding:utf8
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

## 剪枝配置参数
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='checkpoints/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to the model')
parser.add_argument('--save', default='pruned/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## 结果文件夹
resultdir = args.save + str(args.percent)
if not os.path.exists(resultdir):
    os.makedirs(resultdir)

## 定义模型载入参数
model = simpleconv3small(4)
checkpoint = torch.load(args.model,map_location=lambda storage,loc: storage)
model.load_state_dict(checkpoint['state_dict'])
if args.cuda:
    model.cuda()

print(model)

## 通道BN参数的数量
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

## 收集BN缩放系数
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

## 根据绝对值对BN缩放系数进行排序，根据裁剪比例获得阈值
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]
print('prun th='+str(thre))

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float() ##获得掩膜
        pruned = pruned + mask.shape[0] - torch.sum(mask) ## 累加被剪枝的通道数量
        m.weight.data.mul_(mask) ## 根据掩膜调整缩放系数
        m.bias.data.mul_(mask) ## 根据掩膜调整偏置系数
        cfg.append(int(torch.sum(mask))) ## 获得当前层保留的通道数
        cfg_mask.append(mask.clone()) ## 保存当前层掩膜
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

## 计算整体的剪枝率
pruned_ratio = pruned/total

print('Pre-processing Successful!')

## 数据
data_dir = './data'
image_size = 60
crop_size = 48
nclass = 4
    
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(crop_size,scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val': transforms.Compose([
            transforms.Scale(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
    }

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=64,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'val']}
train_loader = dataloaders['train']
test_loader = dataloaders['val']

## 对预剪枝后的模型计算准确率
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)

## 真正进行剪枝
print(cfg)
newmodel = simpleconv3(4)
if args.cuda:
    newmodel.cuda()

## 计算参数量
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(resultdir, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("threshold:"+str(thre))
    fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3) ##数据层通道数
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) ##返回非0的数组元组的索引
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone() ##权重赋值，覆盖之前的维度
        m1.bias.data = m0.bias.data[idx1.tolist()].clone() ##偏置赋值，覆盖之前的维度
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask): ##最后一个全连接层不更新，因为是分类层
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) ##输入通道掩膜
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) ##输出通道掩膜
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone() ##取输入维度
        w1 = w1[idx1.tolist(), :, :, :].clone() ##取输出维度
        m1.weight.data = w1.clone() ##权重赋值，覆盖之前的维度
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(resultdir, 'pruned.pth.tar'))

print(newmodel)
model = newmodel
test(model)
