# -*- coding: utf-8 -*-
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_block_param(layers):
    """
    获得VGG块中的相关参数

    Args:
        layers: 输入需要的层数，VGG16输入16，VGG19输入19

    Returns:
        返回卷积层重复次数、通道数、总块数

    """
    outs = [64, 128, 256, 512, 512]
    if layers == 16:
        return [2, 2, 3, 3, 3], outs, 5
    if layers == 19:
        return [2, 2, 4, 4, 4], outs, 5


def Vgg_block(num_conv, in_channels, out_channels):
    """
    可以使用该函数完成VGG网络块的定义

    Args:
        num_conv: conv-layers numbers
        in_channels: the input channels from the definition
            or the last out_channels from the former block
        out_channels: the output channels from the first layer of one vgg_block

    Returns:
        nn.Sequential: a conv_block

    """
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # 将list中的元素取出，送入Sequential
    return nn.Sequential(*layers)


def make_valid():
    """
    该部分用来完成数据集的再划分，即从训练集中简单划分出验证集
    """
    if not os.path.exists(r'../data/dogs-vs-cats/valid'):
        os.mkdir(r'../data/dogs-vs-cats/valid')

    for fname in os.listdir(r'../data/dogs-vs-cats/train'):
        _, img_num, _ = fname.split('.')
        img_num = int(img_num)
        filename = os.path.join(r'../data/dogs-vs-cats/train', fname)
        if img_num > 10999:
            os.rename(filename, filename.replace('train', 'valid'))


# 该部分用来进行mean和std计算
first_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])
# 该部分用来进行最终的网络图像处理
second_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 常用的是mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    transforms.Normalize((0.487, 0.449, 0.411), (0.211, 0.207, 0.206))
])