# -*- coding: utf-8 -*-
from torch import nn


def conv(in_channels, out_channels, kernel_size, stride=1):
    if kernel_size == 1:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    elif kernel_size == 3:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


class Block(nn.Module):
    # 该参数表示从输入该模块到输出该模块之间的channel倍数差
    # 显然对于18和34来说，这个倍数差是1
    # 根据论文中的模型来看，在34和50这两种网络模型应当对输出通道数进行一定的处理
    in_to_out_times = 1

    def __init__(self, in_channel, out_channel,
                 stride=1, down_sample=None):
        """

        Args:
            in_channel: 模块输入通道数
            out_channel: 模块输出通道数
            stride: 步长，一般是1
            down_sample: 下采样，用于对残差模块进行下采样，进行通道数匹配
        """
        super(Block, self).__init__()
        self.conv1 = conv(in_channel, out_channel, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channel, out_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    in_to_out_times = 4

    # 经常用于50-layers以上
    def __init__(self, in_channel, out_channel,
                 stride=1, down_sample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv(in_channel, out_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv(out_channel, out_channel, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = conv(out_channel, out_channel*self.in_to_out_times, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel*self.in_to_out_times)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out