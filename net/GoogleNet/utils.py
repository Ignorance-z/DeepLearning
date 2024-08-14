# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        该类主要用用来自定义一个卷积层模块

        Args:
            in_channels: 输入数据通道数
            out_channels: 输出数据通道数
            **kwargs: 其他参数
        """
        super(BasicConv2d, self).__init__()
        # self.BC = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
        #     nn.BatchNorm2d(out_channels, eps=1e-5),
        #     nn.ReLU()
        # )
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# 基本模块inception
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        该类定义GoogLeNet的基本模块inception

        Args:
            in_channels: 输入通道数
            ch1x1: 单独的1x1卷积核输出通道数
            ch3x3red: 3x3前置1x1卷积核输出通道数
            ch3x3: 3x3卷积核输出通道数
            ch5x5red: 5x5前置1x1卷积核输出通道数
            ch5x5: 5x5卷积核输出通道数
            pool_proj: 最大池化层输出通道数
        """
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        该部分完成辅助分类器的定义内容

        Args:
            in_channels: 输入通道数，来自于inception
            num_classes: 输出类别数
        """
        super(InceptionAux, self).__init__()

        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        )
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.aux(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        # print(x.shape)
        # 可以注释
        x = F.dropout(x, 0.7, training=self.training)

        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x