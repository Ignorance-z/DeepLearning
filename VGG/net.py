# -*- coding: utf-8 -*-
import torch
from torch import nn
from utils import Vgg_block, get_block_param
from torch.utils.tensorboard import SummaryWriter


class VGG(nn.Module):
    def __init__(self, layers_num, in_channels):
        """
        这里完成VGG网路哦的初始化
        Args:
            layers_num: 这部分是VGG网络的层数，主要是卷积层+全连接层
            in_channels: 这里是输入图片的通道数
        """
        super(VGG, self).__init__()
        self.layers_num = layers_num
        self.in_channels = in_channels
        self.features = self._make_conv_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 2)
        )
        self._init_params()

    def _make_conv_layers(self):
        """
        这里完成VGG块的使用
        Returns:

        """
        layers, outs, length = get_block_param(self.layers_num)
        blocks = []
        in_channels = self.in_channels
        for idx in range(length):
            blocks.append(Vgg_block(
                layers[idx], in_channels, outs[idx]
            ))
            in_channels = outs[idx]
        return nn.Sequential(*blocks)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# x = torch.rand(1, 3, 224, 224)
# layers_num = int(input('请输入通道数（16 or 19）：'))
# vgg_net = VGG(layers_num, 3)
# # print(vgg_net)
# write = SummaryWriter('net')
# write.add_graph(vgg_net, x)
# write.close()