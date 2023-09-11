# # -*- coding: utf-8 -*-
import torch
from torch import nn
from utils import Block, BottleNeck
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self, dim, block, block_num_list, class_num=10):
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.conv1 = nn.Conv2d(dim, self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layers(block, 64, block_num_list[0])
        self.layer2 = self._make_layers(block, 128, block_num_list[1], stride=2)
        self.layer3 = self._make_layers(block, 256, block_num_list[2], stride=2)
        self.layer4 = self._make_layers(block, 512, block_num_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.in_to_out_times, class_num)

    def _make_layers(self, block, out_channel, blocks_num, stride=1):
        downsample = None

        if stride != 1 or self.in_channel != out_channel*block.in_to_out_times:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel*block.in_to_out_times, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel*block.in_to_out_times)
            )

        net_layers = []
        net_layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel*block.in_to_out_times

        for _ in range(1, blocks_num):
            net_layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*net_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# net = ResNet(1, BottleNeck, [3, 4, 6, 3])
# net = ResNet(Block, [3, 4, 6, 3])
# print(net)
# x = torch.rand(1, 1, 96, 96)
# resnet = ResNet(1, BottleNeck, [3, 4, 6, 3])
# write_net = SummaryWriter('net')
# write_net.add_graph(resnet, x)
# write_net.close()