# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class AlexNet(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 步长stride默认是1，可以进行省略
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # classifier
            nn.Flatten(),
            nn.Linear(in_features=(6*6*256), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=out_channel)
        )
        # 权重偏执初始化
    #     self.init_bias()
    #
    # def init_bias(self):
    #     for layer in self.net:
    #         if isinstance(layer, nn.Conv2d):
    #             nn.init.normal_(layer.weight, mean=0, std=0.01)
    #             nn.init.constant_(layer.bias, 0)
    #     nn.init.constant_(self.net[3].bias, 1)
    #     nn.init.constant_(self.net[8].bias, 1)
    #     nn.init.constant_(self.net[10].bias, 1)

    def forward(self, x):
        x = self.net(x)
        return x


# x = torch.rand(1, 3, 224, 224)
# net = AlexNet(3, 2)
# write = SummaryWriter('net')
# write.add_graph(net, x)
# write.close()