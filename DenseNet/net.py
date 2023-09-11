# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.utils.checkpoint as cp
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate,
                 drop_rate, bn_size, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate,
                               kernel_size=1, stride=1, bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)),

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        """
        这段代码定义了一个名为bn_function的函数，它用于对输入特征进行批量归一化。
        首先，它会将输入特征连接在一起，然后使用conv1、relu1和bn1操作进行批量归一化。最后，返回批量归一化后的输出张量
        Args:
            inputs: 输入数据

        Returns:返回合并瓶颈层处理之后的张量

        """
        concated_features = torch.cat(inputs, 1)
        # 这里其实就是实现瓶颈层操作
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input):
        """
        这段代码定义了一个名为any_requires_grad的方法，用于检查输入张量列表中是否至少有一个张量需要计算梯度。
        这个方法被用于检查网络中的所有张量是否都需要计算梯度，以便在训练过程中对参数进行更新。
        具体来说，它遍历输入张量的列表，检查每个张量是否具有requires_grad属性。
        如果至少有一个张量具有requires_grad属性，则返回True，表示至少有一个张量需要计算梯度。
        如果所有张量都不需要计算梯度，则返回False。
        Args:
            input: 输入的数据

        Returns:输出一个布尔值

        """
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input):
        """
        具体来说，它使用torch.jit.checkpoint装饰器对瓶颈层进行包装，以便在训练过程中对瓶颈层进行checkpoint。
        然后，它将输入张量列表作为参数传递给包装后的瓶颈层，并返回输出张量。
        Args:
            input: 输入的数据

        Returns:输出经过checkpoint包装后的数据

        """
        def closure(*inputs):
            return self.bn_function(inputs)
        return cp.checkpoint(closure, *input)

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        # 返回瓶颈层的运行结果
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        # 这里完成3*3卷积的任务
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size,
                 growth_rate, drop_rate, memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate,
                               drop_rate,
                               bn_size,
                               memory_efficient)
            self.add_module('dense_layer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        # for name, layer in self.items():
        # 这里是体现Dense密集连接的重要部分，其主要内容是：
        # 将输入本block块的数据保存至列表中
        # 然后按照顺序读取模块中各个DenseLayer，获取其输出结果，将其保存到已经建立的特征列表中，按照channel进行垒叠
        # 最后将叠加结果进行返回
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10, memory_efficient=False):
        super(DenseNet, self).__init__()
        self.firstBlock = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        num_features = num_init_features
        for i, num_layer in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layer,
                num_input_features=num_features,
                bn_size=bn_size,
                drop_rate=drop_rate,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.firstBlock.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layer*growth_rate
            if i != len(block_config) - 1:
                transition = Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                self.firstBlock.add_module('transition%d' % (i + 1), transition)
                num_features = num_features // 2
        self.firstBlock.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # 权重和偏执初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.firstBlock(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# net = DenseNet()
# print(net)