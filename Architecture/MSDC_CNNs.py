import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BnReluConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel,stride=1,pad=(0,0),bias=False):
        super(BnReluConv,self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes,out_planes,kernel,stride,pad,bias=bias)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer,self).__init__()
        each_planes = growth_rate // (1 + 3 + 5)
        planes_7x = each_planes
        planes_5x = each_planes * 3
        planes_3x = growth_rate - planes_5x - planes_7x
        self.bn = nn.BatchNorm2d(num_input_features, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv7x1 = BnReluConv(num_input_features, planes_7x, (7, 1), 1, (3, 0), bias=False)
        self.conv1x7 = BnReluConv(planes_7x, planes_7x, (1, 7), 1, (0, 3), bias=False)
        self.conv5x1 = BnReluConv(num_input_features, planes_5x, (5, 1), 1, (2, 0), bias=False)
        self.conv1x5 = BnReluConv(planes_5x, planes_5x, (1, 5), 1, (0, 2), bias=False)
        self.conv3x3 = BnReluConv(num_input_features, planes_3x, 3, 1, 1, bias=False)

        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        branch7 = self.conv7x1(x)
        branch7 = self.conv1x7(branch7)

        branch5 = self.conv5x1(x)
        branch5 = self.conv1x5(branch5)

        branch3 = self.conv3x3(x)

        output = torch.cat([branch7, branch5, branch3], dim=1)
        output = self.dropout(output)
        output = torch.cat([output, x], dim=1)
        return output

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class MSDC_CNNs(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(MSDC_CNNs, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        self.block1 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate

        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        First_num = num_features

        self.block2 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate

        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        sec_num = num_features

        self.block3 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate


        # Final batch norm
        self.FinalBN = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features+First_num+sec_num, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)

        block1 = self.block1(features)
        block1 = self.trans1(block1)

        block2 = self.block2(block1)
        block2 = self.trans2(block2)

        block3 = self.block3(block2)

        block3 = F.relu(block3, inplace=True)
        block1 = F.relu(block1)
        block2 = F.relu(block2)

        block1 = F.adaptive_avg_pool2d(block1, (1, 1)).view(block1.size(0), -1)
        block2 = F.adaptive_avg_pool2d(block2, (1, 1)).view(block2.size(0), -1)
        block3 = F.adaptive_avg_pool2d(block3, (1, 1)).view(block3.size(0), -1)

        out = torch.cat([block1,block2,block3],1)

        out = self.classifier(out)

        return out