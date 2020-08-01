import torch
from torch import nn
from torch.nn import functional as F


# noinspection PyAbstractClass
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# noinspection PyAbstractClass
class FeatureAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


# noinspection PyAbstractClass
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()
        self.pool = FeatureAvgPool()
        self.conv1 = ConvBnRelu(in_channels, out_channels, 1)
        self.conv2 = ConvBnRelu(in_channels, out_channels, 3,
                                padding=rates[0], dilation=rates[0])
        self.conv3 = ConvBnRelu(in_channels, out_channels, 3,
                                padding=rates[1], dilation=rates[1])
        self.conv4 = ConvBnRelu(in_channels, out_channels, 3,
                                padding=rates[2], dilation=rates[2])
        self.conv5 = ConvBnRelu(in_channels + out_channels*(len(rates)+1), out_channels, 1)

    def forward(self, x):
        pool = self.pool(x)
        y1 = self.conv1(x)
        y6 = self.conv2(x)
        y12 = self.conv3(x)
        y18 = self.conv4(x)
        concat = torch.cat((y1, y6, y12, y18, pool), 1)
        x = self.conv5(concat)
        return x
