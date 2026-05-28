"""
SE (Squeeze-and-Excitation) 注意力模块
参考: Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
"""

import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块

    通过全局平均池化压缩空间信息，然后通过两个全连接层
    学习通道间的非线性关系，生成通道注意力权重。

    Args:
        in_channels: 输入通道数
        reduction: 缩减比率 (默认16)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Squeeze: 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: 两个全连接层
        reduced_channels = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            加权后的特征图 [B, C, H, W]
        """
        b, c, _, _ = x.size()

        # Squeeze: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.global_avg_pool(x).view(b, c)

        # Excitation: [B, C] -> [B, C]
        y = self.fc(y).view(b, c, 1, 1)

        # Scale: 通道加权
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """
    嵌入SE模块的残差基本块 (用于 ResNet-18/34)
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    """
    嵌入SE模块的瓶颈残差块 (用于 ResNet-50/101/152)

    瓶颈结构: 1x1(降维) -> 3x3 -> 1x1(升维) + SE + Skip Connection
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample

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
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
