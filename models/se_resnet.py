"""
SE-ResNet 模型实现
- SE-ResNet-50: 在 ResNet-50 的每个瓶颈块中嵌入 SE 注意力模块
- 支持 ImageNet 预训练权重加载
- 支持可调节的 SE 缩减比率和 Dropout 率
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .attention import SEBottleneck, SEBasicBlock


class SEResNet(nn.Module):
    """
    SE-ResNet 网络

    在标准 ResNet 的残差块中嵌入 SE (Squeeze-and-Excitation) 注意力模块，
    增强模型的通道特征选择能力。

    架构流程:
        输入 (224x224x3)
        → 初始卷积 (7x7, stride=2) → BN → ReLU → MaxPool
        → [SE-Bottleneck] × 3  (通道: 256)
        → [SE-Bottleneck] × 4  (通道: 512)
        → [SE-Bottleneck] × 6  (通道: 1024)
        → [SE-Bottleneck] × 3  (通道: 2048)
        → GAP → Dropout → FC → Softmax(38类)
    """

    def __init__(
        self,
        block,
        layers: list,
        num_classes: int = 38,
        se_reduction: int = 16,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
    ):
        """
        Args:
            block: 残差块类型 (SEBasicBlock 或 SEBottleneck)
            layers: 各阶段残差块数量 [3, 4, 6, 3]
            num_classes: 分类类别数
            se_reduction: SE模块缩减比率
            dropout_rate: Dropout比率
            pretrained: 是否加载 ImageNet 预训练权重
        """
        super(SEResNet, self).__init__()
        self.inplanes = 64
        self.se_reduction = se_reduction

        # 初始卷积层: 7x7, stride=2, 64 通道
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段的 SE-ResNet 残差块
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout + 全连接层
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重 (未加载预训练时)
        if not pretrained:
            self._initialize_weights()

        # 加载 ImageNet 预训练权重
        if pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建一个阶段的残差块组"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.se_reduction)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=self.se_reduction))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """加载 ImageNet 预训练权重"""
        print("加载 ImageNet 预训练权重...")
        try:
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # 加载除全连接层外的权重
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()

            # 过滤掉不匹配的键 (SE模块和fc层)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

            matched = len(pretrained_dict)
            total = len(model_dict)
            print(f"  预训练权重加载完成: {matched}/{total} 个参数匹配")
        except Exception as e:
            print(f"  警告: 预训练权重加载失败 ({e})，使用随机初始化")
            self._initialize_weights()

    def forward(self, x):
        """前向传播"""
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # SE-ResNet 残差块阶段
        x = self.layer1(x)   # [B, 256, 56, 56]
        x = self.layer2(x)   # [B, 512, 28, 28]
        x = self.layer3(x)   # [B, 1024, 14, 14]
        x = self.layer4(x)   # [B, 2048, 7, 7]

        # 全局平均池化
        x = self.avgpool(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]

        # Dropout + 全连接
        x = self.dropout(x)
        x = self.fc(x)       # [B, num_classes]

        return x

    def get_features(self, x):
        """提取特征向量 (用于 Grad-CAM 等可视化)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = x  # [B, 2048, 7, 7]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)

        return features, logits


def SEResNet50(
    num_classes: int = 38,
    se_reduction: int = 16,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
) -> SEResNet:
    """
    创建 SE-ResNet-50 模型

    架构: [3, 4, 6, 3] 瓶颈块配置，每块均嵌入 SE 模块

    Args:
        num_classes: 分类类别数 (PlantVillage 为 38)
        se_reduction: SE模块缩减比率
        dropout_rate: Dropout比率
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        SE-ResNet-50 模型实例
    """
    model = SEResNet(
        block=SEBottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        se_reduction=se_reduction,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )
    return model


if __name__ == "__main__":
    # 快速测试模型
    print("=" * 60)
    print("SE-ResNet-50 模型测试")
    print("=" * 60)

    model = SEResNet50(num_classes=38, pretrained=True)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 测试特征提取
    features, logits = model.get_features(dummy_input)
    print(f"特征图形状: {features.shape}")
    print(f"Logits形状: {logits.shape}")
    print("模型测试通过!")
