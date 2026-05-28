# 基于改进ResNet的植物叶片病害识别

Plant Leaf Disease Recognition based on improved SE-ResNet-50.

## 项目简介

本项目在 ResNet-50 骨干网络中嵌入 SE (Squeeze-and-Excitation) 通道注意力模块，在 PlantVillage 公开数据集上实现植物叶片病害的高精度自动识别。

- **数据集**: PlantVillage (54,305 张图像, 38 个类别, 14 种作物)
- **模型**: SE-ResNet-50 (ImageNet 预训练 + SE 注意力机制)
- **目标准确率**: >= 98%
- **框架**: PyTorch 2.0+

## 项目结构

```
plant-leaf-disease-recognition/
├── config/
│   └── config.yaml           # 实验配置文件
├── data/
│   ├── __init__.py
│   └── dataset.py            # 数据加载与预处理
├── models/
│   ├── __init__.py
│   ├── attention.py          # SE 注意力模块
│   └── se_resnet.py          # SE-ResNet-50 模型
├── utils/
│   ├── __init__.py
│   ├── config_loader.py      # 配置加载器
│   ├── metrics.py            # 评估指标
│   └── visualization.py       # 可视化工具
├── train.py                  # 训练脚本
├── test.py                   # 测试与评估脚本
├── run.py                    # 主入口 (交互式菜单)
├── requirements.txt          # 依赖包
└── README.md                 # 项目说明
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8 (GPU训练推荐)
- NVIDIA RTX 3060 (6GB) 或同等算力

## 快速开始

### 1. 安装依赖（使用国内镜像源）

```bash
# 清华源 (推荐)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 或阿里源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 永久配置（一劳永逸）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

> 详见 [INSTALL.md](./INSTALL.md)

### 2. 准备数据集

PlantVillage 是植物病害识别标准数据集（54,305 张, 38 类, 14 种作物, ~827MB）。

**国内推荐下载**：阿里云天池直接下载 👉 [tianchi.aliyun.com/dataset/160100](https://tianchi.aliyun.com/dataset/160100)

更多下载方式详见 [DATA.md](./DATA.md)。

解压后目录结构：

```
data/PlantVillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── ...
└── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
```

### 3. 训练模型

```bash
# 交互式启动 (推荐)
python run.py

# 或命令行启动训练
python run.py train

# 或直接运行训练脚本
python train.py
```

### 4. 测试模型

```bash
python run.py test
# 或
python test.py
```

### 5. 单张图片预测

```bash
python run.py predict path/to/leaf.jpg
```

## 模型架构

```
输入图像 (3×224×224)
  → 初始卷积层 (7×7, stride=2) → BN → ReLU → MaxPool
  → [SE-ResNet Block 1] × 3   (通道: 256)
  → [SE-ResNet Block 2] × 4   (通道: 512)
  → [SE-ResNet Block 3] × 6   (通道: 1024)
  → [SE-ResNet Block 4] × 3   (通道: 2048)
  → 全局平均池化 (GAP)
  → Dropout (p=0.5)
  → 全连接层
  → Softmax 输出 (38类)
```

## 训练策略

| 组件 | 配置 |
|------|------|
| 损失函数 | CrossEntropy + Label Smoothing (α=0.1) |
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-4) |
| 学习率调度 | Cosine Annealing |
| 批次大小 | 32 |
| 训练轮数 | 100 (含早停 patience=15) |
| 混合精度 | AMP (自动混合精度) |

## 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)
- Grad-CAM 可视化

## 创新点

1. **SE 注意力机制**: 在 ResNet 瓶颈块中嵌入通道注意力，增强特征选择能力
2. **迁移学习策略**: 利用 ImageNet 预训练权重初始化
3. **标签平滑正则化**: 结合 Dropout、权重衰减综合防止过拟合
4. **Grad-CAM 可解释性**: 可视化模型决策依据

## 参考文献

- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hu, J. et al. (2018). Squeeze-and-Excitation Networks. CVPR.
- Mohanty, S. P. et al. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science.

## License

MIT License
