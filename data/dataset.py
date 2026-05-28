"""
数据加载与预处理模块
- PlantVillage 数据集加载
- 训练/验证/测试集划分 (7:1.5:1.5)
- 数据增强与归一化
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class PlantVillageDataset(Dataset):
    """PlantVillage 植物叶片病害数据集"""

    def __init__(self, root_dir: str, transform=None, phase: str = "train"):
        """
        Args:
            root_dir: 数据集根目录
            transform: 数据变换
            phase: 'train' / 'val' / 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.phase = phase

        self.images = []
        self.labels = []
        self.class_names = []
        self._load_data()

    def _load_data(self):
        """加载数据集并建立类别映射"""
        if not self.root_dir.exists():
            print(f"警告: 数据集目录 {self.root_dir} 不存在。"
                  f"请下载 PlantVillage 数据集并解压到此目录。")
            # 创建模拟数据结构用于测试
            self.root_dir.mkdir(parents=True, exist_ok=True)
            return

        class_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        if not class_dirs:
            print(f"警告: 在 {self.root_dir} 中未找到类别文件夹。")
            return

        self.class_names = [d.name for d in class_dirs]
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_dir in class_dirs:
            class_idx = class_to_idx[class_dir.name]
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)

        print(f"加载数据集: {len(self.images)} 张图像, {len(self.class_names)} 个类别")
        for i, name in enumerate(self.class_names):
            count = self.labels.count(i)
            print(f"  类别 {i}: {name} ({count} 张)")

        # 打乱数据
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 返回一个空白图像作为回退
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 224, mean=None, std=None, phase: str = "train"):
    """
    获取数据变换

    Args:
        image_size: 图像大小
        mean: 归一化均值
        std: 归一化标准差
        phase: 'train' / 'val' / 'test'
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # 验证/测试阶段只做缩放和归一化
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    创建训练/验证/测试数据加载器

    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        image_size: 图像大小
        mean: 归一化均值
        std: 归一化标准差
        num_workers: 数据加载线程数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # 加载完整数据集
    full_dataset = PlantVillageDataset(
        root_dir=data_dir,
        transform=None,
        phase="full"
    )

    # 按比例划分
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    print(f"\n数据集划分: 训练={train_size}, 验证={val_size}, 测试={test_size}")

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 为各子集设置对应的变换
    train_subset.dataset.transform = get_transforms(
        image_size, mean, std, phase="train"
    )

    val_dataset = PlantVillageDataset(data_dir, get_transforms(
        image_size, mean, std, phase="val"
    ))
    test_dataset = PlantVillageDataset(data_dir, get_transforms(
        image_size, mean, std, phase="test"
    ))

    # 使用相同的索引获取子集
    val_subset = torch.utils.data.Subset(
        val_dataset, val_subset.indices
    )
    test_subset = torch.utils.data.Subset(
        test_dataset, test_subset.indices
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    return train_loader, val_loader, test_loader, full_dataset.class_names


if __name__ == "__main__":
    # 快速测试
    train_ldr, val_ldr, test_ldr, classes = create_dataloaders(
        data_dir="./data/PlantVillage",
        batch_size=8,
    )
    print(f"\n类别数: {len(classes)}")
    print(f"类别: {classes[:5]}...")

    # 测试一个批次
    images, labels = next(iter(train_ldr))
    print(f"批次图像形状: {images.shape}")
    print(f"批次标签形状: {labels.shape}")
