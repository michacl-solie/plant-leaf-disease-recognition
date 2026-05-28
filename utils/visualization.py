"""
可视化模块
- 损失/准确率曲线
- 混淆矩阵热力图
- Grad-CAM 热力图
- 学习率变化曲线
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = None,
    title: str = "训练损失曲线",
):
    """
    绘制训练/验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="训练损失", linewidth=1.5)
    plt.plot(epochs, val_losses, "r-", label="验证损失", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"损失曲线已保存: {save_path}")
    plt.close()


def plot_accuracy_curve(
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = None,
    title: str = "训练准确率曲线",
):
    """
    绘制训练/验证准确率曲线

    Args:
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, "b-", label="训练准确率", linewidth=1.5)
    plt.plot(epochs, val_accs, "r-", label="验证准确率", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 标注最佳验证准确率
    best_epoch = np.argmax(val_accs)
    best_acc = val_accs[best_epoch]
    plt.annotate(
        f"Best: {best_acc:.4f}",
        xy=(best_epoch + 1, best_acc),
        xytext=(best_epoch + 1, best_acc - 0.05),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=10,
        color="green",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"准确率曲线已保存: {save_path}")
    plt.close()


def plot_lr_curve(
    lrs: List[float],
    save_path: str = None,
    title: str = "学习率变化曲线",
):
    """
    绘制学习率变化曲线

    Args:
        lrs: 学习率列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(lrs) + 1)
    plt.plot(epochs, lrs, "g-", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"学习率曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    title: str = "混淆矩阵",
    figsize: tuple = (16, 14),
    normalize: bool = True,
):
    """
    绘制混淆矩阵热力图

    Args:
        confusion_matrix: 混淆矩阵 [num_classes, num_classes]
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图像大小
        normalize: 是否归一化
    """
    if normalize:
        cm = confusion_matrix.astype("float") / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm = confusion_matrix
        fmt = "d"
        vmax = None

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        cm,
        annot=False,
        fmt=fmt,
        cmap="YlOrRd",
        square=True,
        cbar_kws={"shrink": 0.8},
        vmin=0,
        vmax=vmax,
    )

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("预测标签", fontsize=13)
    plt.ylabel("真实标签", fontsize=13)

    if class_names and len(class_names) <= 38:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(class_names, rotation=0, fontsize=6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"混淆矩阵已保存: {save_path}")
    plt.close()


def plot_sample_predictions(
    images: torch.Tensor,
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    save_path: str = None,
    num_samples: int = 16,
    mean: list = None,
    std: list = None,
):
    """
    绘制样本预测结果网格

    Args:
        images: 图像张量 [B, C, H, W]
        labels: 真实标签
        preds: 预测标签
        class_names: 类别名称
        save_path: 保存路径
        num_samples: 显示样本数
        mean: 归一化均值
        std: 归一化标准差
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    num_samples = min(num_samples, images.size(0))
    rows = int(np.ceil(num_samples / 4))

    # 反归一化
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    images = images[:num_samples].cpu() * std_t + mean_t
    images = torch.clamp(images, 0, 1)

    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        true_label = class_names[labels[i]] if class_names else str(labels[i])
        pred_label = class_names[preds[i]] if class_names else str(preds[i])

        color = "green" if labels[i] == preds[i] else "red"
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=8)
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"样本预测结果已保存: {save_path}")
    plt.close()


def generate_grad_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: int = None,
    save_path: str = None,
    title: str = "Grad-CAM",
):
    """
    生成 Grad-CAM 热力图

    Args:
        model: 模型
        image: 输入图像 [1, C, H, W]
        target_layer: 目标卷积层 (通常为 layer4 最后一层)
        class_idx: 目标类别索引
        save_path: 保存路径
        title: 图表标题

    Returns:
        Grad-CAM 热力图 (numpy array)
    """
    model.eval()
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # 注册钩子
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # 前向传播
    image = image.unsqueeze(0) if image.dim() == 3 else image
    output = model(image)

    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    # 反向传播
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    # 计算 Grad-CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # 归一化
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 移除钩子
    handle_f.remove()
    handle_b.remove()

    # 可视化 (可选)
    if save_path:
        plt.figure(figsize=(8, 6))
        plt.imshow(cam, cmap="jet", alpha=0.7)
        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.colorbar(label="Activation")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM 已保存: {save_path}")
        plt.close()

    return cam
