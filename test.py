"""
模型测试与评估脚本

功能:
- 加载已训练的模型
- 在测试集上全面评估
- 生成混淆矩阵、Grad-CAM 热力图
- 输出详细的分类报告
"""

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import create_dataloaders, PlantVillageDataset, get_transforms
from models.se_resnet import SEResNet50
from utils.metrics import evaluate_model, print_evaluation_results
from utils.visualization import (
    plot_confusion_matrix,
    plot_sample_predictions,
    generate_grad_cam,
)
from utils.config_loader import load_config


def load_trained_model(
    checkpoint_path: str,
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    加载训练好的模型

    Args:
        checkpoint_path: 检查点文件路径
        model_cfg: 模型配置
        device: 计算设备

    Returns:
        加载好的模型
    """
    model = SEResNet50(
        num_classes=model_cfg["num_classes"],
        se_reduction=model_cfg["se_reduction"],
        dropout_rate=model_cfg["dropout_rate"],
        pretrained=False,  # 不加载 ImageNet，使用我们训练的权重
    )
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"模型加载成功: {checkpoint_path}")
    print(f"  训练轮数: {checkpoint['epoch']}")
    print(f"  最佳验证准确率: {checkpoint['best_val_acc']:.4f}")

    return model


def test_single_image(
    model: nn.Module,
    image_path: str,
    class_names: list,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict:
    """
    对单张图像进行预测

    Args:
        model: 训练好的模型
        image_path: 图像路径
        class_names: 类别名称
        device: 计算设备
        config: 配置

    Returns:
        预测结果
    """
    from PIL import Image

    # 加载并预处理图像
    transform = get_transforms(
        image_size=config["dataset"]["image_size"],
        mean=config["dataset"]["mean"],
        std=config["dataset"]["std"],
        phase="test",
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()

    # Top-5
    top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)), dim=1)
    top5 = [
        {"class": class_names[idx], "probability": round(prob.item() * 100, 2)}
        for prob, idx in zip(top5_probs[0], top5_indices[0])
    ]

    result = {
        "image_path": image_path,
        "predicted_class": class_names[pred_idx],
        "confidence": round(pred_prob * 100, 2),
        "top5": top5,
    }

    print(f"\n预测结果:")
    print(f"  图像: {image_path}")
    print(f"  预测类别: {result['predicted_class']}")
    print(f"  置信度: {result['confidence']}%")
    print(f"  Top-5:")
    for i, item in enumerate(top5):
        print(f"    {i+1}. {item['class']}: {item['probability']}%")

    return result


def generate_grad_cam_visualizations(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list,
    device: torch.device,
    output_dir: str,
    num_samples: int = 5,
):
    """
    为多个样本生成 Grad-CAM 可视化

    Args:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称
        device: 计算设备
        output_dir: 输出目录
        num_samples: 可视化样本数
    """
    print(f"\n生成 {num_samples} 个 Grad-CAM 可视化...")

    # 获取目标层 (layer4 最后一层的最后一个卷积)
    target_layer = model.layer4[-1].conv3

    images, labels = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    output_dir = Path(output_dir) / "grad_cam"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        img = images[i:i+1]
        label = labels[i].item()
        pred = model(img).argmax(dim=1).item()

        title = f"True: {class_names[label]} | Pred: {class_names[pred]}"
        save_path = str(output_dir / f"gradcam_sample_{i+1}_{class_names[label]}.png")

        generate_grad_cam(
            model, img, target_layer,
            class_idx=pred, save_path=save_path, title=title,
        )

    print(f"Grad-CAM 可视化已保存至: {output_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("植物叶片病害识别 - 模型测试与评估")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 设置设备
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # 加载模型
    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        print("请先运行 train.py 训练模型。")
        sys.exit(1)

    model = load_trained_model(
        checkpoint_path, config["model"], device
    )

    # 加载测试数据
    print(f"\n加载测试数据...")
    dataset_cfg = config["dataset"]
    _, _, test_loader, class_names = create_dataloaders(
        data_dir=dataset_cfg["data_dir"],
        batch_size=config["train"]["batch_size"],
        image_size=dataset_cfg["image_size"],
        mean=dataset_cfg["mean"],
        std=dataset_cfg["std"],
        num_workers=config["train"]["num_workers"],
        train_ratio=dataset_cfg["split"]["train"],
        val_ratio=dataset_cfg["split"]["val"],
        test_ratio=dataset_cfg["split"]["test"],
    )

    # 测试集评估
    print("\n在测试集上评估模型...")
    test_results = evaluate_model(model, test_loader, device, class_names)
    print_evaluation_results(test_results, class_names)

    # 生成可视化
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 混淆矩阵
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        class_names=class_names,
        save_path=str(output_dir / "confusion_matrix.png"),
    )

    # 样本预测可视化
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()

    plot_sample_predictions(
        images, labels.numpy(), preds, class_names,
        save_path=str(output_dir / "sample_predictions.png"),
    )

    # Grad-CAM
    if config["visualization"]["grad_cam"]:
        generate_grad_cam_visualizations(
            model, test_loader, class_names, device,
            str(output_dir), num_samples=5,
        )

    print(f"\n所有测试结果保存至: {output_dir}")
    print("测试完成!")

    return test_results


if __name__ == "__main__":
    main()
