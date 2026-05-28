"""
模型评估指标模块
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)
"""

import numpy as np
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
    classification_report,
)


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算批次准确率

    Args:
        outputs: 模型输出 logits [B, C]
        labels: 真实标签 [B]

    Returns:
        准确率 (0-1)
    """
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def compute_precision_recall_f1(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    计算精确率、召回率、F1分数

    Args:
        all_preds: 预测标签
        all_labels: 真实标签
        average: 平均方式 ('macro', 'micro', 'weighted')

    Returns:
        包含 precision, recall, f1 的字典
    """
    return {
        "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
        "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, average=average, zero_division=0),
    }


def compute_confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> np.ndarray:
    """
    计算混淆矩阵

    Args:
        all_preds: 预测标签
        all_labels: 真实标签

    Returns:
        混淆矩阵 [num_classes, num_classes]
    """
    return sk_confusion_matrix(all_labels, all_preds)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str] = None,
    loss_fn: torch.nn.Module = None,
) -> Dict:
    """
    完整评估模型在给定数据集上的性能

    Args:
        model: 待评估模型
        dataloader: 数据加载器
        device: 计算设备
        class_names: 类别名称列表
        loss_fn: 损失函数 (可选)

    Returns:
        评估结果字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # 损失
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * images.size(0)

            # 准确率
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # 概率
            probs = F.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算各项指标
    results = {
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "total_samples": total_samples,
        "top1_correct": total_correct,
    }

    if loss_fn is not None:
        results["loss"] = total_loss / total_samples

    # 宏平均指标
    macro_metrics = compute_precision_recall_f1(all_preds, all_labels, "macro")
    results.update({f"macro_{k}": v for k, v in macro_metrics.items()})

    # 加权平均指标
    weighted_metrics = compute_precision_recall_f1(all_preds, all_labels, "weighted")
    results.update({f"weighted_{k}": v for k, v in weighted_metrics.items()})

    # 混淆矩阵
    results["confusion_matrix"] = compute_confusion_matrix(all_preds, all_labels)

    # 各类别准确率
    cm = results["confusion_matrix"]
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    if class_names:
        results["per_class_accuracy"] = {
            name: float(acc)
            for name, acc in zip(class_names, per_class_acc)
        }

    return results


def print_evaluation_results(results: Dict, class_names: List[str] = None):
    """格式化打印评估结果"""
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    print(f"  测试样本数:    {results.get('total_samples', 'N/A')}")
    print(f"  正确预测数:    {results.get('top1_correct', 'N/A')}")
    print(f"  准确率:        {results.get('accuracy', 0) * 100:.2f}%")

    if "loss" in results:
        print(f"  损失:          {results['loss']:.4f}")

    print(f"\n  宏平均精确率:  {results.get('macro_precision', 0):.4f}")
    print(f"  宏平均召回率:  {results.get('macro_recall', 0):.4f}")
    print(f"  宏平均F1分数:  {results.get('macro_f1_score', 0):.4f}")

    print(f"\n  加权平均精确率: {results.get('weighted_precision', 0):.4f}")
    print(f"  加权平均召回率: {results.get('weighted_recall', 0):.4f}")
    print(f"  加权平均F1分数: {results.get('weighted_f1_score', 0):.4f}")

    # 打印各类别准确率 (Top 5 最佳和最差)
    if "per_class_accuracy" in results and class_names:
        per_class = results["per_class_accuracy"]
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])

        print(f"\n  最佳5类:")
        for name, acc in sorted_classes[-5:]:
            print(f"    {name}: {acc * 100:.1f}%")

        print(f"\n  最差5类:")
        for name, acc in sorted_classes[:5]:
            print(f"    {name}: {acc * 100:.1f}%")

    print("=" * 60)
