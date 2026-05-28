"""
模型训练脚本

基于方案规范实现:
- CrossEntropy + Label Smoothing 损失函数
- AdamW 优化器 (β1=0.9, β2=0.999, weight_decay=1e-4)
- Cosine Annealing 学习率调度
- 早停策略 (patience=15)
- 混合精度训练 (AMP)
- 训练过程可视化
"""

import os
import sys
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import create_dataloaders
from models.se_resnet import SEResNet50
from utils.metrics import compute_accuracy, evaluate_model
from utils.visualization import (
    plot_loss_curve,
    plot_accuracy_curve,
    plot_lr_curve,
)
from utils.config_loader import load_config


class Trainer:
    """训练器类，封装完整的训练流程"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()

        # 创建输出目录
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.log_dir = Path(config["paths"]["log_dir"])
        self.output_dir = Path(config["paths"]["output_dir"])
        for d in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 训练状态记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

        print(f"使用设备: {self.device}")

    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            print("=" * 60)
            print("⚠ 警告: CUDA 不可用！将使用 CPU 训练")
            print("   - 每 epoch 约 60-120 分钟")
            print("   - 请安装 CUDA 版 PyTorch:")
            print("     pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple")
            print("=" * 60)
            return device

        gpu_id = self.config["device"]["gpu_id"]
        props = torch.cuda.get_device_properties(gpu_id)
        vram_gb = props.total_memory / 1024**3

        print(f"GPU: {props.name} ({vram_gb:.1f} GB)")
        print(f"CUDA: {torch.version.cuda} | 计算能力: {props.major}.{props.minor}")

        # 4GB 显存告警
        if vram_gb < 6:
            print(f"⚠ 显存仅 {vram_gb:.1f}GB，如 OOM 请将 config.yaml 中 batch_size 改小 (如 16)")

        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")

    def _build_model(self) -> nn.Module:
        """构建 SE-ResNet-50 模型"""
        model_cfg = self.config["model"]
        model = SEResNet50(
            num_classes=model_cfg["num_classes"],
            se_reduction=model_cfg["se_reduction"],
            dropout_rate=model_cfg["dropout_rate"],
            pretrained=model_cfg["pretrained"],
        )
        model = model.to(self.device)
        return model

    def _build_loss_fn(self) -> nn.Module:
        """构建带标签平滑的交叉熵损失函数"""
        loss_cfg = self.config["train"]["loss"]
        label_smoothing = loss_cfg.get("label_smoothing", 0.1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return loss_fn

    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """构建 AdamW 优化器"""
        optimizer_cfg = self.config["train"]["optimizer"]
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
            betas=tuple(optimizer_cfg["betas"]),
        )
        return optimizer

    def _build_scheduler(self, optimizer: optim.Optimizer, train_loader) -> optim.lr_scheduler._LRScheduler:
        """构建余弦退火学习率调度器"""
        sched_cfg = self.config["train"]["scheduler"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg["T_max"],
            eta_min=float(sched_cfg["eta_min"]),
        )
        return scheduler

    def train_one_epoch(
        self,
        model: nn.Module,
        train_loader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        use_amp = self.config["train"].get("amp", False)

        # tqdm 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度训练
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()

                # 梯度裁剪
                max_norm = self.config["train"].get("max_grad_norm", 1.0)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()

                # 梯度裁剪
                max_norm = self.config["train"].get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()

            # 统计
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            # 实时更新进度条
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{total_correct/total_samples:.3f}",
                "lr": f"{current_lr:.2e}",
            })

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {"loss": avg_loss, "accuracy": accuracy}

    def _check_early_stopping(self, val_acc: float, epoch: int) -> bool:
        """检查早停条件"""
        es_cfg = self.config["train"]["early_stopping"]
        min_delta = es_cfg.get("min_delta", 0.001)
        min_epochs = es_cfg.get("min_epochs", 30)  # 至少训练30轮

        if val_acc > self.best_val_acc + min_delta:
            self.best_val_acc = val_acc
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1

        # 未达最低训练轮数，不触发早停
        if epoch < min_epochs:
            return False

        should_stop = self.epochs_no_improve >= es_cfg["patience"]
        if should_stop:
            print(f"\n  ⏹ 早停触发! 连续 {es_cfg['patience']} 轮验证准确率未提升 (当前最佳: {self.best_val_acc:.4f})")
        return should_stop

    def save_checkpoint(
        self, model: nn.Module, optimizer: optim.Optimizer,
        scheduler, epoch: int, is_best: bool = False
    ):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"

        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")

    def train(self):
        """执行完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练: 基于改进SE-ResNet的植物叶片病害识别")
        print("=" * 60)

        # 加载数据
        dataset_cfg = self.config["dataset"]
        print(f"\n加载 PlantVillage 数据集: {dataset_cfg['data_dir']}")
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            data_dir=dataset_cfg["data_dir"],
            batch_size=self.config["train"]["batch_size"],
            image_size=dataset_cfg["image_size"],
            mean=dataset_cfg["mean"],
            std=dataset_cfg["std"],
            num_workers=self.config["train"]["num_workers"],
            train_ratio=dataset_cfg["split"]["train"],
            val_ratio=dataset_cfg["split"]["val"],
            test_ratio=dataset_cfg["split"]["test"],
        )

        # 构建模型
        print(f"\n构建 SE-ResNet-50 模型...")
        model = self._build_model()
        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # 损失函数、优化器、调度器
        loss_fn = self._build_loss_fn()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, train_loader)

        # 混合精度
        use_amp = self.config["train"].get("amp", False)
        scaler = GradScaler(enabled=use_amp and self.device.type == "cuda")

        # 训练循环
        epochs = self.config["train"]["epochs"]
        es_patience = self.config["train"]["early_stopping"]["patience"]
        start_time = time.time()

        print(f"\n开始训练 {epochs} 个 Epoch...")
        print(f"早停耐心值: {es_patience} 轮")
        print(f"混合精度训练: {'启用' if use_amp else '关闭'}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 训练
            train_metrics = self.train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, epoch
            )

            # 验证
            val_metrics = self.validate(model, val_loader, loss_fn)

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            # 记录指标
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            self.train_accs.append(train_metrics["accuracy"])
            self.val_accs.append(val_metrics["accuracy"])

            epoch_time = time.time() - epoch_start

            # 输出本 epoch 结果
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # 保存最佳模型
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.save_checkpoint(model, optimizer, scheduler, epoch, is_best=True)
                print(f"  *** 新的最佳模型! Val Acc: {self.best_val_acc:.4f} ***")

            # 定期保存检查点
            save_freq = self.config["paths"].get("save_frequency", 10)
            if epoch % save_freq == 0:
                self.save_checkpoint(model, optimizer, scheduler, epoch)

            # 早停检查
            if self._check_early_stopping(val_metrics["accuracy"], epoch):
                break

        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        print("\n" + "=" * 60)
        print(f"训练完成! 总耗时: {hours}h {minutes}m")
        print(f"最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print("=" * 60)

        # 绘制损失和准确率曲线
        print("\n生成训练曲线...")
        plot_loss_curve(
            self.train_losses, self.val_losses,
            save_path=str(self.output_dir / "loss_curve.png"),
        )
        plot_accuracy_curve(
            self.train_accs, self.val_accs,
            save_path=str(self.output_dir / "accuracy_curve.png"),
        )
        plot_lr_curve(
            self.learning_rates,
            save_path=str(self.output_dir / "lr_curve.png"),
        )

        return model, train_loader, val_loader, test_loader, class_names


def main():
    """主函数入口"""
    # 加载配置
    config = load_config()
    print(f"配置文件加载成功")

    # 创建训练器并开始训练
    trainer = Trainer(config)
    model, train_loader, val_loader, test_loader, class_names = trainer.train()

    # 加载最佳模型进行评估
    print("\n加载最佳模型进行测试集评估...")
    best_checkpoint = torch.load(
        Path(config["paths"]["checkpoint_dir"]) / "best_model.pth",
        map_location=trainer.device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # 测试集评估
    from utils.metrics import evaluate_model, print_evaluation_results
    test_results = evaluate_model(model, test_loader, trainer.device, class_names)
    print_evaluation_results(test_results, class_names)

    # 混淆矩阵
    from utils.visualization import plot_confusion_matrix
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        class_names=class_names,
        save_path=str(Path(config["paths"]["output_dir"]) / "confusion_matrix.png"),
    )

    print(f"\n所有输出文件保存至: {config['paths']['output_dir']}")
    print("训练流程结束!")

    return model, test_results


if __name__ == "__main__":
    main()
