"""
主入口脚本 - 植物叶片病害识别系统

使用方式:
    python run.py               # 交互式菜单
    python run.py train         # 训练模型
    python run.py test          # 测试模型
    python run.py eval          # 评估模型
    python run.py predict <img> # 单张图片预测
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import torch


def show_banner():
    """显示欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║       植物叶片病害识别系统                          ║
    ║       Plant Leaf Disease Recognition               ║
    ║       基于改进 SE-ResNet-50                        ║
    ║       PlantVillage 数据集 (38类)                   ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def show_menu():
    """显示交互式菜单"""
    while True:
        print("\n请选择操作:")
        print("  1. 训练模型 (Train)")
        print("  2. 测试模型 (Test)")
        print("  3. 评估模型 (Evaluate)")
        print("  4. 单张图片预测 (Predict)")
        print("  5. 查看系统信息")
        print("  0. 退出")

        choice = input("\n请输入选项 [0-5]: ").strip()

        if choice == "1":
            print("\n启动模型训练...")
            from train import main as train_main
            train_main()
        elif choice == "2":
            print("\n启动模型测试...")
            from test import main as test_main
            test_main()
        elif choice == "3":
            print("\n启动模型评估...")
            evaluate()
        elif choice == "4":
            img_path = input("\n请输入图片路径: ").strip()
            predict_single(img_path)
        elif choice == "5":
            show_system_info()
        elif choice == "0":
            print("\n再见!")
            break
        else:
            print("无效选项，请重新输入。")


def evaluate():
    """评估已训练模型"""
    from test import main as test_main
    test_main()


def predict_single(image_path: str):
    """单张图片预测"""
    from utils.config_loader import load_config
    from models.se_resnet import SEResNet50
    from test import test_single_image

    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() and
                          config["device"]["use_cuda"] else "cpu")

    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"错误: 未找到模型检查点: {checkpoint_path}")
        return

    model_conf = config["model"]
    model = SEResNet50(
        num_classes=model_conf["num_classes"],
        se_reduction=model_conf["se_reduction"],
        dropout_rate=model_conf["dropout_rate"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 类别加载
    dataset_conf = config["dataset"]
    from data.dataset import PlantVillageDataset
    ds = PlantVillageDataset(dataset_conf["data_dir"])
    class_names = ds.class_names

    test_single_image(model, image_path, class_names, device, config)


def show_system_info():
    """显示系统信息"""
    print("\n系统信息:")
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        print(f"  当前显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  工作目录: {Path.cwd()}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="植物叶片病害识别系统 - 基于改进 SE-ResNet-50",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                  # 交互式菜单
  python run.py train            # 训练模型
  python run.py test             # 测试模型
  python run.py eval             # 评估模型
  python run.py predict leaf.jpg # 单张图片预测
        """,
    )

    parser.add_argument(
        "action",
        nargs="?",
        default="menu",
        choices=["train", "test", "eval", "predict", "menu", "info"],
        help="操作类型 (默认: menu)",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="额外参数 (如 predict 时的图片路径)",
    )

    args = parser.parse_args()

    show_banner()

    if args.action == "train":
        from train import main as train_main
        train_main()
    elif args.action == "test":
        from test import main as test_main
        test_main()
    elif args.action == "eval":
        evaluate()
    elif args.action == "predict":
        if not args.args:
            img_path = input("请输入图片路径: ").strip()
        else:
            img_path = args.args[0]
        predict_single(img_path)
    elif args.action == "info":
        show_system_info()
    else:
        show_menu()


if __name__ == "__main__":
    main()
