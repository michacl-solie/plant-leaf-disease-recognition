"""
配置文件加载工具
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径，默认为 config/config.yaml

    Returns:
        配置字典
    """
    if config_path is None:
        # 默认查找项目根目录下的 config/config.yaml
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 确保输出目录存在
    for key in ["checkpoint_dir", "log_dir", "output_dir"]:
        dir_path = Path(config["paths"][key])
        dir_path.mkdir(parents=True, exist_ok=True)

    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    递归合并配置字典
    """
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
