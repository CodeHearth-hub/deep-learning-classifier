"""
工具函数
"""
import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """获取最佳计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Acc: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    return model


def save_class_names(class_names, save_path):
    """保存类别名称映射"""
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, name in enumerate(class_names):
            f.write(f"{i},{name}\n")
    print(f"Class names saved to {save_path}")


def load_class_names(file_path):
    """加载类别名称映射"""
    class_names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                class_names.append(parts[1])
    return class_names
