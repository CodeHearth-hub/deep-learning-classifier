"""
训练入口脚本
用法: python scripts/train.py --config configs/default.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, load_config, get_device, save_class_names
from src.models import build_model, count_parameters
from src.dataset import build_dataloaders
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    set_seed(config['train'].get('seed', 42))

    # 获取设备
    device = get_device()

    # 构建数据加载器
    print("Building dataloaders...")
    train_loader, val_loader, class_names = build_dataloaders(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Num classes: {len(class_names)}")

    # 保存类别名称
    save_class_names(class_names, os.path.join(config['log']['save_dir'], 'class_names.txt'))

    # 构建模型
    print("Building model...")
    model = build_model(config)
    print(f"  Model: {config['model']['name']}")
    print(f"  Parameters: {count_parameters(model):,}")

    # 构建训练器并训练
    trainer = Trainer(model, train_loader, val_loader, config, device=device,
                      class_names=class_names)
    best_acc = trainer.train()

    print(f"\nTraining finished! Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
