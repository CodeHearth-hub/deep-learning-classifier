"""
数据集与数据增强
支持 RandAugment、Mixup、CutMix
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import random


class RandAugment:
    """RandAugment 数据增强策略"""

    def __init__(self, n: int = 2, m: int = 9):
        self.n = n  # 选择的变换数量
        self.m = m  # 增强强度 (0-10)
        self.augment_list = [
            ('autocontrast', 0),
            ('equalize', 0),
            ('rotate', 30),
            ('solarize', 256),
            ('color', 0.9),
            ('contrast', 0.9),
            ('brightness', 0.9),
            ('sharpness', 0.9),
            ('shear_x', 0.3),
            ('shear_y', 0.3),
            ('translate_x', 0.3),
            ('translate_y', 0.3),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, max_val in ops:
            img = self._apply_op(img, op, max_val)
        return img

    def _apply_op(self, img, op, max_val):
        magnitude = self.m / 10.0
        if op == 'autocontrast':
            return transforms.functional.autocontrast(img)
        elif op == 'equalize':
            return transforms.functional.equalize(img)
        elif op == 'rotate':
            angle = magnitude * max_val
            if random.random() > 0.5:
                angle = -angle
            return transforms.functional.rotate(img, angle)
        elif op == 'solarize':
            threshold = int((1 - magnitude) * max_val)
            return transforms.functional.solarize(img, threshold)
        elif op == 'color':
            factor = 1 + magnitude * max_val * (1 if random.random() > 0.5 else -1)
            return transforms.functional.adjust_saturation(img, factor)
        elif op == 'contrast':
            factor = 1 + magnitude * max_val * (1 if random.random() > 0.5 else -1)
            return transforms.functional.adjust_contrast(img, factor)
        elif op == 'brightness':
            factor = 1 + magnitude * max_val * (1 if random.random() > 0.5 else -1)
            return transforms.functional.adjust_brightness(img, factor)
        elif op == 'sharpness':
            factor = 1 + magnitude * max_val * (1 if random.random() > 0.5 else -1)
            return transforms.functional.adjust_sharpness(img, factor)
        elif op in ('shear_x', 'shear_y'):
            shear = magnitude * max_val
            if random.random() > 0.5:
                shear = -shear
            angle = [shear, 0] if op == 'shear_x' else [0, shear]
            return transforms.functional.affine(img, angle=0, translate=[0, 0], scale=1, shear=angle)
        elif op in ('translate_x', 'translate_y'):
            translate = magnitude * max_val
            if random.random() > 0.5:
                translate = -translate
            dx = int(translate * img.size[0]) if op == 'translate_x' else 0
            dy = int(translate * img.size[1]) if op == 'translate_y' else 0
            return transforms.functional.affine(img, angle=0, translate=[dx, dy], scale=1, shear=[0, 0])
        return img


def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # 生成裁剪区域
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def build_transforms(img_size: int = 224, is_train: bool = True,
                     use_randaugment: bool = True) -> transforms.Compose:
    """构建数据变换流水线"""
    if is_train:
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if use_randaugment:
            transform_list.append(RandAugment(n=2, m=9))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:
        transform_list = [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(transform_list)


def build_dataloaders(config: dict) -> tuple:
    """构建训练和验证数据加载器"""
    data_cfg = config['data']
    img_size = data_cfg.get('img_size', 224)
    batch_size = data_cfg.get('batch_size', 64)
    num_workers = data_cfg.get('num_workers', 4)

    train_transform = build_transforms(
        img_size, is_train=True,
        use_randaugment=data_cfg.get('augment', {}).get('randaugment', True)
    )
    val_transform = build_transforms(img_size, is_train=False)

    train_dataset = ImageFolder(data_cfg['train_dir'], transform=train_transform)
    val_dataset = ImageFolder(data_cfg['val_dir'], transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes
