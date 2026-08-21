"""
训练器：支持混合精度、知识蒸馏、早停、学习率调度
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import mixup_data, cutmix_data


class Trainer:
    """图像分类训练器"""

    def __init__(self, model, train_loader, val_loader, config,
                 device='cuda', class_names=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.class_names = class_names

        train_cfg = config['train']
        self.epochs = train_cfg['epochs']
        self.amp_enabled = train_cfg.get('amp', False)
        self.grad_clip = train_cfg.get('grad_clip', 1.0)
        self.early_stop_patience = train_cfg.get('early_stop_patience', 10)
        self.print_freq = config.get('log', {}).get('print_freq', 50)
        self.save_dir = config.get('log', {}).get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        # 优化器
        self.optimizer = self._build_optimizer()
        # 学习率调度器
        self.scheduler = self._build_scheduler()
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # 混合精度
        self.scaler = GradScaler(enabled=self.amp_enabled)
        # TensorBoard
        self.writer = SummaryWriter(log_dir='runs') if config.get('log', {}).get('tensorboard', True) else None

        # 训练状态
        self.best_acc = 0.0
        self.early_stop_counter = 0
        self.current_epoch = 0

        # 数据增强配置
        data_aug = config.get('data', {}).get('augment', {})
        self.mixup_alpha = data_aug.get('mixup_alpha', 0)
        self.cutmix_alpha = data_aug.get('cutmix_alpha', 0)

    def _build_optimizer(self):
        train_cfg = self.config['train']
        opt_name = train_cfg.get('optimizer', 'adamw').lower()
        lr = train_cfg['lr']
        weight_decay = train_cfg.get('weight_decay', 0.05)

        if opt_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr,
                                   momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self):
        train_cfg = self.config['train']
        sched_name = train_cfg.get('scheduler', 'cosine').lower()
        warmup_epochs = train_cfg.get('warmup_epochs', 0)

        if sched_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6)
        elif sched_name == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif sched_name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5)
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def _warmup_lr(self, epoch):
        """学习率预热"""
        warmup_epochs = self.config['train'].get('warmup_epochs', 0)
        warmup_lr = self.config['train'].get('warmup_lr', 1e-5)
        base_lr = self.config['train']['lr']
        if epoch < warmup_epochs:
            lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Mixup / CutMix
            use_mixup = self.mixup_alpha > 0 and np.random.rand() < 0.5
            use_cutmix = self.cutmix_alpha > 0 and not use_mixup and np.random.rand() < 0.5

            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, targets, self.mixup_alpha)
            elif use_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, self.cutmix_alpha)

            self.optimizer.zero_grad()

            with autocast(enabled=self.amp_enabled):
                outputs = self.model(images)
                if use_mixup or use_cutmix:
                    loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if use_mixup or use_cutmix:
                correct += (lam * predicted.eq(targets_a).sum().float() +
                           (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            else:
                correct += predicted.eq(targets).sum().item()

            if batch_idx % self.print_freq == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.amp_enabled):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        acc = 100. * correct / total
        return total_loss / len(self.val_loader), acc, all_preds, all_targets

    def train(self):
        """完整训练流程"""
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}, AMP: {self.amp_enabled}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self._warmup_lr(epoch)

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, preds, targets = self.validate()

            # 学习率调度
            warmup_epochs = self.config['train'].get('warmup_epochs', 0)
            if epoch >= warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # TensorBoard 日志
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Acc/train', train_acc, epoch)
                self.writer.add_scalar('Acc/val', val_acc, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'config': self.config,
                }, os.path.join(self.save_dir, 'best.pth'))
                print(f"  -> New best accuracy: {val_acc:.2f}%")
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 早停
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if self.writer:
            self.writer.close()

        print(f"Training complete. Best validation accuracy: {self.best_acc:.2f}%")
        return self.best_acc
