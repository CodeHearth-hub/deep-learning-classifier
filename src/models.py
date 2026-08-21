"""
图像分类模型定义
支持 ResNet、EfficientNet、Vision Transformer
"""
import torch
import torch.nn as nn
import timm


class Classifier(nn.Module):
    """统一的图像分类模型封装"""

    def __init__(self, model_name: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # 使用 timm 创建模型，支持多种 backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头，自定义
            global_pool='avg'
        )

        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        # 自定义分类头：BN -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        """提取特征向量（用于可视化/检索）"""
        return self.backbone(x)


class DistillationClassifier(nn.Module):
    """知识蒸馏学生模型"""

    def __init__(self, student_model: nn.Module, teacher_model: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, x):
        student_logits = self.student(x)
        if self.training:
            with torch.no_grad():
                teacher_logits = self.teacher(x)
            return student_logits, teacher_logits
        return student_logits


def build_model(config: dict) -> nn.Module:
    """根据配置构建模型"""
    model_cfg = config['model']
    model = Classifier(
        model_name=model_cfg['name'],
        num_classes=model_cfg['num_classes'],
        pretrained=model_cfg.get('pretrained', True),
        dropout=model_cfg.get('dropout', 0.3)
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
