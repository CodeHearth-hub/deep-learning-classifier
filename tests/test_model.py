"""
模型单元测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import Classifier, build_model, count_parameters


def test_model_creation():
    """测试模型创建"""
    model = Classifier('resnet18', num_classes=10, pretrained=False)
    assert model is not None
    assert count_parameters(model) > 0
    print("✓ Model creation test passed")


def test_forward_pass():
    """测试前向传播"""
    model = Classifier('resnet18', num_classes=10, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 10)
    print("✓ Forward pass test passed")


def test_feature_extraction():
    """测试特征提取"""
    model = Classifier('resnet18', num_classes=10, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model.get_features(x)
    assert features.dim() == 2
    print("✓ Feature extraction test passed")


def test_build_model_from_config():
    """测试从配置构建模型"""
    config = {
        'model': {
            'name': 'resnet18',
            'num_classes': 5,
            'pretrained': False,
            'dropout': 0.5
        }
    }
    model = build_model(config)
    assert model.num_classes == 5
    print("✓ Build model from config test passed")


if __name__ == '__main__':
    test_model_creation()
    test_forward_pass()
    test_feature_extraction()
    test_build_model_from_config()
    print("\nAll tests passed!")
