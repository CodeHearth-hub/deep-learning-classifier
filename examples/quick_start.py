"""
深度学习图像分类 - 快速开始 Demo
运行: python examples/quick_start.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.models import Classifier, build_model, count_parameters
from src.inference import Predictor
from src.utils import set_seed, get_device
from src.export_onnx import ModelExporter

print("=" * 60)
print("  Deep Learning Classifier - Quick Start Demo")
print("=" * 60)

# 1. 设置随机种子
set_seed(42)
device = get_device()

# 2. 创建模型
print("\n[1] Creating model...")
model = Classifier('resnet18', num_classes=10, pretrained=False)
print(f"  Model: ResNet-18")
print(f"  Parameters: {count_parameters(model):,}")

# 3. 模拟推理
print("\n[2] Simulating inference...")
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Output logits: {output[0].numpy().round(3)}")

# 4. 特征提取
print("\n[3] Feature extraction...")
with torch.no_grad():
    features = model.get_features(dummy_input)
print(f"  Feature vector shape: {features.shape}")
print(f"  Feature vector (first 10): {features[0][:10].numpy().round(3)}")

# 5. 模型导出（ONNX）
print("\n[4] Model export (ONNX)...")
exporter = ModelExporter(model, input_size=(1, 3, 224, 224), device='cpu')
onnx_path = 'examples/demo_model.onnx'
exporter.export_onnx(onnx_path)
print(f"  ONNX model saved to: {onnx_path}")

# 6. 推理基准测试
print("\n[5] Inference benchmark...")
benchmark = exporter.benchmark_inference(n_warmup=5, n_runs=20, batch_size=1)

# 7. 清理
if os.path.exists(onnx_path):
    os.remove(onnx_path)

print("\n" + "=" * 60)
print("  Demo Complete!")
print("=" * 60)
print("\nNext steps:")
print("  1. Prepare your dataset in data/train and data/val")
print("  2. Edit configs/default.yaml for your settings")
print("  3. Run: python scripts/train.py --config configs/default.yaml")
print("  4. Deploy: uvicorn app.api:app --host 0.0.0.0 --port 8000")
print("  5. Docker: docker-compose up --build")
