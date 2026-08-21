"""
评估与推理脚本
用法:
  python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
  python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth --gradcam --image samples/test.jpg
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.utils import set_seed, load_config, get_device, load_checkpoint, load_class_names
from src.models import build_model
from src.dataset import build_dataloaders
from src.inference import Predictor


def evaluate_model(model, val_loader, class_names, device):
    """在验证集上评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    # 计算指标
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"\nOverall Accuracy: {acc*100:.2f}%\n")

    # 分类报告
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # 混淆矩阵
    print("Confusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)

    return acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate image classifier')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM visualization')
    parser.add_argument('--image', type=str, help='Path to input image for Grad-CAM')
    parser.add_argument('--output', type=str, default='gradcam_output.jpg', help='Grad-CAM output path')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    # 构建模型
    model = build_model(config)
    model = load_checkpoint(model, args.checkpoint, device)

    # 加载类别名称
    class_names_path = os.path.join(os.path.dirname(args.checkpoint), 'class_names.txt')
    if os.path.exists(class_names_path):
        class_names = load_class_names(class_names_path)
    else:
        _, _, class_names = build_dataloaders(config)

    if args.gradcam and args.image:
        # Grad-CAM 可视化
        predictor = Predictor(model, class_names, device=device,
                              img_size=config['data'].get('img_size', 224))
        results = predictor.predict(args.image, top_k=5)
        print("\nTop-5 Predictions:")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['class']}: {r['confidence']*100:.2f}%")

        overlay, pred_class = predictor.predict_with_gradcam(args.image, save_path=args.output)
        print(f"\nGrad-CAM visualization saved to {args.output}")
        print(f"Predicted class: {class_names[pred_class]}")
    else:
        # 完整评估
        _, val_loader, _ = build_dataloaders(config)
        evaluate_model(model, val_loader, class_names, device)


if __name__ == '__main__':
    main()
