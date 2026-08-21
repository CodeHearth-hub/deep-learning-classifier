"""
推理与 Grad-CAM 可解释性分析
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms


class GradCAM:
    """Grad-CAM 可视化实现"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """生成 Grad-CAM 热力图"""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # 全局平均池化梯度
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        return cam.squeeze().cpu().numpy(), target_class

    @staticmethod
    def overlay_heatmap(image, cam, alpha=0.5):
        """将热力图叠加到原图上"""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        return overlay


class Predictor:
    """图像分类推理器"""

    def __init__(self, model, class_names, device='cuda', img_size=224):
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_path, top_k=5):
        """单张图像预测，返回 top-k 结果"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'class': self.class_names[idx.item()],
                'class_id': idx.item(),
                'confidence': prob.item()
            })
        return results

    def predict_with_gradcam(self, image_path, target_class=None, save_path=None):
        """预测并生成 Grad-CAM 可视化"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # 找到最后一个卷积层
        target_layer = self._find_last_conv_layer()
        grad_cam = GradCAM(self.model, target_layer)
        cam, pred_class = grad_cam.generate(input_tensor, target_class)

        # 叠加热力图
        original = np.array(image.resize((224, 224)))
        overlay = GradCAM.overlay_heatmap(original, cam)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay, pred_class

    def _find_last_conv_layer(self):
        """查找模型中最后一个卷积层"""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d,)):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model")
        return last_conv
