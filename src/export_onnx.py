"""
模型导出与推理优化
AI应用工程师核心技能：ONNX导出、模型量化、推理加速
支持：PyTorch -> ONNX导出、INT8量化、推理速度对比
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn


class ModelExporter:
    """模型导出与优化工具"""

    def __init__(self, model, input_size=(1, 3, 224, 224), device='cpu'):
        self.model = model.to(device).eval()
        self.input_size = input_size
        self.device = device
        self.dummy_input = torch.randn(input_size).to(device)

    def export_onnx(self, output_path='model.onnx', opset_version=17):
        """
        导出为 ONNX 格式
        ONNX 是跨框架的模型格式，可用于 TensorRT/OpenVINO/ONNX Runtime 部署
        """
        print(f"Exporting model to ONNX: {output_path}")

        torch.onnx.export(
            self.model,
            self.dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model saved to {output_path}")

        # 验证 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model validation passed ✓")
        except ImportError:
            print("Warning: onnx package not installed, skipping validation")

        return output_path

    def quantize_dynamic(self, output_path='model_quantized.pth'):
        """
        动态量化（INT8）
        适用于CPU推理，权重和激活动态量化，推理速度提升2-3倍
        """
        print("Applying dynamic quantization (INT8)...")

        # 只量化线性层和LSTM
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )

        torch.save(quantized_model.state_dict(), output_path)
        print(f"Quantized model saved to {output_path}")

        # 对比模型大小
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(quantized_model)
        print(f"Model size: {original_size:.2f} MB -> {quantized_size:.2f} MB "
              f"(reduced {100*(1-quantized_size/original_size):.1f}%)")

        return quantized_model

    def benchmark_inference(self, model=None, n_warmup=10, n_runs=100, batch_size=1):
        """
        推理速度基准测试
        返回：平均延迟、吞吐量、P50/P95/P99延迟
        """
        if model is None:
            model = self.model
        model = model.to(self.device).eval()

        dummy_input = torch.randn(batch_size, *self.input_size[1:]).to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy_input)

        # 正式测试
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        results = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_fps': batch_size * 1000 / np.mean(latencies),
            'batch_size': batch_size,
            'device': self.device
        }

        print(f"\nInference Benchmark (batch_size={batch_size}, device={self.device}):")
        print(f"  Mean latency: {results['mean_latency_ms']:.2f} ms")
        print(f"  P50: {results['p50_latency_ms']:.2f} ms | P95: {results['p95_latency_ms']:.2f} ms | P99: {results['p99_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")

        return results

    @staticmethod
    def _get_model_size(model):
        """计算模型大小（MB）"""
        temp_path = '/tmp/_temp_model_size.pth'
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb


def export_pipeline(model, output_dir='deploy', input_size=(1, 3, 224, 224)):
    """完整的模型导出流水线"""
    os.makedirs(output_dir, exist_ok=True)
    exporter = ModelExporter(model, input_size=input_size)

    # 1. 基准测试（原始模型）
    print("=" * 50)
    print("Original Model Benchmark")
    print("=" * 50)
    original_results = exporter.benchmark_inference()

    # 2. 导出 ONNX
    print("\n" + "=" * 50)
    print("ONNX Export")
    print("=" * 50)
    exporter.export_onnx(os.path.join(output_dir, 'model.onnx'))

    # 3. 动态量化
    print("\n" + "=" * 50)
    print("Model Quantization")
    print("=" * 50)
    quantized_model = exporter.quantize_dynamic(os.path.join(output_dir, 'model_quantized.pth'))

    # 4. 量化后基准测试
    print("\n" + "=" * 50)
    print("Quantized Model Benchmark")
    print("=" * 50)
    quantized_results = exporter.benchmark_inference(model=quantized_model)

    # 5. 对比报告
    print("\n" + "=" * 50)
    print("Comparison Report")
    print("=" * 50)
    print(f"{'Metric':<25} {'Original':>12} {'Quantized':>12} {'Improvement':>12}")
    print("-" * 65)
    print(f"{'Mean Latency (ms)':<25} {original_results['mean_latency_ms']:>12.2f} {quantized_results['mean_latency_ms']:>12.2f} {100*(1-quantized_results['mean_latency_ms']/original_results['mean_latency_ms']):>11.1f}%")
    print(f"{'Throughput (FPS)':<25} {original_results['throughput_fps']:>12.1f} {quantized_results['throughput_fps']:>12.1f} {100*(quantized_results['throughput_fps']/original_results['throughput_fps']-1):>11.1f}%")

    return {
        'original': original_results,
        'quantized': quantized_results
    }
