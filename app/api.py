"""
FastAPI 推理服务
模型部署为REST API
支持：单图预测、批量预测、Grad-CAM可视化、健康检查
运行: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from src.models import build_model
from src.inference import Predictor
from src.utils import load_config, load_checkpoint, load_class_names, get_device

app = FastAPI(
    title="Image Classification API",
    description="基于深度学习的图像分类推理服务",
    version="1.0.0"
)

# 全局模型实例
predictor = None
model_config = None


@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global predictor, model_config
    try:
        config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pth")

        if os.path.exists(checkpoint_path):
            model_config = load_config(config_path)
            device = get_device()
            model = build_model(model_config)
            model = load_checkpoint(model, checkpoint_path, device)

            class_names_path = os.path.join(os.path.dirname(checkpoint_path), 'class_names.txt')
            if os.path.exists(class_names_path):
                class_names = load_class_names(class_names_path)
            else:
                class_names = [f"class_{i}" for i in range(model_config['model']['num_classes'])]

            predictor = Predictor(model, class_names, device=device,
                                  img_size=model_config['data'].get('img_size', 224))
            print(f"Model loaded successfully. Classes: {len(class_names)}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, running in demo mode")
            predictor = None
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


@app.get("/")
async def root():
    """根路径"""
    return {"message": "Image Classification API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "device": "cuda" if predictor and predictor.device.type == 'cuda' else "cpu"
    }


@app.get("/classes")
async def get_classes():
    """获取所有类别"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": predictor.class_names, "count": len(predictor.class_names)}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...), top_k: int = 5):
    """
    单张图像预测
    - file: 上传的图像文件
    - top_k: 返回前K个预测结果
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 保存临时文件用于预测
        temp_path = f"/tmp/temp_{os.getpid()}.jpg"
        image.save(temp_path)
        results = predictor.predict(temp_path, top_k=top_k)
        os.remove(temp_path)

        return JSONResponse({
            "filename": file.filename,
            "predictions": results,
            "top_prediction": results[0] if results else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """批量图像预测"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            temp_path = f"/tmp/temp_batch_{os.getpid()}.jpg"
            image.save(temp_path)
            pred = predictor.predict(temp_path, top_k=3)
            os.remove(temp_path)
            results.append({"filename": file.filename, "predictions": pred})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return JSONResponse({"results": results, "total": len(results)})


@app.post("/gradcam")
async def gradcam_visualization(file: UploadFile = File(...)):
    """Grad-CAM 可解释性可视化，返回热力图"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        temp_path = f"/tmp/temp_gradcam_{os.getpid()}.jpg"
        image.save(temp_path)

        output_path = f"/tmp/gradcam_output_{os.getpid()}.jpg"
        overlay, pred_class = predictor.predict_with_gradcam(temp_path, save_path=output_path)
        os.remove(temp_path)

        # 返回base64编码的图片
        with open(output_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        os.remove(output_path)

        return JSONResponse({
            "predicted_class": predictor.class_names[pred_class],
            "gradcam_image": f"data:image/jpeg;base64,{img_base64}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
