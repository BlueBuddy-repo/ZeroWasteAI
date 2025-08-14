from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import io
from PIL import Image
import numpy as np
import onnxruntime as ort

import torchvision.transforms as transforms

MODEL_PATH = "model.onnx"
THRESHOLD = 0.7

app = FastAPI(title="Reusable Classifier API (ONNX)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ONNX 세션 생성
ort_session = ort.InferenceSession(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True, "onnx_runtime": ort.__version__}

@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능해요")

    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        x = preprocess(image).unsqueeze(0).numpy()  # [1, 3, 224, 224]

        # ONNX 추론
        inputs = {ort_session.get_inputs()[0].name: x}
        outputs = ort_session.run(None, inputs)
        probs = outputs[0][0].tolist()  # [non_reusable, reusable]
        reusable_prob = probs[1]

        label = "reusable" if reusable_prob >= THRESHOLD else "non_reusable"
        return label

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))