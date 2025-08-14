from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import io
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

MODEL_PATH = "model.pt"  
THRESHOLD = 0.7         

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Reusable Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 나중에 프론트로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 전처리 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 모델 로드
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model

model = load_model()

class PredictResponse(BaseModel):
    label: Literal["reusable", "non_reusable"]
    isReusable: bool
    score: float                 
    probs: list[float]           # [non_reusable, reusable] 확률
    threshold: float


@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}

# 예측 API
@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    # 확장자 검증
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니니다")

    try:
        # 이미지 로드
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 전처리
        x = preprocess(image).unsqueeze(0).to(DEVICE)

        # 추론
        with torch.no_grad():
            logits = model(x)                
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()  
            reusable_prob = probs[1]          # 클래스 1 = reusable

        label = "reusable" if reusable_prob >= THRESHOLD else "non_reusable"

        return label
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))