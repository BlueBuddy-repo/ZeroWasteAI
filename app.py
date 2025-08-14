from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np
import onnxruntime as ort

MODEL_PATH = "model.onnx"
THRESHOLD = 0.7

# FastAPI 앱 생성
app = FastAPI(title="Reusable Classifier API (ONNX)", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시엔 제한 필요
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONNX 세션 생성
ort_session = ort.InferenceSession(MODEL_PATH)

# 이미지 전처리 함수
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224)) 
    image = np.array(image).astype(np.float32) / 255.0  
    image = (image - 0.5) / 0.5  #
    image = np.transpose(image, (2, 0, 1)) 
    return np.expand_dims(image, axis=0)  

# 헬스체크 엔드포인트
@app.get("/health")
def health():
    return {"ok": True, "onnx_runtime": ort.__version__}

# 예측 엔드포인트
@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능해요")

    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        x = preprocess(image)

        # ONNX 추론
        inputs = {ort_session.get_inputs()[0].name: x}
        outputs = ort_session.run(None, inputs)
        probs = outputs[0][0].tolist()  # [non_reusable, reusable]
        reusable_prob = probs[1]

        label = "reusable" if reusable_prob >= THRESHOLD else "non_reusable"
        return label

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))