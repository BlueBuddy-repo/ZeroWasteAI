from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import scipy.special
import io

MODEL_PATH = "model.onnx"
THRESHOLD = 0.7  

# FastAPI 앱 
app = FastAPI(title="Reusable Classifier API (ONNX)", version="1.0.0")

# CORS 허용 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 도메인 제한 필요
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONNX 모델 로드
ort_session = ort.InferenceSession(MODEL_PATH)

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.transpose(image, (2, 0, 1)) 
    return np.expand_dims(image, axis=0).astype(np.float32)  

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": True,
        "onnxruntime_version": ort.__version__,
    }

@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능해요")

    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        x = preprocess(image)

        inputs = {ort_session.get_inputs()[0].name: x}
        outputs = ort_session.run(None, inputs)

        logits = outputs[0][0]
        probs = scipy.special.softmax(logits).tolist()
        reusable_prob = probs[1]

        label = "reusable" if reusable_prob >= THRESHOLD else "non_reusable"
        return label

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))