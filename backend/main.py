"""
ScrewTrue - Object Detection API (Roboflow Version)
FastAPI backend for screw head type classification using Roboflow API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import requests
from typing import List, Dict, Any
import os

app = FastAPI(
    title="ScrewTrue API (Roboflow)",
    description="Vida basligi tanima icin nesne tespit API'si - Roboflow",
    version="3.0.0",
)

# CORS ayarlari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model siniflari - Vida basligi turleri
CLASS_NAMES = ["phillips", "pozidriv", "torx", "hex_allen", "slotted", "security_torx", "pentalobe", "tri_wing", "spanner", "triangle"]

# Sinif renkleri
CLASS_COLORS = {
    "phillips": "#E74C3C",           # Kirmizi
    "pozidriv": "#3498DB",           # Mavi
    "torx": "#F39C12",               # Turuncu
    "hex_allen": "#9B59B6",          # Mor
    "slotted": "#1ABC9C",            # Turkuaz
    "security_torx": "#E67E22",      # Turuncu-kahve
    "pentalobe": "#34495E",          # Koyu gri
    "tri_wing": "#16A085",           # Yesil-turkuaz
    "spanner": "#C0392B",            # Koyu kirmizi
    "triangle": "#27AE60",           # Yesil
}

# Sinif isimleri
CLASS_LABELS_TR = {
    "phillips": "Phillips",
    "pozidriv": "Pozidriv",
    "torx": "Torx",
    "hex_allen": "Hex/Allen",
    "slotted": "Slotted",
    "security_torx": "Security Torx",
    "pentalobe": "Pentalobe",
    "tri_wing": "Tri-wing",
    "spanner": "Spanner",
    "triangle": "Triangle",
}

# Roboflow API Configuration
ROBOFLOW_API_URL = "https://serverless.roboflow.com/dataminingproject-avr2o/2"
ROBOFLOW_API_KEY = "rSNcCctYlXx2bkMecwZk"


def call_roboflow_api(image_base64: str, confidence: float = 0.25) -> dict:
    """
    Roboflow API'sine goruntu gonder ve sonuc al
    """
    try:
        # Roboflow API endpoint - API key query parameter olarak
        url = f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}"
        
        # Base64 header'ini kaldir (varsa)
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Roboflow API genellikle base64 string bekler
        # İki format deneyebiliriz: JSON veya form-data
        try:
            # Format 1: JSON body
            payload = {
                "image": image_base64
            }
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except:
            # Format 2: Base64 string direkt
            payload = image_base64
            headers = {
                "Content-Type": "text/plain"
            }
            response = requests.post(url, data=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Roboflow API hatasi: {str(e)}")


def convert_roboflow_to_detections(roboflow_response: dict, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Roboflow API yanitini uygulama formatina donustur
    """
    detections = []
    
    if "predictions" not in roboflow_response:
        return detections
    
    predictions = roboflow_response["predictions"]
    image_width = roboflow_response.get("image", {}).get("width", 0)
    image_height = roboflow_response.get("image", {}).get("height", 0)
    
    for pred in predictions:
        pred_confidence = pred.get("confidence", 0.0)
        
        # Confidence threshold kontrolu
        if pred_confidence < confidence_threshold:
            continue
        
        # Class adini al
        class_name = pred.get("class", "").lower()
        
        # Class adini normalize et (Roboflow'dan gelen class adini CLASS_NAMES ile eslestir)
        class_name_mapped = class_name
        if class_name not in CLASS_NAMES:
            # Benzer class adlarini bul
            for cn in CLASS_NAMES:
                if cn in class_name or class_name in cn:
                    class_name_mapped = cn
                    break
            # Bulunamazsa orijinal adi kullan
            if class_name_mapped not in CLASS_NAMES:
                class_name_mapped = class_name
        
        # Bounding box
        x_center = pred.get("x", 0)
        y_center = pred.get("y", 0)
        width = pred.get("width", 0)
        height = pred.get("height", 0)
        
        # Center format -> Corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Sinirlari kontrol et
        x1 = max(0, min(int(x1), image_width))
        y1 = max(0, min(int(y1), image_height))
        x2 = max(0, min(int(x2), image_width))
        y2 = max(0, min(int(y2), image_height))
        
        # Class ID bul
        class_id = CLASS_NAMES.index(class_name_mapped) if class_name_mapped in CLASS_NAMES else 0
        
        detections.append({
            "class_id": class_id,
            "class_name": class_name_mapped,
            "class_label": CLASS_LABELS_TR.get(class_name_mapped, class_name_mapped),
            "confidence": round(pred_confidence, 3),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "color": CLASS_COLORS.get(class_name_mapped, "#FFFFFF"),
        })
    
    return detections


def run_inference(image: np.ndarray, confidence: float = 0.25) -> tuple:
    """
    Roboflow API ile inference yap
    """
    # Goruntuyu base64'e donustur
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Roboflow API'ye gonder
    roboflow_response = call_roboflow_api(image_base64, confidence)
    
    # Response'u uygulama formatina donustur
    detections = convert_roboflow_to_detections(roboflow_response, confidence)
    
    # Image boyutlari
    image_height, image_width = image.shape[:2]
    
    return detections, image_width, image_height


@app.on_event("startup")
async def startup_event():
    """Uygulama baslagicinda kontrol"""
    print(f"[OK] Roboflow API hazir: {ROBOFLOW_API_URL}")


@app.get("/")
async def root():
    """API durumu"""
    return {
        "status": "active",
        "message": "ScrewTrue Roboflow API calisiyor",
        "version": "3.0.0",
        "runtime": "Roboflow API",
    }


@app.get("/health")
async def health_check():
    """Saglik kontrolu"""
    return {
        "status": "healthy",
        "api_configured": True,
        "api_type": "Roboflow",
        "api_url": ROBOFLOW_API_URL,
        "classes": CLASS_NAMES,
    }


@app.get("/classes")
async def get_classes():
    """Mevcut siniflari dondur"""
    return {
        "classes": [
            {
                "id": i,
                "name": name,
                "label": CLASS_LABELS_TR.get(name, name),
                "color": CLASS_COLORS.get(name, "#FFFFFF"),
            }
            for i, name in enumerate(CLASS_NAMES)
        ]
    }


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.25):
    """
    Goruntude nesne tespiti yap
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Gecersiz dosya tipi. Sadece goruntu dosyalari kabul edilir.",
            )

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Goruntu okunamadi")

        # Roboflow API inference
        detections, w, h = run_inference(image, confidence)

        return {
            "success": True,
            "image_size": {"width": w, "height": h},
            "detections_count": len(detections),
            "detections": detections,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tespit hatasi: {str(e)}")


@app.post("/detect/base64")
async def detect_objects_base64(data: dict):
    """
    Base64 kodlu goruntude nesne tespiti yap
    """
    try:
        image_data = data.get("image")
        confidence = data.get("confidence", 0.25)

        if not image_data:
            raise HTTPException(status_code=400, detail="Goruntu verisi gerekli")

        # Base64 header'ini kaldir
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Goruntu decode edilemedi")

        # Roboflow API inference
        detections, w, h = run_inference(image, confidence)

        return {
            "success": True,
            "image_size": {"width": w, "height": h},
            "detections_count": len(detections),
            "detections": detections,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tespit hatasi: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
