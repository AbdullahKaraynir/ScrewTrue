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

# Roboflow API Configuration - Model 1 (Phillips/Pozidriv)
MODEL_1_URL = "https://serverless.roboflow.com/dataminingproject-avr2o/2"
MODEL_1_API_KEY = "rSNcCctYlXx2bkMecwZk"
MODEL_1_ALLOWED = ["Phillips", "Pozidriv"]
MODEL_1_THRESHOLD = 0.65

# Roboflow API Configuration - Model 2 (Other screw types)
MODEL_2_URL = "https://serverless.roboflow.com/dataminingproject-avr2o/5"
MODEL_2_API_KEY = "rSNcCctYlXx2bkMecwZk"
MODEL_2_ALLOWED = [
    "Phillips",
    "Pozidriv",
    "Torx",
    "Hex/Allen",
    "Slotted",
    "Security Torx",
    "Pentalobe",
    "Tri-wing",
    "Spanner",
    "Triangle"
]
MODEL_2_THRESHOLD = 0.55


def call_roboflow_api(image_base64: str, api_url: str, api_key: str) -> dict:
    """
    Roboflow API'sine goruntu gonder ve sonuc al
    """
    try:
        # Roboflow API endpoint - API key query parameter olarak
        url = f"{api_url}?api_key={api_key}"
        
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


def handle_model1_response(predictions: List[Dict[str, Any]], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Model 1 sonucunu filtrele (Phillips/Pozidriv)
    """
    for pred in predictions:
        pred_class = pred.get("class", "")
        pred_confidence = pred.get("confidence", 0.0)
        
        # Debug log
        print(f"[DEBUG] Model 1 - class: {pred_class}, conf: {pred_confidence}")
        
        # Allowed sınıf ve threshold kontrolü (case-insensitive)
        pred_class_normalized = pred_class.strip()
        is_allowed = any(allowed.lower() == pred_class_normalized.lower() for allowed in MODEL_1_ALLOWED)
        
        if is_allowed and pred_confidence >= MODEL_1_THRESHOLD:
            # Class adını normalize et
            class_name = pred_class_normalized.lower()
            if "phillips" in class_name:
                class_name = "phillips"
            elif "pozidriv" in class_name:
                class_name = "pozidriv"
            
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
            class_id = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else 0
            
            result = {
                "class_id": class_id,
                "class_name": class_name,
                "class_label": CLASS_LABELS_TR.get(class_name, class_name),
                "confidence": round(pred_confidence, 3),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "color": CLASS_COLORS.get(class_name, "#FFFFFF"),
                "model": 1
            }
            
            print(f"[DEBUG] Model 1 ACCEPTED - class: {result['class_label']}, conf: {result['confidence']}")
            return result
    
    return None


def handle_model2_response(predictions: List[Dict[str, Any]], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Model 2 sonucunu filtrele (Diğer vida tipleri)
    """
    for pred in predictions:
        pred_class = pred.get("class", "")
        pred_confidence = pred.get("confidence", 0.0)
        
        # Debug log
        print(f"[DEBUG] Model 2 - class: {pred_class}, conf: {pred_confidence}")
        
        # Allowed sınıf ve threshold kontrolü (case-insensitive)
        pred_class_normalized = pred_class.strip()
        is_allowed = any(allowed.lower() == pred_class_normalized.lower() for allowed in MODEL_2_ALLOWED)
        
        if is_allowed and pred_confidence >= MODEL_2_THRESHOLD:
            # Class adını normalize et
            class_name = pred_class_normalized.lower()
            # Mapping - Roboflow'dan gelen class isimlerini CLASS_NAMES formatına çevir
            if "phillips" in class_name:
                class_name = "phillips"
            elif "pozidriv" in class_name:
                class_name = "pozidriv"
            elif "torx" in class_name:
                if "security" in class_name:
                    class_name = "security_torx"
                else:
                    class_name = "torx"
            elif "hex" in class_name or "allen" in class_name:
                class_name = "hex_allen"
            elif "slotted" in class_name:
                class_name = "slotted"
            elif "pentalobe" in class_name:
                class_name = "pentalobe"
            elif "tri" in class_name and "wing" in class_name:
                class_name = "tri_wing"
            elif "spanner" in class_name:
                class_name = "spanner"
            elif "triangle" in class_name:
                class_name = "triangle"
            
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
            class_id = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else 0
            
            result = {
                "class_id": class_id,
                "class_name": class_name,
                "class_label": CLASS_LABELS_TR.get(class_name, class_name),
                "confidence": round(pred_confidence, 3),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "color": CLASS_COLORS.get(class_name, "#FFFFFF"),
                "model": 2
            }
            
            print(f"[DEBUG] Model 2 ACCEPTED - class: {result['class_label']}, conf: {result['confidence']}")
            return result
    
    return None




def detect_screw_cascade(image_base64: str, image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Cascade sistem: Önce Model 1, sonra Model 2
    """
    # ADIM 3 - MODEL 1 İÇİN REQUEST
    print("[DEBUG] Calling Model 1 (Phillips/Pozidriv)...")
    try:
        res1 = call_roboflow_api(image_base64, MODEL_1_URL, MODEL_1_API_KEY)
        predictions1 = res1.get("predictions", [])
        
        # Image boyutlarını response'tan al (varsa)
        img_w = res1.get("image", {}).get("width", image_width)
        img_h = res1.get("image", {}).get("height", image_height)
        
        # ADIM 4 - MODEL 1 SONUCUNU FİLTRELE
        m1_result = handle_model1_response(predictions1, img_w, img_h)
        if m1_result:
            return {"model": 1, "result": m1_result}
    except Exception as e:
        print(f"[DEBUG] Model 1 error: {str(e)}")
    
    # ADIM 5 - MODEL 2'YE GEÇ
    print("[DEBUG] Model 1 no result, calling Model 2...")
    try:
        res2 = call_roboflow_api(image_base64, MODEL_2_URL, MODEL_2_API_KEY)
        predictions2 = res2.get("predictions", [])
        
        # Image boyutlarını response'tan al (varsa)
        img_w = res2.get("image", {}).get("width", image_width)
        img_h = res2.get("image", {}).get("height", image_height)
        
        # ADIM 5 - MODEL 2 SONUCUNU FİLTRELE
        m2_result = handle_model2_response(predictions2, img_w, img_h)
        if m2_result:
            return {"model": 2, "result": m2_result}
    except Exception as e:
        print(f"[DEBUG] Model 2 error: {str(e)}")
    
    print("[DEBUG] No result from both models")
    return None


def run_inference(image: np.ndarray, confidence: float = 0.25) -> tuple:
    """
    Roboflow API ile inference yap (CASCADE SİSTEM)
    """
    # Goruntuyu base64'e donustur
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Image boyutlari
    image_height, image_width = image.shape[:2]
    
    # Cascade sistem kullan
    cascade_result = detect_screw_cascade(image_base64, image_width, image_height)
    
    # Sonucu formatla
    detections = []
    if cascade_result and cascade_result.get("result"):
        detections.append(cascade_result["result"])
    
    return detections, image_width, image_height


@app.on_event("startup")
async def startup_event():
    """Uygulama baslagicinda kontrol"""
    print(f"[OK] Model 1 hazir: {MODEL_1_URL}")
    print(f"[OK] Model 2 hazir: {MODEL_2_URL}")
    print(f"[OK] Cascade sistem aktif")


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
        "api_type": "Roboflow Cascade",
        "model_1_url": MODEL_1_URL,
        "model_2_url": MODEL_2_URL,
        "model_1_allowed": MODEL_1_ALLOWED,
        "model_1_threshold": MODEL_1_THRESHOLD,
        "model_2_allowed": MODEL_2_ALLOWED,
        "model_2_threshold": MODEL_2_THRESHOLD,
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
