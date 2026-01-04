# ScrewTrue Backend API

FastAPI tabanlı atık sınıflandırma API'si.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment

### Railway

1. Railway hesabına giriş yap
2. "New Project" > "Deploy from GitHub repo"
3. Backend klasörünü seç
4. Environment variables ekle (gerekirse)

### Render

1. Render'da "New Web Service" oluştur
2. GitHub repo'yu bağla
3. Root Directory: `ScrewTrue_app/backend`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Heroku

```bash
heroku create screwtrue-api
git subtree push --prefix ScrewTrue_app/backend heroku main
```

## API Endpoints

- `GET /` - API durumu
- `GET /health` - Sağlık kontrolü
- `GET /classes` - Desteklenen sınıflar
- `POST /detect` - Dosya yükleme ile tespit
- `POST /detect/base64` - Base64 görüntü ile tespit

## Model Yolu

Model dosyası `screwtrue_model/run/weights/best.onnx` konumunda olmalıdır.

