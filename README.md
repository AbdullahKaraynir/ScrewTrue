# 🔩 ScrewTrue - Vida Başlığı Tanıma Sistemi

ScrewTrue, yapay zeka destekli bir vida başlığı tanıma ve sınıflandırma sistemidir. Mobil uygulama ve REST API ile çalışan bu sistem, Roboflow'un cascade model mimarisi kullanarak 10 farklı vida tipini yüksek doğrulukla tespit edebilir.

##  İçindekiler

- [Özellikler](#-özellikler)
- [Teknoloji Stack](#-teknoloji-stack)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
  - [Backend Kurulumu](#backend-kurulumu)
  - [Mobil Uygulama Kurulumu](#mobil-uygulama-kurulumu)
- [Çalıştırma](#-çalıştırma)
  - [Backend'i Çalıştırma](#backendi-çalıştırma)
  - [Mobil Uygulamayı Çalıştırma](#mobil-uygulamayı-çalıştırma)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Cascade Model Sistemi](#-cascade-model-sistemi)
- [Deployment](#-deployment)
- [Geliştirme](#-geliştirme)

##  Özellikler

-  **10 Farklı Vida Tipi Tanıma**: Phillips, Pozidriv, Torx, Hex/Allen, Slotted, Security Torx, Pentalobe, Tri-wing, Spanner, Triangle
-  **Cascade Model Sistemi**: İki aşamalı model yapısı ile optimize edilmiş doğruluk
-  **Cross-Platform Mobil Uygulama**: iOS ve Android desteği
-  **Gerçek Zamanlı Tespit**: Canlı kamera akışında anlık tespit
-  **Galeri Desteği**: Mevcut fotoğraflardan tespit yapma
-  **Renk Kodlu Sınıflandırma**: Her vida tipi için özel renk gösterimi
-  **Güven Skoru Gösterimi**: Tespit doğruluğunu görselleştirme
-  **RESTful API**: Kolay entegrasyon için standart API yapısı

##  Teknoloji Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Python**: 3.11+
- **ML/AI**: Roboflow API (Cascade Models)
- **Image Processing**: OpenCV, NumPy, Pillow
- **Server**: Uvicorn

### Mobile
- **Framework**: React Native (Expo)
- **Camera**: expo-camera
- **Image Picker**: expo-image-picker
- **UI**: React Native Components + Linear Gradient

##  Proje Yapısı

```
ScrewTrue-main/
├── backend/                 # FastAPI backend servisi
│   ├── main.py            # Ana API dosyası
│   ├── requirements.txt   # Python bağımlılıkları
│   ├── Procfile          # Heroku deployment config
│   ├── runtime.txt       # Python versiyonu
│   └── README.md         # Backend dokümantasyonu
│
├── mobile/                # React Native mobil uygulama
│   ├── App.js            # Ana uygulama bileşeni
│   ├── index.js          # Entry point
│   ├── package.json      # Node.js bağımlılıkları
│   ├── app.json          # Expo konfigürasyonu
│   ├── eas.json          # EAS Build konfigürasyonu
│   └── README.md         # Mobile dokümantasyonu
│
└── README.md             # Bu dosya
```

##  Kurulum

### Backend Kurulumu

#### 1. Gereksinimler
- Python 3.11 veya üzeri
- pip (Python paket yöneticisi)

#### 2. Proje Klasörüne Geçin
```bash
cd backend
```

#### 3. Virtual Environment Oluşturun (Önerilen)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 4. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

Yüklenecek paketler:
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `python-multipart==0.0.6` - File upload desteği
- `numpy==1.26.2` - Numerik işlemler
- `opencv-python-headless==4.8.1.78` - Görüntü işleme
- `pillow==10.1.0` - Görüntü manipülasyonu
- `requests==2.31.0` - HTTP istekleri

#### 5. Roboflow API Yapılandırması
Backend zaten yapılandırılmış durumda. `main.py` dosyasında şu ayarlar mevcut:

```python
# Model 1: Phillips/Pozidriv
MODEL_1_URL = "https://serverless.roboflow.com/dataminingproject-avr2o/2"
MODEL_1_API_KEY = "rSNcCctYlXx2bkMecwZk"
MODEL_1_THRESHOLD = 0.65

# Model 2: Diğer vida tipleri
MODEL_2_URL = "https://serverless.roboflow.com/dataminingproject-avr2o/5"
MODEL_2_API_KEY = "rSNcCctYlXx2bkMecwZk"
MODEL_2_THRESHOLD = 0.55
```

### Mobil Uygulama Kurulumu

#### 1. Gereksinimler
- Node.js 18+ ve npm
- Expo CLI (global olarak kurulacak)
- Android Studio (Android geliştirme için)
- Xcode (iOS geliştirme için, sadece macOS)

#### 2. Expo CLI Kurulumu
```bash
npm install -g expo-cli
```

#### 3. Proje Klasörüne Geçin
```bash
cd mobile
```

#### 4. Bağımlılıkları Yükleyin
```bash
npm install
```

#### 5. API URL Yapılandırması
`app.json` dosyasını açın ve API URL'ini yapılandırın:

```json
{
  "expo": {
    "extra": {
      "apiUrl": "http://localhost:8000",  // Development için
      "apiUrlLocal": "http://localhost:8000"
    }
  }
}
```

**Not**: Development modunda uygulama otomatik olarak `localhost:8000` kullanır. Production için gerçek API URL'inizi ekleyin.

##  Çalıştırma

### Backend'i Çalıştırma

#### Yöntem 1: Uvicorn ile (Önerilen)
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` parametresi development modunda kod değişikliklerini otomatik yükler.

#### Yöntem 2: Python ile
```bash
cd backend
python main.py
```

#### Yöntem 3: Windows Batch Script
```bash
cd backend
start_backend.bat
```

#### Yöntem 4: PowerShell Script
```bash
cd backend
.\start_backend.ps1
```

Backend başarıyla çalıştığında şu mesajları göreceksiniz:
```
[OK] Model 1 hazir: https://serverless.roboflow.com/...
[OK] Model 2 hazir: https://serverless.roboflow.com/...
[OK] Cascade sistem aktif
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### API Test
Tarayıcınızda şu URL'leri ziyaret ederek API'nin çalıştığını doğrulayın:
- `http://localhost:8000` - API durumu
- `http://localhost:8000/docs` - Swagger UI (interaktif API dokümantasyonu)
- `http://localhost:8000/health` - Sağlık kontrolü

### Mobil Uygulamayı Çalıştırma

#### 1. Expo Go Uygulamasını İndirin
- **Android**: [Google Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)
- **iOS**: [App Store](https://apps.apple.com/app/expo-go/id982107779)

#### 2. Development Server'ı Başlatın
```bash
cd mobile
npm start
```

Bu komut Expo Development Server'ı başlatır ve QR kod gösterir.

#### 3. Uygulamayı Açın

**Android:**
- Expo Go uygulamasını açın
- QR kodu tarayın veya "Enter URL manually" ile `exp://localhost:8081` girin

**iOS:**
- Kamera uygulamasını açın
- QR kodu tarayın
- Açılan bildirime tıklayın

**Web:**
```bash
npm run web
```

#### 4. Alternatif Çalıştırma Yöntemleri
```bash
# Sadece Android
npm run android

# Sadece iOS (macOS gerekli)
npm run ios

# Web tarayıcısında
npm run web
```

##  API Dokümantasyonu

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. API Durumu
```http
GET /
```

**Response:**
```json
{
  "status": "active",
  "message": "ScrewTrue Roboflow API calisiyor",
  "version": "3.0.0",
  "runtime": "Roboflow API"
}
```

#### 2. Sağlık Kontrolü
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "api_configured": true,
  "api_type": "Roboflow Cascade",
  "model_1_url": "https://serverless.roboflow.com/...",
  "model_2_url": "https://serverless.roboflow.com/...",
  "model_1_allowed": ["Phillips", "Pozidriv"],
  "model_1_threshold": 0.65,
  "model_2_allowed": ["Torx", "Hex/Allen", ...],
  "model_2_threshold": 0.55,
  "classes": ["phillips", "pozidriv", ...]
}
```

#### 3. Desteklenen Sınıflar
```http
GET /classes
```

**Response:**
```json
{
  "classes": [
    {
      "id": 0,
      "name": "phillips",
      "label": "Phillips",
      "color": "#E74C3C"
    },
    ...
  ]
}
```

#### 4. Dosya Yükleme ile Tespit
```http
POST /detect
Content-Type: multipart/form-data
```

**Request:**
- `file`: Görüntü dosyası (form-data)
- `confidence`: (opsiyonel) Güven eşiği (varsayılan: 0.25)

**cURL Örneği:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@screw_image.jpg" \
  -F "confidence=0.25"
```

**Response:**
```json
{
  "success": true,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections_count": 1,
  "detections": [
    {
      "class_id": 0,
      "class_name": "phillips",
      "class_label": "Phillips",
      "confidence": 0.85,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 350
      },
      "color": "#E74C3C",
      "model": 1
    }
  ]
}
```

#### 5. Base64 Görüntü ile Tespit
```http
POST /detect/base64
Content-Type: application/json
```

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "confidence": 0.25
}
```

**cURL Örneği:**
```bash
curl -X POST "http://localhost:8000/detect/base64" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "confidence": 0.25
  }'
```

**Response:** (Yukarıdaki ile aynı format)

### Swagger UI
Interaktif API dokümantasyonu için:
```
http://localhost:8000/docs
```

##  Cascade Model Sistemi

ScrewTrue, iki aşamalı bir cascade model sistemi kullanır:

### Model 1: Phillips/Pozidriv Tespiti
- **URL**: `https://serverless.roboflow.com/dataminingproject-avr2o/2`
- **Sınıflar**: Phillips, Pozidriv
- **Threshold**: 0.65 (65% güven eşiği)
- **Öncelik**: İlk kontrol edilen model

### Model 2: Diğer Vida Tipleri
- **URL**: `https://serverless.roboflow.com/dataminingproject-avr2o/5`
- **Sınıflar**: Torx, Hex/Allen, Slotted, Security Torx, Pentalobe, Tri-wing, Spanner, Triangle
- **Threshold**: 0.55 (55% güven eşiği)
- **Öncelik**: Model 1'de sonuç bulunamazsa çalışır

### Cascade Akışı

```
1. Görüntü alınır
   ↓
2. Model 1 çağrılır (Phillips/Pozidriv)
   ↓
3. Sonuç filtrelenir (threshold: 0.65)
   ├─→ Başarılı → Sonuç döndürülür
   └─→ Başarısız → Model 2'ye geç
       ↓
4. Model 2 çağrılır (Diğer tipler)
   ↓
5. Sonuç filtrelenir (threshold: 0.55)
   ├─→ Başarılı → Sonuç döndürülür
   └─→ Başarısız → null döndürülür
```

### Avantajlar
-  **Hızlı Tespit**: Yaygın vida tipleri (Phillips/Pozidriv) için optimize edilmiş
-  **Yüksek Doğruluk**: Her model kendi uzmanlık alanında optimize edilmiş
-  **Maliyet Optimizasyonu**: Gerekli durumlarda sadece Model 2 çağrılır
-  **Güvenilirlik**: İki model katmanı ile daha güvenilir sonuçlar

## Deployment

### Backend Deployment

#### Railway
1. [Railway](https://railway.app) hesabına giriş yapın
2. "New Project" > "Deploy from GitHub repo" seçin
3. Repository'yi seçin
4. Root Directory: `backend` olarak ayarlayın
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Render
1. [Render](https://render.com) hesabına giriş yapın
2. "New Web Service" oluşturun
3. GitHub repository'yi bağlayın
4. Ayarlar:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: 3.11

#### Heroku
```bash
# Heroku CLI kurulumu gerekli
heroku login
heroku create screwtrue-api

# Backend'i deploy et
cd backend
git subtree push --prefix backend heroku main
```

### Mobil Uygulama Build

#### EAS Build Kurulumu
```bash
npm install -g eas-cli
eas login
```

#### Android APK Build
```bash
cd mobile

# Preview build (test için)
npm run build:android

# Production build
npm run build:android:prod
```

#### iOS Build (macOS gerekli)
```bash
cd mobile
eas build --platform ios --profile production
```

Build tamamlandıktan sonra EAS dashboard'dan APK/IPA dosyalarını indirebilirsiniz.

## 🛠️ Geliştirme

### Debug Modu

Backend'de debug logları aktif:
```python
[DEBUG] Calling Model 1 (Phillips/Pozidriv)...
[DEBUG] Model 1 - class: Phillips, conf: 0.85
[DEBUG] Model 1 ACCEPTED - class: Phillips, conf: 0.85
```

### Kod Yapısı

#### Backend (`backend/main.py`)
- `call_roboflow_api()`: Roboflow API çağrısı yapar
- `handle_model1_response()`: Model 1 sonuçlarını filtreler
- `handle_model2_response()`: Model 2 sonuçlarını filtreler
- `detect_screw_cascade()`: Cascade akışını yönetir
- `run_inference()`: Ana inference fonksiyonu

#### Mobile (`mobile/App.js`)
- Gerçek zamanlı kamera tespiti
- Galeri'den görüntü seçme
- API entegrasyonu
- Sonuç görselleştirme

### Test Etme

#### Backend Test
```bash
# API test
curl http://localhost:8000/health

# Görüntü testi
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg"
```

#### Mobile Test
- Expo Go ile development test
- EAS Build ile production test

##  Notlar

- Backend default olarak `localhost:8000` portunda çalışır
- Mobil uygulama development modunda otomatik olarak localhost'u bulur
- Production deployment'ta API URL'ini `app.json`'da güncelleyin
- Cascade sistem sayesinde aynı frame'de iki sonuç asla gösterilmez
- Model threshold'ları `main.py` içinde ayarlanabilir

##  Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

##  Lisans

Bu proje özel bir lisans altındadır.

##  İletişim

Sorularınız için issue açabilir veya repository'yi inceleyebilirsiniz.

---

**ScrewTrue** - Akıllı Vida Tanıma Sistemi 

