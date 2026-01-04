# RecyclEye Mobil Uygulama

Expo tabanlı React Native mobil uygulama.

## 🚀 Hızlı Başlangıç

### Development

```bash
npm install
npm start
```

### APK Build

Detaylı bilgi için [BUILD.md](./BUILD.md) dosyasına bakın.

```bash
# EAS CLI kurulumu
npm install -g eas-cli

# Giriş yap
eas login

# Build al
npm run build:android
```

## ⚙️ Yapılandırma

### API URL

`app.json` dosyasındaki `extra.apiUrl` değerini güncelleyin:

```json
{
  "expo": {
    "extra": {
      "apiUrl": "https://your-deployed-api-url.com"
    }
  }
}
```

Development modunda otomatik olarak `localhost:8000` kullanılır.

## 📦 Build Profilleri

- **preview**: Test APK (internal distribution)
- **production**: Production APK

## 📱 Özellikler

- Gerçek zamanlı nesne tespiti
- Kamera ve galeri desteği
- Güven skoru gösterimi
- Renk kodlu sınıflandırma

