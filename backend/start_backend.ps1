# Backend başlatma scripti (PowerShell)
Write-Host "ScrewTrue Backend başlatılıyor..." -ForegroundColor Green
py -m uvicorn main:app --host 0.0.0.0 --port 8000

