import cv2
import os
import time
import csv
from datetime import datetime
import numpy as np
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
LAPTOP_IP = "192.168.1.47" 
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv")

# --- LOG SİSTEMİ ---
def log_activity(name, percent):
    try:
        file_exists = os.path.isfile(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Tarih', 'Saat', 'Isim', 'Eminlik_Yuzdesi'])
            
            now = datetime.now()
            writer.writerow([
                now.strftime("%d-%m-%Y"), 
                now.strftime("%H:%M:%S"), 
                name, 
                f"%{percent}"
            ])
        return True
    except Exception as e:
        print(f"❌ Log Hatası: {e}")
        return False

# --- BAŞLAT ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

detector = FaceDetector()
cam = Camera(source=STREAM_URL)

last_logged_name = ""
last_logged_time = 0
log_wait_duration = 10  # Aynı kişi için 10 saniyede bir log tutsun ki dosya şişmesin

print("🕵️‍♂️ Gelişmiş Tanıma & Analiz Sistemi Aktif...")

try:
    while True:
        ret, frame = cam.read()
        if not ret: continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)

            # Güven skorunu yüzdeye çevir (LBPH'da düşük confidence = yüksek başarı)
            # 0 çok iyi, 100 çok kötü. Biz bunu 100 üzerinden ters çeviriyoruz.
            match_percent = int(max(0, 100 - confidence))
            name = labels.get(label_id, "Bilinmeyen")
            
            current_time = time.time()

            # --- MANTIK KATMANI ---
            if match_percent >= 65:
                # KESİN TANIMA: %65 ve üzeri
                print(f"✅ GİRİŞ YAPILDI: {name.upper()} (%{match_percent})")
                
                # Sadece farklı biriyse veya üzerinden 10 saniye geçtiyse logla
                if (name != last_logged_name) or (current_time - last_logged_time > log_wait_duration):
                    log_activity(name, match_percent)
                    last_logged_name = name
                    last_logged_time = current_time
            
            elif 30 <= match_percent < 65:
                # ŞÜPHE AŞAMASI: %30-%64 arası
                print(f"🔍 Kişiden emin olunuyor... ({name} - %{match_percent})")
                # Log yazmıyoruz, sadece terminalde takip ediyoruz
            
            else:
                # %30 ALTI: Tanınmıyor
                if current_time - last_logged_time > 5:
                    print("👤 Yabancı şahıs analizi yapılıyor...")
                    last_logged_time = current_time

except KeyboardInterrupt:
    print("\n👋 Sistem kapatıldı.")
finally:
    cam.release()