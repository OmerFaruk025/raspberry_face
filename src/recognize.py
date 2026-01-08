import cv2
import os
import time
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv")

COOLDOWN_TIME = 6 
last_logged_person = ""
last_logged_time = 0

def log_activity(name, percent):
    try:
        file_exists = os.path.isfile(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Tarih', 'Saat', 'Isim', 'Eminlik'])
            now = datetime.now()
            writer.writerow([now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S"), name, f"%{percent}"])
    except Exception as e:
        print(f"❌ Log Yazma Hatasi: {e}")

# --- KONTROL MERKEZI ---
print("🔍 Sistem kontrolleri yapiliyor...")

if not os.path.exists(MODEL_PATH):
    print("❌ KRITIK HATA: 'lbph_model.yml' bulunamadi! Once egitim yapmalisin kanka.")
    exit()

if not os.path.exists(LABEL_PATH):
    print("❌ KRITIK HATA: 'labels.txt' bulunamadi!")
    exit()

# Modeli Yukle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Etiketleri Yukle
labels = {}
try:
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    print(f"✅ {len(labels)} kisi tanimlama icin yuklendi.")
except Exception as e:
    print(f"❌ Etiket okunurken hata: {e}")
    exit()

detector = FaceDetector()
cam = Camera()

print("\n" + "="*40)
print("🕵️‍♂️ TANIMLAMA AKTIF (rpicam-apps modu)")
print("="*40)

try:
    while True:
        # 1. Adim: Goruntu Al
        ret, frame = cam.read()
        if not ret or frame is None:
            print("⚠️ Kamera karesi bos geldi, tekrar deneniyor...")
            continue

        # 2. Adim: Yuz Tespiti
        # Buradaki detector senin face_detect.py içindeki detect_and_crop fonksiyonun
        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            print("👤 Bir yuz yakalandi! Analiz ediliyor...")
            
            # 3. Adim: Isleme
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            
            # 4. Adim: Tahmin
            label_id, confidence = recognizer.predict(gray_face)
            
            # LBPH'de mesafe (confidence) ne kadar dusukse o kadar benzerdir
            match_percent = int(max(0, 100 - confidence))
            name = labels.get(label_id, "Bilinmeyen")
            current_time = time.time()

            # 5. Adim: Karar Mekanizmasi
            if match_percent >= 45: # Esik degeri (Threshold)
                if name == last_logged_person and (current_time - last_logged_time < COOLDOWN_TIME):
                    # Cooldown aktifse sadece terminalde guncelleme yap, loglama
                    print(f"ℹ️ {name.upper()} hala kamerada... (%{match_percent})", end="\r")
                else:
                    print(f"\n✅ ERISIM ONAYLANDI: {name.upper()} (%{match_percent})")
                    log_activity(name, match_percent)
                    last_logged_person = name
                    last_logged_time = current_time
            else:
                print(f"🔍 Bilinmeyen Sahis veya Dusuk Eminlik (%{match_percent})")
        
        else:
            # Yuz bulunamadigi her saniye terminale bir nokta koyalim (canli miyiz gorelim)
            print("🔍 Yuz aranıyor... (Kamera aktif)", end="\r")
        
        # CPU'yu korumak ve rpicam-still'e nefes aldirmak icin
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n👋 Operasyon durduruldu. Gorusuruz kanka!")
finally:
    cam.release()