import cv2
import os
import time
import csv # <--- Log yazmak için
from datetime import datetime # <--- Tarih ve saat için
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
LAPTOP_IP = "192.168.1.47" 
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "activity_log.csv") # <--- Log dosyasının yolu

# --- LOG SİSTEMİ FONKSİYONU ---
def log_activity(name, confidence):
    """Tanınan kişiyi tarih ve saatle CSV dosyasına kaydeder."""
    file_exists = os.path.isfile(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Dosya yeni oluşturuluyorsa başlıkları ekle
        if not file_exists:
            writer.writerow(['Tarih', 'Saat', 'Isim', 'Guven_Skoru'])
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        writer.writerow([date_str, time_str, name, int(confidence)])

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

last_seen_name = ""
last_seen_time = 0
wait_duration = 2 

print(f"🚀 Pi Tanıma & Log Sistemi Başladı...")
print(f"📝 Kayıtlar '{LOG_FILE_PATH}' dosyasına yazılıyor.")

try:
    while True:
        ret, frame = cam.read()
        if not ret: continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)

            current_time = time.time()

            if confidence < 95:
                name = labels.get(label_id, "Bilinmeyen")
                
                if (current_time - last_seen_time > wait_duration) or (name != last_seen_name):
                    print(f"✅ TANINDI: {name.upper()} - Log kaydedildi.")
                    log_activity(name, confidence) # <--- Log kaydını yap
                    last_seen_name = name
                    last_seen_time = current_time
            else:
                if current_time - last_seen_time > wait_duration:
                    print("👤 Yabancı biri görüldü - Log kaydedildi.")
                    log_activity("Yabanci", confidence)
                    last_seen_time = current_time
                    last_seen_name = "Yabancı"

except KeyboardInterrupt:
    print("\n👋 Defter kapatıldı, sistem durdu.")
finally:
    cam.release()