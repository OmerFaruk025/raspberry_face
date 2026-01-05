import cv2
import os
import time  # <--- Zaman kontrolü için şart
import numpy as np
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
LAPTOP_IP = "192.168.1.47" 
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")

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

# --- TAKİP DEĞİŞKENLERİ ---
last_seen_name = ""
last_seen_time = 0
wait_duration = 2  # 2 saniye bekleme süresi

print("🚀 Pi Tanıma Sistemi Başladı (2 Saniye Gecikmeli)...")

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
                
                # Eğer 2 saniye geçtiyse VEYA farklı birini gördüyse yazdır
                if (current_time - last_seen_time > wait_duration) or (name != last_seen_name):
                    print(f"✅ TANINDI: {name.upper()} (Güven: {int(confidence)})")
                    last_seen_name = name
                    last_seen_time = current_time
            else:
                # Tanınmayan biri olduğunda da sürekli yazmasın diye kontrol
                if current_time - last_seen_time > wait_duration:
                    print("👤 Yabancı birisi var...")
                    last_seen_time = current_time
                    last_seen_name = "Yabancı"

except KeyboardInterrupt:
    print("\n👋 Eyvallah kral, sistem kapatılıyor.")
finally:
    cam.release()