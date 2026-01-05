import cv2
import os
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

# Pi'deki standart Türkçe destekli font yolu
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 

# --- BAŞLAT ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

detector = FaceDetector()
cam = Camera(source=STREAM_URL) # <--- ARTIK AĞDAN ALIYOR

print("🚀 Pi Tanıma Sistemi Başladı...")

try:
    while True:
        ret, frame = cam.read()
        if not ret: continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)

            if confidence < 95:
                name = labels.get(label_id, "Bilinmeyen")
                print(f"✅ TANINDI: {name} (Güven: {int(confidence)})")
            else:
                print("👤 Yabancı birisi var...")
except KeyboardInterrupt:
    print("\n👋 Sistem kapatılıyor.")
finally:
    cam.release()