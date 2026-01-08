import cv2
import os
import time
import csv
from datetime import datetime
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv")

def log_activity(name, percent):
    try:
        file_exists = os.path.isfile(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Tarih', 'Saat', 'Isim', 'Eminlik'])
            now = datetime.now()
            writer.writerow([now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S"), name, f"%{percent}"])
    except: pass

# Modeli ve Etiketleri Yukle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) == 2: labels[int(parts[0])] = parts[1]

detector = FaceDetector()
cam = Camera()
last_logged = ["", 0] # [isim, zaman]

print("\n🕵️‍♂️ TANIMLAMA BASLADI...")

try:
    while True:
        ret, frame = cam.read()
        if not ret: continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)
            
            match_percent = int(max(0, 100 - confidence))
            name = labels.get(label_id, "Bilinmeyen")
            
            if match_percent >= 45:
                if not (name == last_logged[0] and (time.time() - last_logged[1] < 6)):
                    print(f"✅ HOS GELDIN: {name.upper()} (%{match_percent})")
                    log_activity(name, match_percent)
                    last_logged = [name, time.time()]
            else:
                print(f"🔍 Bilinmeyen Sahis (%{match_percent})", end="\r")
        else:
            print("🔍 Yuz araniyor...", end="\r")
        
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n👋 Kapatildi.")
finally:
    cam.release()