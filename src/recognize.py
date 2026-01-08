import cv2
import time
import csv
import os
from datetime import datetime
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

cam = Camera()
detector = FaceDetector()

last_logged_person = ""
last_logged_time = 0
COOLDOWN_TIME = 6


def log_activity(name, percent):
    file_exists = os.path.isfile(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Tarih", "Saat", "Isim", "Eminlik"])
        now = datetime.now()
        writer.writerow([
            now.strftime("%d-%m-%Y"),
            now.strftime("%H:%M:%S"),
            name,
            f"%{percent}"
        ])


print("🟢 Tanıma aktif")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        face_img, _ = detector.detect_and_crop(frame)

        if face_img is not None:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            label_id, confidence = recognizer.predict(gray)

            match = int(max(0, 100 - confidence))
            name = labels.get(label_id, "Bilinmeyen")
            now = time.time()

            if match >= 60:
                if name != last_logged_person or (now - last_logged_time) > COOLDOWN_TIME:
                    print(f"✅ {name} (%{match})")
                    log_activity(name, match)
                    last_logged_person = name
                    last_logged_time = now
                    time.sleep(3)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("👋 Çıkıldı")
finally:
    cam.release()
