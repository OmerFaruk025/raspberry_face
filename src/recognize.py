print("🔥 recognize.py başladı")

import cv2
import time
import csv
from collections import deque
from pathlib import Path
from datetime import datetime
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
MATCH_THRESHOLD = 60
COOLDOWN_SECONDS = 3

SCORE_BUFFER_SIZE = 5
BBOX_BUFFER_SIZE = 5

MIN_FACE_AREA_RATIO = 0.08   # frame'in %8'inden küçükse çöpe at
ASPECT_RATIO_RANGE = (0.75, 1.35)

LOG_FILE = "hakan_fidan.csv"

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "lbph_model.yml"
LABEL_PATH = ROOT_DIR / "labels.txt"

# -----------------------------
# MODEL & LABEL
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

print(f"✅ Model yüklendi | Kişi sayısı: {len(labels)}")

# -----------------------------
# NESNELER
# -----------------------------
cam = Camera()
detector = FaceDetector()

score_buffer = deque(maxlen=SCORE_BUFFER_SIZE)
bbox_buffer = deque(maxlen=BBOX_BUFFER_SIZE)

last_recognized_time = 0
face_active = False

print("📸 Kamera hazır, tanıma aktif")

# -----------------------------
# CSV HAZIRLIK
# -----------------------------
if not Path(LOG_FILE).exists():
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["isim", "tarih", "saat", "benzerlik"])

# -----------------------------
# ANA DÖNGÜ
# -----------------------------
try:
    while True:
        now = time.time()

        # ---- COOLDOWN ----
        if now - last_recognized_time < COOLDOWN_SECONDS:
            time.sleep(0.2)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]

        face_img, bbox = detector.detect_and_crop(frame, return_bbox=True)

        if bbox is None:
            face_active = False
            bbox_buffer.clear()
            score_buffer.clear()
            continue

        x, y, bw, bh = bbox
        area_ratio = (bw * bh) / (w * h)
        aspect_ratio = bw / bh if bh != 0 else 0

        # ---- SAÇMA YÜZLERİ ELE ----
        if (
            area_ratio < MIN_FACE_AREA_RATIO or
            not ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]
        ):
            continue

        bbox_buffer.append(bbox)

        # ---- STABIL BBOX ----
        avg_x = int(sum(b[0] for b in bbox_buffer) / len(bbox_buffer))
        avg_y = int(sum(b[1] for b in bbox_buffer) / len(bbox_buffer))
        avg_w = int(sum(b[2] for b in bbox_buffer) / len(bbox_buffer))
        avg_h = int(sum(b[3] for b in bbox_buffer) / len(bbox_buffer))

        face_crop = frame[avg_y:avg_y+avg_h, avg_x:avg_x+avg_w]

        if face_crop.size == 0:
            continue

        if not face_active:
            print("👤 Yüz algılandı")
            face_active = True

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        label_id, confidence = recognizer.predict(gray)
        match_percent = max(0, int(100 - confidence))
        name = labels.get(label_id, "Bilinmeyen")

        score_buffer.append(match_percent)
        avg_score = sum(score_buffer) / len(score_buffer)

        if avg_score >= MATCH_THRESHOLD:
            print(f"✅ TANINDI → {name.upper()} | Benzerlik: %{round(avg_score,1)}")

            # ---- CSV LOG ----
            now_dt = datetime.now()
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    now_dt.strftime("%Y-%m-%d"),
                    now_dt.strftime("%H:%M:%S"),
                    round(avg_score, 1)
                ])

            last_recognized_time = time.time()
            face_active = False
            score_buffer.clear()
            bbox_buffer.clear()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 Sistem kapatıldı")

finally:
    cam.release()
    print("✅ Kamera serbest bırakıldı")