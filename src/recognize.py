import cv2
import time
from collections import deque
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
LAPTOP_IP = "192.168.1.47"
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "lbph_model.yml"
LABEL_PATH = ROOT_DIR / "labels.txt"

MATCH_THRESHOLD = 60        # Kabul eşiği
COOLDOWN_SECONDS = 3        # Kabul sonrası bekleme
SCORE_BUFFER_SIZE = 5       # Ortalama için kaç frame

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

print("✅ Model yüklendi | Kişi sayısı:", len(labels))

# -----------------------------
# NESNELER
# -----------------------------
detector = FaceDetector()
cam = Camera(source=STREAM_URL)

score_buffer = deque(maxlen=SCORE_BUFFER_SIZE)
last_recognized_time = 0

print("📸 Kamera başladı (SSH uyumlu)")

# -----------------------------
# ANA DÖNGÜ
# -----------------------------
try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("⚠️ Kamera görüntüsü yok")
            time.sleep(0.2)
            continue

        now = time.time()

        # ---- COOLDOWN ----
        if now - last_recognized_time < COOLDOWN_SECONDS:
            print("⏳ Cooldown aktif – yüz okunmuyor")
            time.sleep(0.2)
            continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is None:
            score_buffer.clear()
            print("👤 Yüz YOK")
            time.sleep(0.1)
            continue

        print("👤 Yüz VAR")

        # ---- TANIMA ----
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        label_id, confidence = recognizer.predict(gray)
        match_percent = max(0, int(100 - confidence))
        name = labels.get(label_id, "Bilinmeyen")

        score_buffer.append(match_percent)
        avg_score = sum(score_buffer) / len(score_buffer)

        # ---- KARAR ----
        if avg_score >= MATCH_THRESHOLD:
            print(
                f"✅ KABUL → {name.upper()} | "
                f"Anlık: %{match_percent} | "
                f"Ortalama: %{round(avg_score,1)}"
            )
            last_recognized_time = now
            score_buffer.clear()
        else:
            print(
                f"❌ RED | Tahmin: {name} | "
                f"Anlık: %{match_percent} | "
                f"Ortalama: %{round(avg_score,1)}"
            )

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 Sistem kapatıldı")

finally:
    cam.release()
