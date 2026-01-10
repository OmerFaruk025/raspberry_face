import cv2
import time
from collections import deque
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
MATCH_THRESHOLD = 60        # Kabul eşiği
COOLDOWN_SECONDS = 3        # Tanıma sonrası bekleme
SCORE_BUFFER_SIZE = 5       # Ortalama skor için frame sayısı

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
last_recognized_time = 0
face_active = False

print("📸 Kamera hazır, tanıma aktif")

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
            time.sleep(0.2)
            continue

        # 🔴 KRİTİK NOKTA – HATAYI ÇÖZEN SATIR
        result = detector.detect_and_crop(frame, return_bbox=True)

        # FaceDetector hiçbir şey bulamazsa None döner
        if result is None or result[0] is None:
            face_active = False
            score_buffer.clear()
            time.sleep(0.1)
            continue

        face_img, bbox = result

        # ---- YÜZ İLK KEZ ALGILANDI ----
        if not face_active:
            print("👤 Yüz algılandı")
            face_active = True

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
                f"✅ TANINDI → {name.upper()} | "
                f"Benzerlik: %{round(avg_score, 1)}"
            )
            last_recognized_time = time.time()
            face_active = False
            score_buffer.clear()
        else:
            print(
                f"❌ Tanınmadı | Tahmin: {name} | "
                f"Benzerlik: %{round(avg_score, 1)}"
            )

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 Sistem kapatıldı")

finally:
    cam.release()
