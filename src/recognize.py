import cv2
import time
from collections import deque
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
MATCH_THRESHOLD = 60
COOLDOWN_SECONDS = 3
SCORE_BUFFER_SIZE = 5

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
last_print_time = 0   # spam print kontrolü

print("📸 Kamera hazır, tanıma aktif")

# -----------------------------
# ANA DÖNGÜ
# -----------------------------
try:
    while True:
        now = time.time()

        # ---- COOLDOWN ----
        if now - last_recognized_time < COOLDOWN_SECONDS:
            time.sleep(0.15)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.15)
            continue

        face_img, bbox = detector.detect_and_crop(frame)

        # ---- YÜZ YOK ----
        if face_img is None:
            if face_active:
                score_buffer.clear()
                face_active = False
            time.sleep(0.1)
            continue

        # ---- YÜZ İLK KEZ ----
        if not face_active:
            print("👤 Yüz algılandı")
            face_active = True
            score_buffer.clear()

        # ---- TANIMA ----
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        label_id, distance = recognizer.predict(gray)
        match_percent = max(0, 100 - distance)
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
            # spam engelle (0.7 sn'de 1 yaz)
            if now - last_print_time > 0.7:
                print(
                    f"❌ Tanınmadı | Tahmin: {name} | "
                    f"Benzerlik: %{round(avg_score, 1)}"
                )
                last_print_time = now

        time.sleep(0.08)

except KeyboardInterrupt:
    print("\n👋 Sistem kapatıldı")

finally:
    cam.release()
