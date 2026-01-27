import cv2
import time
import csv
from collections import deque
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector
import threading

# -----------------------------
# AYARLAR
# -----------------------------
CONFIDENCE_THRESHOLD = 65
COOLDOWN_SECONDS = 2          # ⬅️ TANINDIKTAN SONRA BEKLEME
CONF_BUFFER_SIZE = 5
UNRECOGNIZED_PRINT_DELAY = 0.75  # ⬅️ 750 ms

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "lbph_model.yml"
LABEL_PATH = ROOT_DIR / "labels.txt"
LOG_PATH   = ROOT_DIR / "hakan_fidan.csv"

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

print(f"✅ Model yüklendi | Kişi: {len(labels)}")

# -----------------------------
cam = Camera()
detector = FaceDetector()

conf_buffer = deque(maxlen=CONF_BUFFER_SIZE)
last_recognized_time = 0
last_unrecognized_print = 0
face_active = False

print("📸 Kamera hazır, tanıma aktif")

# -----------------------------
# RUNNING FLAG (WEB PANEL KONTROLÜ İÇİN)
RUNNING = True
RUNNING_LOCK = threading.Lock()  # thread-safe kontrol

# -----------------------------
try:
    while True:
        with RUNNING_LOCK:
            if not RUNNING:
                # Sistem durdurulmuşsa beklemede kal
                time.sleep(0.1)
                continue

        now = time.time()

        if now - last_recognized_time < COOLDOWN_SECONDS:
            time.sleep(0.15)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        face_img, _ = detector.detect_and_crop(frame, return_bbox=True)
        if face_img is None:
            face_active = False
            conf_buffer.clear()
            continue

        if not face_active:
            print("👤 Yüz algılandı")
            face_active = True

        # -------------------------
        # PREPROCESS (TRAIN İLE AYNI)
        # -------------------------
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        label_id, confidence = recognizer.predict(gray)
        name = labels.get(label_id, "Bilinmeyen")

        conf_buffer.append(confidence)
        avg_conf = sum(conf_buffer) / len(conf_buffer)

        # -------------------------
        # KARAR
        # -------------------------
        if avg_conf <= CONFIDENCE_THRESHOLD:
            print(
                f"✅ TANINDI → {name.upper()} | "
                f"Confidence: {round(avg_conf, 1)}"
            )

            # ---- CSV LOG ----
            timestamp = time.strftime("%d.%m.%Y %H:%M:%S")
            with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name])

            last_recognized_time = time.time()
            face_active = False
            conf_buffer.clear()

        else:
            if now - last_unrecognized_print >= UNRECOGNIZED_PRINT_DELAY:
                print(
                    f"❌ Tanınmadı | Tahmin: {name} | "
                    f"Confidence: {round(avg_conf, 1)}"
                )
                last_unrecognized_print = now

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 Sistem kapatıldı")

finally:
    cam.release()
    print("📷 Kamera kapatıldı")
