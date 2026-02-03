import cv2
import time
import csv
from collections import deque
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector
import RPi.GPIO as GPIO

# -----------------------------
# AYARLAR
CONFIDENCE_THRESHOLD = 65
COOLDOWN_SECONDS = 2
CONF_BUFFER_SIZE = 5
UNRECOGNIZED_PRINT_DELAY = 0.75

# Röle ayarları
RELAY_PIN = 17          # S pinine bağlı GPIO
RELAY_OPEN_TIME = 1.0   # saniye

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "lbph_model.yml"
LABEL_PATH = ROOT_DIR / "labels.txt"
LOG_PATH   = ROOT_DIR / "hakan_fidan.csv"

# -----------------------------
# GPIO SETUP
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # röle kapalı başlasın

# -----------------------------
# MODEL & LABEL
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

print(f"✅ Model yüklendi | Kişi: {len(labels)}")

# -----------------------------
# CAMERA & DETECTOR
cam = Camera()
detector = FaceDetector()

conf_buffer = deque(maxlen=CONF_BUFFER_SIZE)
last_recognized_time = 0
last_unrecognized_print = 0
face_active = False

print("📸 Kamera hazır, tanıma aktif")

# -----------------------------
# RUNNING FLAG (web için)
RUNNING = [True]   # standalone modda True
FRAME_QUEUE = None # web panel için kullanılacak

# -----------------------------
def open_door():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    time.sleep(RELAY_OPEN_TIME)
    GPIO.output(RELAY_PIN, GPIO.HIGH)

# -----------------------------
def run(frame_queue=None, running=None):
    global FRAME_QUEUE, RUNNING
    if frame_queue is not None:
        FRAME_QUEUE = frame_queue
    if running is not None:
        RUNNING = running

    global last_recognized_time, last_unrecognized_print, face_active

    while True:
        # Web modda start/stop kontrolü
        if RUNNING is not None and isinstance(RUNNING, list):
            if not RUNNING[0]:
                time.sleep(0.1)
                continue

        now = time.time()

        if now - last_recognized_time < COOLDOWN_SECONDS:
            time.sleep(0.05)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # Web modda frame paylaşımı
        if FRAME_QUEUE is not None:
            if FRAME_QUEUE.full():
                try:
                    FRAME_QUEUE.get_nowait()
                except:
                    pass
            FRAME_QUEUE.put(frame)

        # -------------------------
        face_img, _ = detector.detect_and_crop(frame, return_bbox=True)
        if face_img is None:
            face_active = False
            conf_buffer.clear()
            continue

        if not face_active:
            face_active = True
            print("👤 Yüz algılandı")

        # -------------------------
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3,3),0)

        label_id, confidence = recognizer.predict(gray)
        name = labels.get(label_id, "Bilinmeyen")

        conf_buffer.append(confidence)
        avg_conf = sum(conf_buffer) / len(conf_buffer)

        if avg_conf <= CONFIDENCE_THRESHOLD:
            print(f"✅ TANINDI → {name.upper()} | Confidence: {round(avg_conf,1)}")
            timestamp = time.strftime("%d.%m.%Y %H:%M:%S")

            with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([timestamp, name])

            # 🔓 KAPIYI AÇ
            open_door()

            last_recognized_time = time.time()
            face_active = False
            conf_buffer.clear()

        else:
            if now - last_unrecognized_print >= UNRECOGNIZED_PRINT_DELAY:
                print(f"❌ Tanınmadı | Tahmin: {name} | Confidence: {round(avg_conf,1)}")
                last_unrecognized_print = now

        time.sleep(0.05)

# -----------------------------
# Standalone mod
if __name__ == "__main__":
    try:
        run()
    finally:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        GPIO.cleanup()
        print("GPIO temizlendi, çıkılıyor.")