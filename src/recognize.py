import cv2
import os
import time
import csv
from datetime import datetime
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv")

COOLDOWN_TIME = 6 
last_logged_person = ""
last_logged_time = 0

def log_activity(name, percent):
    try:
        file_exists = os.path.isfile(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Tarih', 'Saat', 'Isim', 'Eminlik'])
            now = datetime.now()
            writer.writerow([now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S"), name, f"%{percent}"])
    except Exception as e:
        print(f"❌ Log Hatasi: {e}")

# --- MODEL YUKLE ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists(MODEL_PATH):
    print("❌ HATA: Model bulunamadi! Once egitim yap kanka.")
    exit()

recognizer.read(MODEL_PATH)
labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) == 2:
            idx, name = parts
            labels[int(idx)] = name

detector = FaceDetector()
cam = Camera() # Yeni nesil rpicam kamerasını başlatır

print("🕵️‍♂️ Tanimlama Aktif... (rpicam-apps modu)")

try:
    while True:
        # --- DİKKAT: cam.cap.grab() SATIRI SİLİNDİ ---
        # rpicam-still zaten her seferinde taze kare çeker.
        
        ret, frame = cam.read()
        if not ret or frame is None: 
            continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)
            
            # LBPH'de confidence (mesafe) düştükçe benzerlik artar
            match_percent = int(max(0, 100 - confidence))
            name = labels.get(label_id, "Bilinmeyen")
            current_time = time.time()

            if match_percent >= 45: # Eşik değerini (threshold) biraz esnettim
                if not (name == last_logged_person and (current_time - last_logged_time < COOLDOWN_TIME)):
                    print(f"✅ GIRIS: {name.upper()} (%{match_percent})")
                    log_activity(name, match_percent)
                    last_logged_person = name
                    last_logged_time = current_time
            else:
                print(f"🔍 Tanimlanamayan yüz (Eminlik: %{match_percent})")
        
        # rpicam-still biraz yavaş olduğu için buradaki sleep'i küçülttük
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n👋 Sistem operasyonel olarak kapatildi.")
finally:
    cam.release()