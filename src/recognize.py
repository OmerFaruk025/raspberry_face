import cv2
import os
import time
import csv
from datetime import datetime
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
# Laptop IP bağımlılığı kaldırıldı, yerel kamera (0) aktif!
CAMERA_SOURCE = 0

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv")

last_logged_person = ""
last_logged_time = 0
COOLDOWN_TIME = 6 # Aynı kişi için 6 saniye bekleme

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
        print(f"❌ Log Hatası: {e}")

# --- BAŞLAT ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

detector = FaceDetector()

# Doğrudan PiCam V2.1 üzerinden başlatıyoruz
print(f"🕵️‍♂️ Tanımlama Aktif... (Kaynak: PiCam)")
cam = Camera(source=CAMERA_SOURCE)

try:
    while True:
        ret, frame = cam.read()
        if not ret: 
            continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)
            
            match_percent = int(max(0, 100 - confidence))
            name = labels.get(label_id, "Bilinmeyen")
            current_time = time.time()

            # --- MANTIK KATMANI ---
            if match_percent >= 60:
                if name == last_logged_person and (current_time - last_logged_time < COOLDOWN_TIME):
                    pass 
                else:
                    print(f"✅ GİRİŞ YAPILDI: {name.upper()} (%{match_percent})")
                    log_activity(name, match_percent)
                    last_logged_person = name
                    last_logged_time = current_time
                    
                    print("⏱️  3 saniye bekleniyor...")
                    time.sleep(3) # Fiziksel bekleme (Saniyede 5 giriş engeli)
            
            elif 30 <= match_percent < 60:
                # İstediğin % raporlaması burada kanka
                print(f"🔍 {name.upper()} kişisinden emin olunuyor: %{match_percent}")
        
        # CPU'yu korumak için kısa mola
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n👋 Sistem operasyonel olarak kapatıldı.")
finally:
    cam.release()