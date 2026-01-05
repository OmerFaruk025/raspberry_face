import cv2
import os
import time
import csv
from datetime import datetime
import numpy as np
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# --- AYARLAR ---
LAPTOP_IP = "192.168.1.47" 
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

# Dosyayı direkt projenin ana klasörüne (root) zorla kaydediyoruz
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")
# Dosya adını senin istediğin gibi "hakan_fidan.csv" yaptık kral
LOG_FILE_PATH = str(ROOT_DIR / "hakan_fidan.csv") 

# --- LOG SİSTEMİ (GARANTİCİ VERSİYON) ---
def log_activity(name, confidence):
    try:
        file_exists = os.path.isfile(LOG_FILE_PATH)
        # 'a' (append) modu: Yoksa oluşturur, varsa sonuna ekler
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Tarih', 'Saat', 'Tespit_Edilen', 'Eminlik_Orani'])
            
            now = datetime.now()
            writer.writerow([
                now.strftime("%d-%m-%Y"), 
                now.strftime("%H:%M:%S"), 
                name, 
                f"%{100-int(confidence)}" # Güven skorunu yüzdeye çevirdik (opsiyonel)
            ])
        return True
    except Exception as e:
        print(f"❌ Dosya Yazma Hatası: {e}")
        return False

# --- BAŞLAT ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

detector = FaceDetector()
cam = Camera(source=STREAM_URL)

last_seen_name = ""
last_seen_time = 0
wait_duration = 2 

print(f"🕵️‍♂️ İstihbarat Kaydı Başladı...")
print(f"📂 Dosya: {LOG_FILE_PATH}")

try:
    while True:
        ret, frame = cam.read()
        if not ret: continue

        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            label_id, confidence = recognizer.predict(gray_face)

            current_time = time.time()

            if confidence < 95:
                name = labels.get(label_id, "Bilinmeyen")
                
                if (current_time - last_seen_time > wait_duration) or (name != last_seen_name):
                    success = log_activity(name, confidence)
                    if success:
                        print(f"✅ {name.upper()} tespit edildi, dosyaya işlendi.")
                    last_seen_name = name
                    last_seen_time = current_time
            else:
                if current_time - last_seen_time > wait_duration:
                    log_activity("Süpheli Şahıs", confidence)
                    print("⚠️ Yabancı şahıs kayda alındı.")
                    last_seen_time = current_time
                    last_seen_name = "Yabancı"

except KeyboardInterrupt:
    print("\n🤐 Operasyon bitti, kayıtlar güvende.")
finally:
    cam.release()