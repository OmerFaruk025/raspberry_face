import cv2
import os
import time  # <--- Bekleme için lazım
import numpy as np
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR & YOLLAR
# -----------------------------
SOURCE = 0 
user_name = input("Kanka kimin yüzünü kaydediyoruz? (İsim gir): ").strip()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / user_name
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# SİSTEMİ BAŞLAT
# -----------------------------
cam = Camera(source=SOURCE)
detector = FaceDetector()

count = 0
max_count = 50 

print(f"📸 Kayıt başlıyor! Her kare arasında 0.2 saniye bekleyeceğim.")
print("Kanka kafanı hafif hafif sağa, sola, yukarı, aşağı oynatmayı unutma!")

while count < max_count:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    face_img, bbox = detector.detect_and_crop(frame)

    if bbox is not None:
        x, y, w, h = bbox
        
        count += 1
        img_filename = f"{user_name}_{count}.jpg"
        img_path = str(DATA_DIR / img_filename)
        
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Karakter geçirmez kayıt yöntemi
        _, buffer = cv2.imencode('.jpg', gray_face)
        with open(img_path, 'wb') as f:
            f.write(buffer)
        
        print(f"🚀 [{count}/{max_count}] Kaydedildi. Poz değiştir!")

        # Ekranda geri bildirim
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Çekim anında mavi kutu
        cv2.putText(frame, f"FOTO CEKILDI: {count}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # --- BURASI KRİTİK: BEKLEME SÜRESİ ---
        # 0.2 saniye idealdir (Saniyede 5 fotoğraf çeker). 
        # Eğer hala çok hızlı dersen bu sayıyı 0.5 yapabilirsin.
        cv2.imshow("Kayıt Ekranı", frame)
        cv2.waitKey(200) # 200 milisaniye bekle
    else:
        cv2.imshow("Kayıt Ekranı", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n🥳 Klasör doldu kral! Şimdi train_lbph.py'yi çalıştırabilirsin.")
cam.release()
cv2.destroyAllWindows()