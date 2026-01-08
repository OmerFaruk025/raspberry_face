import cv2
import os
import time
import shutil
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
FACES_PATH = DATA_PATH / "faces"
FACES_PATH.mkdir(parents=True, exist_ok=True)

def collect_data(user_name, mode="ekle"):
    user_dir = FACES_PATH / user_name
    if mode == "guncelle" and user_dir.exists():
        shutil.rmtree(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    cam = Camera()
    detector = FaceDetector()
    count, max_count = 0, 50
    
    print(f"🚀 KAYIT BASLIYOR: {user_name}")
    
    try:
        while count < max_count:
            ret, frame = cam.read()
            if not ret: continue

            face_img, bbox = detector.detect_and_crop(frame)

            if bbox is not None:
                count += 1
                img_path = str(user_dir / f"{user_name}_{count}.jpg")
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                _, buffer = cv2.imencode('.jpg', gray_face)
                with open(img_path, 'wb') as f:
                    f.write(buffer)
                
                print(f"📸 {count}/{max_count} - Yuz yakalandi!")
            else:
                print("🔍 Yuz araniyor... (Kameraya odaklan)", end="\r")
                time.sleep(0.01) # Maksimum hız için mola süresi düştü
    finally:
        cam.release()
        print(f"\n✅ {user_name} icin {count} adet veri toplandi.")

if __name__ == "__main__":
    name = input("Kaydedilecek Isim: ").strip()
    if name: collect_data(name)