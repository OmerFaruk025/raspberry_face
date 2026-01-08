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

def collect_data(user_name, mode="ekle"):
    user_dir = FACES_PATH / user_name
    if mode == "guncelle" and user_dir.exists():
        shutil.rmtree(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    cam = Camera()
    detector = FaceDetector()
    count = 0
    max_count = 50 
    
    print(f"🚀 Kayit basliyor: {user_name}")

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
                
                print(f"📸 Fotoğraf {count}/{max_count} kaydedildi.")
            else:
                # Yüz bulamazsa terminalde küçük bir uyarı
                print("🔍 Yüz aranıyor...", end="\r")
                
    finally:
        cam.release()
        print(f"✅ İşlem tamamlandı: {user_name}")

if __name__ == "__main__":
    name = input("İsim: ").strip()
    if name: collect_data(name)