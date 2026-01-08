import cv2
import os
import time
import numpy as np
import shutil
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR & KONFIGURASYON
# -----------------------------
# rpicam-apps kullandigimiz icin CAMERA_SOURCE indexi onemli degil
# Sistem otomatik olarak rpicam-still komutunu tetikler.
SHOW_DISPLAY = False 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
FACES_PATH = DATA_PATH / "faces"

FACES_PATH.mkdir(parents=True, exist_ok=True)

def get_registered_users():
    """data/faces altindaki kisileri listeler."""
    return [d.name for d in FACES_PATH.iterdir() if d.is_dir()]

def collect_data(user_name, mode="ekle"):
    user_dir = FACES_PATH / user_name
    
    if mode == "guncelle":
        print(f"🔄 '{user_name}' verileri temizleniyor...")
        if user_dir.exists():
            shutil.rmtree(user_dir)
    
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # rpicam destekli yeni Camera sinifi baslatiliyor
    cam = Camera() 
    detector = FaceDetector()
    
    count = 0
    max_count = 50 
    
    print(f"🚀 Kayit basliyor: {user_name}")
    print("💡 İpucu: rpicam-still her karede netleme yapabilir, sabit dur kanka!")
    time.sleep(2)

    try:
        while count < max_count:
            # rpicam-still ile taze kare cekiliyor
            ret, frame = cam.read()
            
            if not ret or frame is None:
                print("⚠️ Kameradan anlik goruntu yakalanamadi, tekrar deneniyor...")
                continue

            # Yuz tespiti ve kirpma
            face_img, bbox = detector.detect_and_crop(frame)

            if bbox is not None:
                x, y, w, h = bbox
                count += 1
                
                img_path = str(user_dir / f"{user_name}_{count}.jpg")
                
                # Gri tonlamaya cevir (LBPH egitimi icin)
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Kaydetme islemi (Turkce karakter hatasina karsi onlem)
                _, buffer = cv2.imencode('.jpg', gray_face)
                with open(img_path, 'wb') as f:
                    f.write(buffer)
                
                print(f"📸 Fotoğraf {count}/{max_count} kaydedildi.")

                if SHOW_DISPLAY:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.imshow("Veri Toplama Paneli", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                # Yuz bulunamazsa CPU'yu bosa yormamak icin kisa mola
                print("🔍 Yüz aranıyor...")
                time.sleep(0.1)

    finally:
        cam.release()
        print(f"✅ Islem tamamlandi. Veriler burada: data/faces/{user_name}")

def main_menu():
    while True:
        users = get_registered_users()
        print("\n" + "="*40)
        print("🛡️  Pi-FaceID YÖNETİM PANELİ (RPICAM) 🛡️")
        print("="*40)
        if not users:
            print("⚠️ Kayitli kimse yok. | 1-Yeni Ekle | 3-Cikis")
        else:
            print(f"👥 Kayitli Kisiler: {', '.join(users)}")
            print("1-Yeni Ekle | 2-Guncelle | 3-Cikis")
        
        secim = input("\nSecim: ").strip()
        if secim == "1":
            name = input("Kaydedilecek Isim: ").strip()
            if name: collect_data(name, mode="ekle")
        elif secim == "2" and users:
            print("\nGuncellenecek kisiyi secin:")
            for i, u in enumerate(users, 1): print(f"{i}- {u}")
            u_secim = input("No: ").strip()
            if u_secim.isdigit() and int(u_secim) <= len(users):
                collect_data(users[int(u_secim)-1], mode="guncelle")
        elif secim == "3":
            print("Gorusuruz kanka, sistem kapaniyor! 👋")
            break

if __name__ == "__main__":
    main_menu()