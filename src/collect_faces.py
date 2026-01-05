import cv2
import os
import time
import numpy as np
import shutil
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR & KONFİGÜRASYON
# -----------------------------
RUNNING_ON_PI = True  # Pi'de laptop kamerası kullanıyorsan True kalsın
LAPTOP_IP = "192.168.1.47"
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

BASE_DIR = Path(__file__).resolve().parent.parent
# ARTIK VERİLER FACES KLASÖRÜNE GİDİYOR
DATA_PATH = BASE_DIR / "faces"
DATA_PATH.mkdir(exist_ok=True)

def get_registered_users():
    return [d.name for d in DATA_PATH.iterdir() if d.is_dir()]

def collect_data(user_name, mode="ekle"):
    user_dir = DATA_PATH / user_name
    
    if mode == "guncelle":
        print(f"🔄 '{user_name}' verileri temizleniyor...")
        if user_dir.exists():
            shutil.rmtree(user_dir)
    
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Akıllı kaynak ve ekran seçimi
    if RUNNING_ON_PI:
        source = STREAM_URL
        show_display = False
        print(f"🌐 MOD: Raspberry Pi | 🖥️ EKRAN: Kapalı")
    else:
        source = 0
        show_display = True
        print(f"🌐 MOD: Laptop | 🖥️ EKRAN: Açık")
    
    cam = Camera(source=source)
    detector = FaceDetector()
    
    count = 0
    max_count = 50 
    
    print(f"📸 Kayıt başlıyor: {user_name}")
    time.sleep(2)

    try:
        while count < max_count:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("⚠️ Görüntü kesildi!")
                break

            face_img, bbox = detector.detect_and_crop(frame)

            if bbox is not None:
                x, y, w, h = bbox
                count += 1
                
                img_path = str(user_dir / f"{user_name}_{count}.jpg")
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Kayıt işlemi
                _, buffer = cv2.imencode('.jpg', gray_face)
                with open(img_path, 'wb') as f:
                    f.write(buffer)
                
                print(f"🚀 Fotoğraf {count}/{max_count} kaydedildi.")

                if show_display:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.imshow("Veri Toplama Paneli", frame)
                    if cv2.waitKey(200) & 0xFF == ord('q'): break
                else:
                    time.sleep(0.2)
            else:
                if show_display:
                    cv2.imshow("Veri Toplama Paneli", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cam.release()
        if show_display:
            cv2.destroyAllWindows()
        print(f"✅ '{user_name}' kaydı faces/ klasörüne tamamlandı.")

def main_menu():
    while True:
        users = get_registered_users()
        print("\n" + "="*30)
        print("🛡️  Pi-FaceID YÖNETİM PANELİ  🛡️")
        print("="*30)
        if not users:
            print("⚠️ Kayıt yok. | 1-Ekle | 3-Çıkış")
        else:
            print(f"👥 Kayıtlılar: {', '.join(users)}")
            print("1-Ekle | 2-Güncelle | 3-Çıkış")
        
        secim = input("\nSeçim: ").strip()
        if secim == "1":
            name = input("İsim: ").strip()
            if name: collect_data(name, mode="ekle")
        elif secim == "2" and users:
            print("\nGüncellenecek kişi:")
            for i, u in enumerate(users, 1): print(f"{i}- {u}")
            u_secim = input("No: ").strip()
            if u_secim.isdigit() and int(u_secim) <= len(users):
                collect_data(users[int(u_secim)-1], mode="guncelle")
        elif secim == "3":
            print("Görüşürüz kral! 👋")
            break

if __name__ == "__main__":
    main_menu()