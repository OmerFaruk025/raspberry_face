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
RUNNING_ON_PI = True  # <--- Pi'de çalıştırırken True, Laptopta çalıştırırken False yap kanka!
LAPTOP_IP = "192.168.1.47"
STREAM_URL = f"http://{LAPTOP_IP}:5000/video"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)

def get_registered_users():
    """Kayıtlı kullanıcıları listeler."""
    return [d.name for d in DATA_PATH.iterdir() if d.is_dir()]

def collect_data(user_name, mode="ekle"):
    """Yüz verisi toplama ana fonksiyonu."""
    user_dir = DATA_PATH / user_name
    
    if mode == "guncelle":
        print(f"🔄 '{user_name}' verileri güncelleniyor (eskiler siliniyor)...")
        if user_dir.exists():
            shutil.rmtree(user_dir)
    
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Kaynak seçimi: Pi'deysek Streamer'a, Laptopta isek kameraya bağlan
    source = STREAM_URL if RUNNING_ON_PI else 0
    print(f"🌐 Bağlanılan kaynak: {source}")
    
    cam = Camera(source=source)
    detector = FaceDetector()
    
    count = 0
    max_count = 50 
    
    print(f"📸 Hazırlan kral! {user_name} için kayıt başlıyor.")
    print("İpucu: Kafanı yavaşça sağa sola, yukarı aşağı hareket ettir.")
    time.sleep(2)

    try:
        while count < max_count:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("⚠️ Görüntü alınamıyor, kaynak bağlantısını kontrol et!")
                break

            face_img, bbox = detector.detect_and_crop(frame)

            if bbox is not None:
                x, y, w, h = bbox
                count += 1
                
                img_path = str(user_dir / f"{user_name}_{count}.jpg")
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Türkçe karakter dostu kayıt (Numpy üzerinden)
                _, buffer = cv2.imencode('.jpg', gray_face)
                with open(img_path, 'wb') as f:
                    f.write(buffer)
                
                # Görsel Geri Bildirim
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"KAYIT: {count}/{max_count}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                print(f"🚀 Fotoğraf {count} alındı...")
                
                # Eğer Pi'de SSH ile çalışıyorsan imshow bazen hata verebilir. 
                # Hata alırsan aşağıdaki 2 satırı yorum satırı yapabilirsin.
                cv2.imshow("Veri Toplama Paneli", frame)
                cv2.waitKey(200) # Poz değiştirmek için süre tanı
            else:
                cv2.imshow("Veri Toplama Paneli", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print(f"✅ İşlem bitti! '{user_name}' klasörü güncel.")

# -----------------------------
# ANA ARAYÜZ (MENU)
# -----------------------------
def main_menu():
    while True:
        users = get_registered_users()
        
        print("\n" + "="*30)
        print("🛡️  Pi-FaceID YÖNETİM PANELİ  🛡️")
        print("="*30)
        
        if not users:
            print("⚠️ Sistemde henüz kayıtlı kimse yok.")
            print("1 - Yeni Kişi Ekle")
            print("3 - Çıkış")
        else:
            print(f"👥 Kayıtlı Kişiler: {', '.join(users)}")
            print("1 - Yeni Kişi Ekle")
            print("2 - Kişi Güncelle (Verileri Sil ve Yenile)")
            print("3 - Çıkış")
        
        secim = input("\nSeçiminiz: ").strip()

        if secim == "1":
            name = input("Yeni kişinin ismi: ").strip()
            if not name:
                print("❌ İsim boş olamaz!")
            elif name in users:
                print(f"❌ '{name}' zaten kayıtlı! Güncellemeyi seç kanka.")
            else:
                collect_data(name, mode="ekle")
        
        elif secim == "2" and users:
            print("\nGüncellenecek kişiyi seçin:")
            for i, u in enumerate(users, 1):
                print(f"{i} - {u}")
            
            u_secim = input("Kişi no (İptal için '0'): ").strip()
            if u_secim != "0" and u_secim.isdigit() and int(u_secim) <= len(users):
                target_user = users[int(u_secim)-1]
                collect_data(target_user, mode="guncelle")
            else:
                print("İptal edildi.")

        elif secim == "3":
            print("Sistemden çıkılıyor... Görüşürüz kral! 👋")
            break
        else:
            print("❌ Geçersiz seçim, tekrar dene.")

if __name__ == "__main__":
    main_menu()