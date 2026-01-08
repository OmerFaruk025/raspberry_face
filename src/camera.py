import cv2
import subprocess
import numpy as np
import time

class Camera:
    def __init__(self, source=0):
        print("🛡️ rpicam-apps (libcamera) HIZLANDIRILMIS mod aktif!")
        try:
            # Kamerayı test et
            subprocess.run(["rpicam-hello", "--timeout", "1"], check=True, capture_output=True)
            print("✅ Kamera baglantisi basarili.")
        except:
            print("❌ HATA: Kamera bulunamadi!")

    def read(self):
        try:
            # -t 50: Işık ayarı için sadece 50ms bekle (Hız için kritik)
            # --immediate: Deklanşöre hemen bas
            # --denoise cdn_off: İşlemciyi yormamak için gürültü filtresini kapat
            cmd = [
                "rpicam-still", "-n", "-t", "50", "-e", "jpg", 
                "-o", "-", "--immediate", "--denoise", "cdn_off"
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                data = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Tanıma hızı için görüntüyü küçük tutuyoruz
                    frame = cv2.resize(frame, (640, 480))
                    return True, frame
            return False, None
        except Exception as e:
            print(f"⚠️ Goruntu hatasi: {e}")
            return False, None

    def release(self):
        cv2.destroyAllWindows()