import cv2
import subprocess
import numpy as np
import os

class Camera:
    def __init__(self, source=0):
        print("🛡️ rpicam-apps (libcamera) modu aktif!")
        # Kameranın hazır olup olmadığını küçük bir testle anlıyoruz
        try:
            subprocess.run(["rpicam-hello", "--timeout", "1"], check=True, capture_output=True)
            print("✅ Kamera baglantisi basarili.")
        except:
            print("❌ HATA: rpicam-hello calismadi. Kabloyu kontrol et kanka!")

    def read(self):
        """
        rpicam-still kullanarak anlik bir kare yakalar ve OpenCV formatina donusturur.
        """
        try:
            # -n: pencere acma, -t: bekleme süresi, -e: format, -o -: çıktıyı standart out'a ver
            cmd = ["rpicam-still", "-n", "-t", "10", "-e", "jpg", "-o", "-"]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                # Standart çıktıdan gelen veriyi numpy dizisine (resme) çevir
                data = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Pi'yi yormamak için 640x480'e indirgeyelim
                    frame = cv2.resize(frame, (640, 480))
                    return True, frame
            return False, None
        except Exception as e:
            print(f"⚠️ Goruntu yakalama hatasi: {e}")
            return False, None

    def release(self):
        # Alt süreç (subprocess) kullandığımız için kapatılacak bir nesne yok
        print("📸 Kamera serbest birakildi.")
        cv2.destroyAllWindows()