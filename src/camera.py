import cv2

class Camera:
    def __init__(self, source=0):
        """
        Kanka PiCam V2.1'e erişmek için cv2.CAP_V4L2 ekledik.
        Bu, yeni nesil Pi'lerde OpenCV'nin kamerayı yakalamasını sağlar.
        """
        # --- BURASI KRİTİK DEĞİŞİKLİK ---
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        
        # Çözünürlük ayarları (PiCam V2.1 için 640x480 idealdir)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print(f"❌ HATA: PiCam donanımına (source:{source}) ulaşılamadı!")
            print("İpucu: Kablo yönünü ve 'libcamera-hello' komutunu kontrol et kanka.")

    def read(self):
        # Kameradan bir kare oku
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        # Kamerayı serbest bırak
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()