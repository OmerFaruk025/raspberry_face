import cv2
import time

class Camera:
    def __init__(self, source=0):
        # CAP_V4L2 yerine bazen default (0) daha stabil kalabilir, 
        # ama Pi üzerinde V4L2 en hızlısıdır.
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        
        # Kameranın kendine gelmesi için 2 saniye mola (Kritik!)
        print("⏳ Kamera uyandiriliyor...")
        time.sleep(2)
        
        # Çözünürlüğü sabitle
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Buffer boyutunu 1 yapıyoruz ki eski kare birikmesin (Lag önleyici)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        # Kameradan veri gelene kadar 3 kez dene
        for _ in range(3):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return ret, frame
            time.sleep(0.1)
        return False, None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()