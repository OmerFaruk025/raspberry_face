import cv2

class Camera:
    def __init__(self, source=0):
        """
        Kanka artık laptopu aradan çıkardık. 
        source=0 doğrudan Pi'ye takılı olan PiCam V2.1'i temsil eder.
        """
        self.cap = cv2.VideoCapture(source)
        
        # PiCam V2.1 için ideal çözünürlük ayarları (Pi'yi yormaz)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print(f"❌ HATA: PiCam donanımına (source:{source}) ulaşılamadı!")
            print("İpucu: 'libcamera-hello' ile kamerayı test etmeyi unutma kanka.")

    def read(self):
        # Kameradan anlık kareyi çek
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        # Kamerayı serbest bırak ve pencereleri kapat
        self.cap.release()
        cv2.destroyAllWindows()