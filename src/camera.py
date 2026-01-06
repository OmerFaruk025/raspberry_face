import cv2

class Camera:
    def __init__(self, source=0):
        """
        Kanka source kısmına ya 0 (yerel kamera) 
        ya da "http://IP:5000/video" (stream) verdik.
        """
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"❌ HATA: Kamera kaynağına ({source}) bağlanılamadı!")

    def read(self):
        # Kameradan bir kare oku
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        #kapat
        self.cap.release()
        cv2.destroyAllWindows()