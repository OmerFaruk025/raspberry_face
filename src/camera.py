import cv2

class Camera:
    def __init__(self, source=0):
        """
        source: 0 (yerel kamera) veya 'http://LAPTOP_IP:5000/video' (IP Stream)
        """
        # Raspberry Pi'de bazen 0 çalışmazsa 1, 2 denenebilir 
        # ama biz burayı senin laptop IP'ne göre güncelleyeceğiz.
        self.cap = cv2.VideoCapture(source)
        
        # Buffer (tampon) ayarı - Gecikmeyi (lag) önlemek için Pi'de önemli
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        self.cap.release()