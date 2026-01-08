import cv2

class Camera:
    def __init__(self, source=0):
        # CAP_V4L2 ile hızlı bağlanıyoruz
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        
        # Kare boyutunu 480p yapalım (Yüz tanıma için en ideal ve hızlı boyut)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS'i 30'a sabitleyelim ki Pi nefes alsın
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            # Görüntüyü hafifçe keskinleştirmek tanımayı hızlandırır (opsiyonel)
            return True, frame
        return False, None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()