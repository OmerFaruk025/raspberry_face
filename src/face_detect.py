import cv2

class FaceDetector:
    def __init__(self):
        # OpenCV'nin klasik yüz bulma algoritmasını (Haar Cascade) yüklüyoruz
        # Bu xml dosyası kütüphane ile otomatik gelir, yolunu böyle çekmek en garantisidir.
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_and_crop(self, frame):
        """
        Gelen karedeki yüzü bulur ve kırpılmış halini döndürür.
        """
        # Algoritmanın daha hızlı ve doğru çalışması için gri tonlamaya çeviriyoruz
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri ara
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        # Eğer yüz bulamazsa hiçbir şey döndürme
        if len(faces) == 0:
            return None, None

        # Birden fazla yüz varsa, en büyük olanı (genelde en yakındaki) seçelim
        # (x, y) koordinat, (w, h) ise genişlik ve yükseklik
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Orijinal kareden sadece yüzün olduğu bölgeyi kes
        face_img = frame[y:y+h, x:x+w]

        return face_img, (x, y, w, h)