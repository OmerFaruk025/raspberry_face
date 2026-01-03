import cv2

class FaceDetector:
    def __init__(self):
        # OpenCV'nin hazır yüz bulma modelini yüklüyoruz
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Yüzleri bul
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        # İlk bulunan yüzü al (en büyük olanı)
        (x, y, w, h) = faces[0]
        
        # Yüzü kırp ve 200x200 boyutuna getir (LBPH için standart boyuttur)
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        
        return face_img, [x, y, w, h]