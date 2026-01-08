import cv2
import os

class FaceDetector:
    def __init__(self):
        # Cascade dosyasının yolunu bul
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_and_crop(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor 1.1: Görüntüyü %10 adımlarla tarar (daha detaylı)
        # minNeighbors 4: Onay mekanizmasını biraz gevşettik (hızlı algılama)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4, 
            minSize=(60, 60)
        )

        if len(faces) > 0:
            # En büyük yüzü al
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_img = frame[y:y+h, x:x+w]
            return face_img, (x, y, w, h)
        
        return None, None