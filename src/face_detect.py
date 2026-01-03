import cv2

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, img):
        # İşlem için griye çeviriyoruz ama...
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        (x, y, w, h) = faces[0]
        
        # ...Burada görüntüyü RENKLİ olarak kesip gönderiyoruz (collect_faces.py öyle bekliyor)
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        
        return face_img, [x, y, w, h]