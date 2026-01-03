import cv2

class FaceDetector:
    def __init__(self):
        # OpenCV'nin içinde hazır gelen yüz bulma dosyasını yüklüyoruz
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def find_faces(self, img):
        # Yüz bulmak için görüntüyü griye çevirmek performansı artırır
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri ara
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        face_results = []
        for (x, y, w, h) in faces:
            # Diğer kodların (collect_faces.py vb.) bozulmaması için 
            # MediaPipe formatında bir sonuç döndürüyoruz
            face_results.append({
                'box': [x, y, w, h],
                'center': (x + w//2, y + h//2)
            })
        return face_results