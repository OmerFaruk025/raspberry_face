import cv2

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_and_crop(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor'ü 1.05 yaptık (Hassas tarama)
        # minNeighbors'u 3 yaptık (Hızlı algılama)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(30, 30)
        )

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            return face_img, (x, y, w, h)
        
        return None, None