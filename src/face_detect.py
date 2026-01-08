import cv2
import os

class FaceDetector:
    def __init__(self):
        # Debian / Raspberry Pi için sabit ve güvenilir yol
        cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            raise RuntimeError(f"❌ Haar Cascade bulunamadı: {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("❌ Haar Cascade yüklenemedi!")

        print(f"✅ Haar Cascade yüklendi → {cascade_path}")

    def detect_and_crop(self, frame):
        """
        Frame içinde yüz bulur.
        - face_img: kırpılmış yüz
        - bbox: (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None, None

        # En büyük yüzü al
        x, y, w, h = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True
        )[0]

        face_img = frame[y:y+h, x:x+w]
        return face_img, (x, y, w, h)
