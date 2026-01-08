import cv2
import os

class FaceDetector:
    def __init__(self):
        possible_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        ]

        cascade_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                break

        if cascade_path is None:
            raise RuntimeError(
                "❌ Haar Cascade bulunamadı.\n"
                "Çözüm: sudo apt install opencv-data"
            )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("❌ Cascade dosyası bozuk veya okunamadı")

        print(f"✅ Haar Cascade yüklendi → {cascade_path}")

    def detect(self, gray_frame):
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )
