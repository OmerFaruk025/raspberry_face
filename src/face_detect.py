import cv2

class FaceDetector:
    def __init__(self):
        # Debian / Raspberry Pi için SABİT ve DOĞRU yol
        cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Güvenlik kontrolü (ÖNEMLİ)
        if self.face_cascade.empty():
            raise RuntimeError(
                "❌ Haar Cascade yüklenemedi! "
                "Dosya yolu yanlış veya opencv-data eksik."
            )

    def detect(self, gray_frame):
        """
        Gri görüntü alır, yüzleri döndürür
        return: [(x, y, w, h), ...]
        """
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return faces
