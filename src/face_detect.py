import cv2
import os

class FaceDetector:
    def __init__(self):
        # Haar cascade yolu (sisteme göre otomatik bulur)
        possible_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        ]

        cascade_path = None
        for p in possible_paths:
            if os.path.exists(p):
                cascade_path = p
                break

        if cascade_path is None:
            raise RuntimeError("❌ Haar Cascade bulunamadı (opencv-data eksik olabilir)")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("❌ Haar Cascade yüklenemedi")

        print(f"✅ Haar Cascade yüklendi → {cascade_path}")

        # Parametreler (denge noktası)
        self.scaleFactor = 1.1
        self.minNeighbors = 4
        self.minSize = (80, 80)

    def detect_and_crop(self, frame):
        if frame is None:
            return None, None

        h_img, w_img = frame.shape[:2]

        # --- GRI + KONTRAST ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )

        if len(faces) == 0:
            return None, None

        # En büyük yüzü al (en yakındaki)
        x, y, w, h = sorted(
            faces, key=lambda f: f[2] * f[3], reverse=True
        )[0]

        # -----------------------------
        # 🔥 CROP DÜZELTME (ÇENE FIX)
        # -----------------------------
        expand_down = int(h * 0.25)   # çene + sakal
        expand_up   = int(h * 0.05)   # alnı kesmesin
        expand_lr   = int(w * 0.05)   # yanaklar

        x2 = max(0, x - expand_lr)
        y2 = max(0, y - expand_up)

        w2 = min(w_img - x2, w + 2 * expand_lr)
        h2 = min(h_img - y2, h + expand_down + expand_up)

        face_img = frame[y2:y2 + h2, x2:x2 + w2]

        # Son güvenlik
        if face_img.size == 0:
            return None, None

        return face_img, (x2, y2, w2, h2)
