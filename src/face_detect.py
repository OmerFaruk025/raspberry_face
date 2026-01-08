import cv2
import os

class FaceDetector:
    def __init__(self):
        base_paths = [
            "/usr/share/opencv4/haarcascades/",
            "/usr/share/opencv/haarcascades/",
            "/usr/local/share/opencv4/haarcascades/"
        ]

        face_path = None
        eye_path = None

        for p in base_paths:
            if os.path.exists(p + "haarcascade_frontalface_default.xml"):
                face_path = p + "haarcascade_frontalface_default.xml"
                eye_path = p + "haarcascade_eye.xml"
                break

        if not face_path:
            raise RuntimeError("Haar cascade bulunamadı")

        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

        print(f"✅ Haar yüklendi → {face_path}")

        self.scaleFactor = 1.1
        self.minNeighbors = 4
        self.minSize = (90, 90)

    def detect_and_crop(self, frame):
        if frame is None:
            return None, None

        h_img, w_img = frame.shape[:2]

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

        # En büyükten küçüğe sırala
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        for (x, y, w, h) in faces:
            # --- GÖZ DOĞRULAMA ---
            face_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )

            if len(eyes) < 2:
                continue  # ❌ tek göz / yanlış yüz

            # --- CROP DÜZELTME ---
            expand_down = int(h * 0.25)
            expand_up   = int(h * 0.05)
            expand_lr   = int(w * 0.05)

            x2 = max(0, x - expand_lr)
            y2 = max(0, y - expand_up)
            w2 = min(w_img - x2, w + 2 * expand_lr)
            h2 = min(h_img - y2, h + expand_down + expand_up)

            face_img = frame[y2:y2 + h2, x2:x2 + w2]

            if face_img.size == 0:
                continue

            return face_img, (x2, y2, w2, h2)

        return None, None
