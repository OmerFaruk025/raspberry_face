import cv2
import os

class FaceDetector:
    def __init__(self):
        # Mevcut sistem yollarından cascade'i bul
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
            raise RuntimeError("Haar cascade bulunamadı. sudo apt install opencv-data")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Cascade yüklenemedi.")

        # Göz cascade (doğrulama için)
        eye_path = cascade_path.replace('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml')
        if os.path.exists(eye_path):
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
        else:
            self.eye_cascade = None

        # Parametreler (ince ayar için)
        self.scaleFactor = 1.1
        self.minNeighbors = 4
        self.minSize = (80, 80)  # yüzün minimum boyutu
        self.aspect_ratio_min = 0.6
        self.aspect_ratio_max = 1.6

        print(f"✅ Haar Cascade yüklendi → {cascade_path}")

    def detect_and_crop(self, frame):
        """
        Gelen BGR frame'den yüz bulup kırpılmış yüz ve bbox döndürür.
        Eğer geçerli yüz yoksa (güvenlik doğrulamadan geçmez) -> (None, None)
        """
        if frame is None:
            return None, None

        # 1) Gri + eşitleme (kontrast artırma)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # 2) Yüz bul (daha küçük scaleFactor => daha hassas, ama ağır)
        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )

        if len(faces) == 0:
            return None, None

        # 3) Filtrele: aspect ratio ve area'ya göre uygunsa seç
        candidates = []
        h_img, w_img = gray.shape
        for (x, y, w, h) in faces:
            ar = w / float(h) if h > 0 else 0
            area = w * h
            # yüzün görüntü içinde aşırı aşağıda (gövde) olmadığından emin olalım:
            if y > h_img * 0.85:  # çok alçakta ise reddet
                continue
            if ar < self.aspect_ratio_min or ar > self.aspect_ratio_max:
                continue
            candidates.append((x, y, w, h, area))

        if not candidates:
            # fallback: eğer hiçbir candidate yoksa en büyük yüzü yine deneriz
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        else:
            # en büyük alana göre seç
            x, y, w, h, _ = sorted(candidates, key=lambda c: c[4], reverse=True)[0]

        # 4) Göz doğrulaması (opsiyonel, varsa kullan)
        if self.eye_cascade is not None:
            face_gray = gray_eq[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            if len(eyes) == 0:
                # Eğer göz yoksa muhtemelen yanlış tespit -> reddet
                return None, None

        # 5) Kırpma güvenlik sınırları
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        face_img = frame[y:y+h, x:x+w]

        return face_img, (x, y, w, h)
