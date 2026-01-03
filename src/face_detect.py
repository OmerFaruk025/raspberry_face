import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, confidence=0.6):
        # MediaPipe face detection modeli
        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(
            model_selection=0, # 2 metre içindeki yüzler için ideal (Laptop/Pi mesafesi)
            min_detection_confidence=confidence
        )

    def detect_and_crop(self, frame):
        """
        Yüz algılar, crop'lar ve döner
        return:
          face_img -> kırpılmış yüz (None olabilir)
          bbox     -> (x, y, w, h)
        """
        if frame is None:
            return None, None

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face.process(rgb)

        if results.detections:
            detection = results.detections[0]
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            # Taşmaları engelle (Frame sınırları dışına çıkma)
            x, y = max(0, x), max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            face_img = frame[y:y+bh, x:x+bw]

            if face_img.size == 0:
                return None, None

            return face_img, (x, y, bw, bh)

        return None, None

    # main.py ile uyumluluk için takma isim (Shortcut)
    def detect(self, rgb_frame, shape):
        """main.py'nin beklediği format için eklendi"""
        # Sadece koordinatları (bbox) döndürür
        h, w = shape[0], shape[1]
        results = self.face.process(rgb_frame)
        if results.detections:
            box = results.detections[0].location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)
            return (x, y, bw, bh)
        return None