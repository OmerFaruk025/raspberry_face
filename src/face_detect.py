import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, confidence=0.6):
        # MediaPipe face detection modeli
        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence
        )

    def detect_and_crop(self, frame):
        """
        Yüz algılar, crop'lar ve döner
        return:
          face_img -> kırpılmış yüz (None olabilir)
          bbox     -> (x, y, w, h)
        """

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

            # taşmaları engelle
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+bh, x:x+bw]

            if face_img.size == 0:
                return None, None

            return face_img, (x, y, bw, bh)

        return None, None
