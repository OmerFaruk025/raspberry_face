import cv2
import time
import numpy as np
from picamera2 import Picamera2


class Camera:
    def __init__(self, source=0):
        """
        PiCam versiyonu.
        source parametresi artık kullanılmıyor ama
        eski kodlar bozulmasın diye korunuyor.
        """

        self.picam2 = Picamera2()

        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        time.sleep(1)  # Kamera stabilizasyonu

    def read(self):
        try:
            frame = self.picam2.capture_array()
            # Picamera2 RGB verir → OpenCV BGR ister
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            print(f"❌ Kamera okuma hatası: {e}")
            return False, None

    def release(self):
        self.picam2.stop()
        cv2.destroyAllWindows()
