import cv2
import subprocess
import numpy as np

class Camera:
    def __init__(self, source=0):
        print("🛡️ rpicam-apps modu aktif!")

    def read(self):
        try:
            # -t 10: Işık ayarı için bekleme, direkt çek.
            # --immediate: Deklanşöre anında bas.
            cmd = ["rpicam-still", "-n", "-t", "10", "-e", "jpg", "-o", "-", "--immediate"]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                data = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
            return False, None
        except:
            return False, None

    def release(self):
        cv2.destroyAllWindows()