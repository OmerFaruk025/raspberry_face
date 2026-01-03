import cv2
from camera import Camera
from face_detect import FaceDetector
from collections import deque

# Kanka buraya laptopunun IP'sini yazıyoruz
# Eğer lokal kamera kullanacaksan sadece 0 yazabilirsin
LAPTOP_IP = "192.168.1.47" # Bunu kendi laptop IP'nle değiştir kanka
stream_url = f"http://{LAPTOP_IP}:5000/video"

# Kamera nesnesi (Artık IP stream'e hazır)
cam = Camera(source=stream_url)

# MediaPipe yüz dedektörü
detector = FaceDetector(confidence=0.6)

# Yüz daha önce görülmüş mü?
face_visible = False

FACE_SIZE = (160, 160)
bbox_history = deque(maxlen=5)

while True:
    # Kameradan bir frame al
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Görüntü alınamadı, stream bekleniyor...")
        continue

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Kanka burada yeni eklediğimiz detect metodunu kullanıyoruz
    bbox = detector.detect(rgb, frame.shape)

    if bbox:
        x, y, bw, bh = bbox

        # ---- GÜVENLİK KONTROLLERİ ----
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        
        # Ortalama bbox hesapla (Titremeyi engeller)
        bbox_history.append((x, y, bw, bh))

        x = int(sum(b[0] for b in bbox_history) / len(bbox_history))
        y = int(sum(b[1] for b in bbox_history) / len(bbox_history))
        bw = int(sum(b[2] for b in bbox_history) / len(bbox_history))
        bh = int(sum(b[3] for b in bbox_history) / len(bbox_history))

        # ---- FACE CROP ----
        face_crop = frame[y:y+bh, x:x+bw]

        # ---- DEBUG AMAÇLI ÇİZİM ----
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        # ---- LOG (SADECE İLK ALGILAMADA) ----
        if not face_visible:
            print(f"YÜZ ALGILANDI → x:{x}, y:{y}, w:{bw}, h:{bh}")
            face_visible = True
            
        if face_crop.size != 0:
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, FACE_SIZE)
            cv2.imshow("Face Crop (Resized)", resized_face)
    else:
        bbox_history.clear()
        face_visible = False

    # Ana kamera görüntüsünü göster
    cv2.imshow("Pi-FaceID | Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Temizlik
cam.release()
cv2.destroyAllWindows()