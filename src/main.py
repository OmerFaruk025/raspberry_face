import cv2
from camera import Camera
from face_detect import FaceDetector
from collections import deque

# Kamera nesnesi (OpenCV sadece görüntü alıyor)
cam = Camera()

# MediaPipe yüz dedektörü
detector = FaceDetector(confidence=0.6)

# Yüz daha önce görülmüş mü?
# Spam log basmamak için state tutuyoruz
face_visible = False

FACE_SIZE = (160, 160)#LBPH için uygun bir ölçü
bbox_history = deque(maxlen=5)

while True:
    # Kameradan bir frame al
    ret, frame = cam.read()
    if not ret:
        break  # kamera kapandıysa çık

    h, w, _ = frame.shape  # Frame boyutlarını al (yükseklik, genişlik)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV BGR verir, MediaPipe RGB ister
    
    bbox = detector.detect(rgb, frame.shape)   # Yüz tespiti → bbox (x, y, w, h) veya None

    if bbox:
        x, y, bw, bh = bbox

        # ---- GÜVENLİK KONTROLLERİ ----
        # Bounding box frame dışına taşmasın
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        
        #------------------------------------#
        # Ortalama bbox hesapla
        bbox_history.append(bbox)

        xs = [b[0] for b in bbox_history]
        ys = [b[1] for b in bbox_history]
        ws = [b[2] for b in bbox_history]
        hs = [b[3] for b in bbox_history]

        x = int(sum(xs) / len(xs))
        y = int(sum(ys) / len(ys))
        bw = int(sum(ws) / len(ws))
        bh = int(sum(hs) / len(hs))

        # ---- FACE CROP ----

        face_crop = frame[y:y+bh, x:x+bw]

        # ---- DEBUG AMAÇLI ÇİZİM ----
        # Kamerada yüzün etrafına dikdörtgen çiz
        cv2.rectangle(
            frame,
            (x, y),
            (x + bw, y + bh),
            (0, 255, 0),
            2
        )
        
        
        
        # ---- LOG (SADECE İLK ALGILAMADA) ----
        if not face_visible:
            
            print(f"YÜZ ALGILANDI → x:{x}, y:{y}, w:{bw}, h:{bh}")
            face_visible = True
            
        if face_crop.size !=0:
            
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)    # ---- GRİ TON ----
            resized_face = cv2.resize(gray_face, FACE_SIZE)   # ---- RESIZE(TÜm Fotoğraflar Aynı Boyuta geliyor) ----
            cv2.imshow("Face Crop (Resized)", resized_face)
        

    else:
        # Yüz kaybolduysa state sıfırla
        bbox_history.clear()
        face_visible = False

    # Ana kamera görüntüsünü göster
    #!!!!!! cv2.imshow("Pi-FaceID | Camera", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):# q tuşuna basılırsa çık
        break
    

# Temizlik
cam.release()
cv2.destroyAllWindows()
